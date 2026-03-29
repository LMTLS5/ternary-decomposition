"""
Greedy Ternary Matrix Decomposition

Decomposes a matrix A into A ≈ B * diag(D) * C
where B and C contain only {-1, 0, +1} values.

Two algorithms are provided:
  - greedy_ternary_decomposition: Heuristic thresholding (fast, practical)
  - greedy_ternary_decomposition_projection: Optimal ternary projection (higher quality)
"""

import torch
import time


def greedy_ternary_decomposition(A: torch.Tensor, k_components: int, threshold_scale: float = 0.7):
    """
    Greedy ternary decomposition using adaptive mean-threshold binarization.

    Decomposes A (m x n) into B (m x k), D (k,), C (k x n) such that:
        A ≈ (B.float() * D) @ C.float()

    Args:
        A:               Input matrix (m x n), any float dtype.
        k_components:    Number of ternary components to compute.
        threshold_scale: Fraction of mean absolute value used as ternary threshold.
                         Lower = denser vectors; higher = sparser. Default 0.7.

    Returns:
        B (torch.int8, m x k), D (torch.float32, k,), C (torch.int8, k x n)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    R = A.clone().to(device, dtype=torch.float32)
    m, n = R.shape

    B = torch.zeros((m, k_components), dtype=torch.int8, device=device)
    C = torch.zeros((k_components, n), dtype=torch.int8, device=device)
    D_diag = torch.zeros(k_components, dtype=torch.float32, device=device)

    start_time = time.time()
    for i in range(k_components):
        # Initialize with a random ternary vector
        u_cont = torch.randn(m, device=device, dtype=torch.float32)
        thresh_u = threshold_scale * torch.mean(torch.abs(u_cont))
        u_ternary = torch.sign(u_cont) * (torch.abs(u_cont) > thresh_u).float()

        for _ in range(15):
            # Given ternary u, find optimal ternary v
            v_cont = torch.matmul(R.T, u_ternary)
            thresh_v = threshold_scale * torch.mean(torch.abs(v_cont))
            v_ternary = torch.sign(v_cont) * (torch.abs(v_cont) > thresh_v).float()
            if torch.all(v_ternary == 0):
                idx = torch.argmax(torch.abs(v_cont))
                v_ternary[idx] = torch.sign(v_cont[idx])

            # Given ternary v, find optimal ternary u
            u_cont = torch.matmul(R, v_ternary)
            thresh_u = threshold_scale * torch.mean(torch.abs(u_cont))
            u_new = torch.sign(u_cont) * (torch.abs(u_cont) > thresh_u).float()
            if torch.all(u_new == 0):
                idx = torch.argmax(torch.abs(u_cont))
                u_new[idx] = torch.sign(u_cont[idx])

            if torch.all(u_ternary == u_new):
                break
            u_ternary = u_new

        # Compute scalar scale d
        norm_sq = torch.sum(u_ternary ** 2) * torch.sum(v_ternary ** 2)
        d = torch.dot(u_ternary, torch.matmul(R, v_ternary)) / (norm_sq + 1e-9)

        # Deflate residual
        R -= d * torch.outer(u_ternary, v_ternary)

        B[:, i] = u_ternary.to(torch.int8)
        C[i, :] = v_ternary.to(torch.int8)
        D_diag[i] = d

        if (i + 1) % 100 == 0:
            elapsed = time.time() - start_time
            print(f"  Component {i+1}/{k_components} | Residual Norm: {torch.norm(R, p='fro').item():.4f} | {elapsed:.2f}s")
            start_time = time.time()

    return B.cpu(), D_diag.cpu(), C.cpu()


def _optimal_ternary_projection(u: torch.Tensor) -> torch.Tensor:
    """
    Project a continuous vector u onto the nearest ternary vector
    (in a least-squares sense) by finding the optimal sparsity level.
    """
    abs_u = torch.abs(u)
    sorted_vals, sorted_idx = torch.sort(abs_u, descending=True)
    s_range = torch.arange(1, u.shape[0] + 1, device=u.device, dtype=torch.float32)
    f = torch.cumsum(sorted_vals, dim=0) / torch.sqrt(s_range)
    s_star = torch.argmax(f).item() + 1

    u_tern = torch.zeros_like(u)
    if torch.all(abs_u == 0):
        return u_tern
    u_tern[sorted_idx[:s_star]] = torch.sign(u[sorted_idx[:s_star]])
    return u_tern


def greedy_ternary_decomposition_projection(A: torch.Tensor, k_components: int):
    """
    Greedy ternary decomposition using optimal ternary projection.

    Uses a closed-form optimal projection at each alternating step, which
    produces sparser, higher-quality components than the heuristic version.

    Args:
        A:            Input matrix (m x n).
        k_components: Number of ternary components to compute.

    Returns:
        B (torch.int8, m x k), D (torch.float32, k,), C (torch.int8, k x n)
    """
    device = A.device
    R = A.clone().to(dtype=torch.float32)
    m, n = R.shape

    B = torch.zeros((m, k_components), dtype=torch.int8, device=device)
    C = torch.zeros((k_components, n), dtype=torch.int8, device=device)
    D_diag = torch.zeros(k_components, dtype=torch.float32, device=device)

    start_time = time.time()
    for i in range(k_components):
        u_ternary = _optimal_ternary_projection(torch.randn(m, device=device, dtype=torch.float32))

        for _ in range(15):
            target_v = torch.matmul(R.T, u_ternary) / (torch.sum(u_ternary ** 2) + 1e-9)
            v_ternary = _optimal_ternary_projection(target_v)

            target_u = torch.matmul(R, v_ternary) / (torch.sum(v_ternary ** 2) + 1e-9)
            u_new = _optimal_ternary_projection(target_u)

            if torch.all(u_ternary == u_new):
                break
            u_ternary = u_new

        norm_sq = torch.sum(u_ternary ** 2) * torch.sum(v_ternary ** 2)
        d = torch.dot(u_ternary, torch.matmul(R, v_ternary)) / (norm_sq + 1e-9)

        R -= d * torch.outer(u_ternary, v_ternary)

        B[:, i] = u_ternary.to(torch.int8)
        C[i, :] = v_ternary.to(torch.int8)
        D_diag[i] = d

        if (i + 1) % 100 == 0:
            elapsed = time.time() - start_time
            print(f"  Proj Component {i+1}/{k_components} | Residual Norm: {torch.norm(R, p='fro').item():.4f} | {elapsed:.2f}s")
            start_time = time.time()

    return B.cpu(), D_diag.cpu(), C.cpu()


def energy_preserved(A: torch.Tensor, B: torch.Tensor, D: torch.Tensor, C: torch.Tensor) -> float:
    """Compute the fraction of matrix energy (Frobenius norm squared) preserved by the decomposition."""
    A_hat = (B.float() * D) @ C.float()
    return 1.0 - (torch.norm(A.float() - A_hat) ** 2 / torch.norm(A.float()) ** 2).item()


if __name__ == "__main__":
    torch.manual_seed(42)
    m, n, k = 512, 512, 128
    A = torch.randn(m, n)
    print(f"Matrix: {m}x{n}, k={k}\n")

    print("Heuristic Thresholding:")
    B, D, C = greedy_ternary_decomposition(A, k)
    print(f"  Energy preserved: {energy_preserved(A, B, D, C):.4f}\n")

    print("Optimal Projection:")
    B, D, C = greedy_ternary_decomposition_projection(A, k)
    print(f"  Energy preserved: {energy_preserved(A, B, D, C):.4f}")
