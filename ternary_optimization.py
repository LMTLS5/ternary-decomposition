"""
Optimization-Based Ternary Matrix Factorization

Finds a low-rank ternary factorization A ≈ U * diag(S) * V^T
where U and V are constrained to {-1, 0, +1} values.

Uses gradient descent with two penalty terms to encourage ternary solutions:
  - L1 penalty: promotes sparsity
  - Boundary penalty x²(1-|x|)²: pushes values toward {-1, 0, +1}
"""

import torch
import torch.nn as nn
import torch.optim as optim


def ternary_matrix_factorization(
    A: torch.Tensor,
    k: int,
    epochs: int = 1000,
    lr: float = 0.005,
    lambda_l1: float = 0.01,
    lambda_boundary: float = 0.1,
    verbose: bool = True,
):
    """
    Optimization-based ternary matrix factorization.

    Factorizes A (m x n) as A ≈ (U * S) @ V^T where U ∈ {-1,0,1}^(m×k)
    and V ∈ {-1,0,1}^(n×k) after rounding, and S ∈ R^k is a diagonal scale.

    Initialized from the top-k SVD components, then refined by gradient descent
    with L1 and boundary penalties to push U, V toward ternary values.

    Args:
        A:               Input matrix (m x n) as a float32 tensor.
        k:               Number of ternary rank-1 components.
        epochs:          Number of gradient descent steps.
        lr:              Adam learning rate.
        lambda_l1:       Weight of L1 sparsity penalty on U and V.
        lambda_boundary: Weight of boundary penalty x²(1-|x|)² on U and V.
        verbose:         Print progress every 200 epochs.

    Returns:
        U_ternary (torch.int8, m x k),
        S (torch.float32, k,),
        V_ternary (torch.int8, n x k)
    """
    device = A.device
    m, n = A.shape

    # SVD-based initialization
    with torch.no_grad():
        U_full, S_full, Vh_full = torch.linalg.svd(A, full_matrices=False)
        rank = S_full.shape[0]

        U_init = torch.randn(m, k, device=device) * 0.1
        V_init = torch.randn(n, k, device=device) * 0.1
        Sigma_init = torch.zeros(k, device=device)

        num_svd = min(k, rank)
        U_init[:, :num_svd] = U_full[:, :num_svd]
        V_init[:, :num_svd] = Vh_full.T[:, :num_svd]
        Sigma_init[:num_svd] = S_full[:num_svd]

    U_var = nn.Parameter(U_init)
    V_var = nn.Parameter(V_init)
    Sigma_var = nn.Parameter(Sigma_init)
    optimizer = optim.Adam([U_var, V_var, Sigma_var], lr=lr)

    for epoch in range(epochs):
        optimizer.zero_grad()

        A_approx = torch.matmul(U_var * Sigma_var, V_var.T)
        loss_recon = torch.norm(A - A_approx, p='fro') ** 2
        loss_l1 = lambda_l1 * (torch.norm(U_var, p=1) + torch.norm(V_var, p=1))
        loss_boundary = lambda_boundary * (
            torch.sum(U_var ** 2 * (1 - torch.abs(U_var)) ** 2) +
            torch.sum(V_var ** 2 * (1 - torch.abs(V_var)) ** 2)
        )

        (loss_recon + loss_l1 + loss_boundary).backward()
        optimizer.step()

        with torch.no_grad():
            U_var.clamp_(-1.0, 1.0)
            V_var.clamp_(-1.0, 1.0)

        if verbose and (epoch + 1) % 200 == 0:
            sparsity = (torch.abs(U_var) < 0.1).float().mean().item()
            print(f"  Epoch {epoch+1:4d} | Recon Loss: {loss_recon.item():.4f} | Sparsity: {sparsity:.2%}")

    # Round to ternary and re-solve for optimal scale
    with torch.no_grad():
        U_ternary = torch.round(U_var)
        V_ternary = torch.round(V_var)

        Sigma_clean = torch.zeros(k, device=device)
        numerator = torch.diag(U_ternary.T @ A @ V_ternary)
        denominator = torch.sum(U_ternary ** 2, dim=0) * torch.sum(V_ternary ** 2, dim=0)
        valid = denominator > 0
        Sigma_clean[valid] = numerator[valid] / denominator[valid]

    return U_ternary.to(torch.int8).cpu(), Sigma_clean.cpu(), V_ternary.to(torch.int8).cpu()


def energy_preserved(A: torch.Tensor, U: torch.Tensor, S: torch.Tensor, V: torch.Tensor) -> float:
    """Fraction of matrix energy preserved: 1 - ||A - A_hat||² / ||A||²"""
    A_hat = (U.float() * S) @ V.float().T
    return 1.0 - (torch.norm(A.float() - A_hat) ** 2 / torch.norm(A.float()) ** 2).item()


if __name__ == "__main__":
    torch.manual_seed(42)
    m, n, k = 256, 256, 64
    A = torch.randn(m, n)
    print(f"Matrix: {m}x{n}, k={k}\n")

    U, S, V = ternary_matrix_factorization(A, k=k, epochs=1000)
    print(f"\nEnergy preserved: {energy_preserved(A, U, S, V):.4f}")
    sparsity = 1.0 - (torch.count_nonzero(U) + torch.count_nonzero(V)).item() / (U.numel() + V.numel())
    print(f"Sparsity (U+V):   {sparsity:.2%}")
