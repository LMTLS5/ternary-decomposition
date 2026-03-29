"""
Example: applying ternary decomposition to an LLM weight matrix.

Usage:
    python example.py --file weights.npy
    python example.py --file model.safetensors --key transformer.h.0.mlp.c_fc.weight
"""

import argparse
import torch
import numpy as np

from ternary_decomposition import greedy_ternary_decomposition, greedy_ternary_decomposition_projection, energy_preserved


def load_matrix(path: str, key: str | None = None) -> torch.Tensor:
    if path.endswith(".npy"):
        return torch.from_numpy(np.load(path)).float()
    elif path.endswith(".safetensors"):
        from safetensors import safe_open
        with safe_open(path, framework="pt", device="cpu") as f:
            if key is None:
                keys = list(f.keys())
                key = keys[0]
                print(f"No key specified — using: {key}")
            return f.get_tensor(key).float()
    else:
        raise ValueError(f"Unsupported format: {path}. Use .npy or .safetensors")


def main():
    parser = argparse.ArgumentParser(description="Ternary decomposition of a weight matrix")
    parser.add_argument("--file", required=True, help="Path to .npy or .safetensors file")
    parser.add_argument("--key", default=None, help="Tensor key (for safetensors)")
    parser.add_argument("--k", type=int, default=None, help="Number of components (default: min(m,n)//2)")
    parser.add_argument("--method", choices=["threshold", "projection"], default="threshold",
                        help="Decomposition algorithm to use")
    parser.add_argument("--save", action="store_true", help="Save B, D, C to .npy files")
    args = parser.parse_args()

    A = load_matrix(args.file, args.key)
    m, n = A.shape
    k = args.k or min(m, n) // 2

    print(f"Matrix: {m}x{n} | Components k={k} | Method: {args.method}")

    if args.method == "threshold":
        B, D, C = greedy_ternary_decomposition(A, k)
    else:
        B, D, C = greedy_ternary_decomposition_projection(A, k)

    e = energy_preserved(A, B, D, C)
    sparsity = 1.0 - (torch.count_nonzero(B) + torch.count_nonzero(C)).item() / (B.numel() + C.numel())
    print(f"\nEnergy preserved: {e:.4f}")
    print(f"Sparsity (B+C):   {sparsity:.2%}")

    if args.save:
        np.save("ternary_B.npy", B.numpy())
        np.save("ternary_D.npy", D.numpy())
        np.save("ternary_C.npy", C.numpy())
        print("Saved: ternary_B.npy, ternary_D.npy, ternary_C.npy")


if __name__ == "__main__":
    main()
