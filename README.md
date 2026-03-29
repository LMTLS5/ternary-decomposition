# Ternary Matrix Decomposition

Decomposes a matrix **A** into a ternary low-rank factorization:

```
A ≈ B · diag(D) · C
```

where **B** and **C** contain only `{-1, 0, +1}` values and **D** is a real-valued diagonal scale vector.

This algorithm is designed for extreme compression of LLM weight matrices. In experiments on MLP blocks of SOTA LLMs, we achieve **>99% matrix energy preservation** with a rank approximately **2.5x** the original matrix rank.

## Algorithms

Two variants of greedy alternating optimization are provided in `ternary_decomposition.py`:

| Algorithm | Description |
|---|---|
| Greedy Thresholding | Fast alternating greedy search with adaptive mean-threshold binarization |
| Optimal Projection | Alternating greedy search with closed-form optimal ternary projection (sparser, higher quality) |

## Usage

```python
import torch
from ternary_decomposition import greedy_ternary_decomposition, energy_preserved

A = torch.randn(1024, 1024)
k = 2560  # e.g., 2.5x original rank

B, D, C = greedy_ternary_decomposition(A, k_components=k)
print(f"Energy preserved: {energy_preserved(A, B, D, C):.4f}")

# Reconstruct
A_hat = (B.float() * D) @ C.float()
```

### Apply to a weight file

```bash
# From a .npy file
python example.py --file weights.npy --k 2560

# From a safetensors file
python example.py --file model.safetensors --key model.layers.0.mlp.up_proj.weight --save
```

## Install

```bash
pip install -r requirements.txt
```

## Files

- `ternary_decomposition.py` — core greedy ternary decomposition algorithms
- `example.py` — CLI for decomposing weight matrices from `.npy` or `.safetensors`
