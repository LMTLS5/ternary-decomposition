# Ternary Matrix Decomposition

Decomposes a matrix **A** into a ternary low-rank factorization:

```
A ≈ B · diag(D) · C
```

where **B** and **C** contain only `{-1, 0, +1}` values and **D** is a real-valued diagonal scale vector. This is useful for compressing LLM weight matrices to near-binary representations while preserving most of the original information.

## Algorithms

Two algorithms are provided:

| Algorithm | File | Description |
|---|---|---|
| Greedy Thresholding | `ternary_decomposition.py` | Fast alternating greedy search with adaptive mean-threshold binarization |
| Optimal Projection | `ternary_decomposition.py` | Alternating greedy search with closed-form optimal ternary projection (sparser, higher quality) |
| Optimization-Based | `ternary_optimization.py` | Gradient descent with L1 + boundary penalties, initialized from SVD |

## Usage

```python
import torch
from ternary_decomposition import greedy_ternary_decomposition, energy_preserved

A = torch.randn(1024, 1024)
k = 256  # number of components

B, D, C = greedy_ternary_decomposition(A, k_components=k)
print(f"Energy preserved: {energy_preserved(A, B, D, C):.4f}")

# Reconstruct
A_hat = (B.float() * D) @ C.float()
```

### Apply to a weight file

```bash
# From a .npy file
python example.py --file weights.npy --k 256

# From a safetensors file
python example.py --file model.safetensors --key model.layers.0.mlp.up_proj.weight --save
```

## Install

```bash
pip install -r requirements.txt
```

## Files

- `ternary_decomposition.py` — greedy ternary decomposition (two variants)
- `ternary_optimization.py` — optimization-based ternary factorization
- `example.py` — CLI for decomposing weight matrices from `.npy` or `.safetensors`
