@software
{pasupuleti2026kr_code,
  author    = {Pasupuleti, RamaKrishna},
  title     = {K-R Defect Framework: Python Library and Benchmarks},
  version   = {v1.0.1},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.21339639},
  url       = {https://doi.org/10.5281/zenodo.21339639}
}
# K-R Defect Framework — Python Library

Based on the published paper:
> "A K-R Defect Framework for Classical Inequalities: Curvature Recovery,
> Uniqueness, Stability, and Applications to Nonlinear Boundary Value Problems"
> RamaKrishna Pasupuleti, Boundary Value Problems, SpringerOpen, 2026.

## Quick Start

```python
import math
from kr_defect import kr_second_derivative, kr_both_derivatives

# Estimate f''(0.5) for f(x) = e^x
est = kr_second_derivative(math.exp, x=0.5, h=0.01)
print(f"K-R estimate: {est:.4f}")   # ~1.6487
print(f"True value:   {math.exp(0.5):.4f}")

# Get BOTH f'(x) and f''(x) from three evaluations
fp, fpp = kr_both_derivatives(math.exp, x=1.0, h=0.01)
```

## Functions

| Function | Description |
|---|---|
| `kr_defect(f,x,y,K)` | Raw K-R defect D_f |
| `kr_normalised(f,x,y,K)` | Normalised defect Φ_f = (1/2)f″(ξ) |
| `kr_second_derivative(f,x,h,K)` | Estimate f″(x), optional averaging |
| `kr_first_derivative(f,x,h,K)` | Estimate f′(x) |
| `kr_both_derivatives(f,x,h,K)` | Both f′ and f″ from 3 evaluations |
| `kr_reconstruct(data,grid,K,f0,fp0)` | Recover f from defect measurements |
| `kr_inverse_bvp(u_data,x_grid,K)` | Recover f in -u″=f(u) from u data |
| `kr_convexity(f,x_grid,h,K)` | Classify convex/concave at each point |
| `kr_stability_bound(eps,a,b)` | Stability Theorem bound |

## Key Advantages Over Finite Differences

- Works near domain boundaries (all evaluation points stay interior)
- Parameter averaging over K ∈ (0,1) reduces noise sensitivity
- Uniqueness Principle guarantees unique reconstruction
- Stability Theorem gives explicit error bound

## Run Benchmarks

```bash
python3 benchmarks.py
python3 generate_plots.py
```
