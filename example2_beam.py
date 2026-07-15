# kr-defect — Multivariable K–R Defect Framework

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://python.org)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21339639.svg)](https://doi.org/10.5281/zenodo.21339639)

**Hessian recovery from interior-only directional measurements.**

Works at structural supports, organ walls, domain boundaries, scattered
sensor positions, and with missing data — settings where classical finite
differences are not directly applicable.

---

## Installation

```bash
pip install numpy        # only dependency
# Clone and install locally (before PyPI release):
git clone https://github.com/workisfun415/kr-defect-framework
cd kr-defect-framework
pip install -e .
```

---

## Quick Start (60 seconds)

```python
import numpy as np
from kr_defect import KRHessian

# Any function you can evaluate
f = lambda x: np.exp(x[0] + x[1])

# Create the estimator
kr = KRHessian(f, step=0.05, n_kvals=9)

# Recover the Hessian at any point
H = kr.compute(np.array([0.5, 0.5]))
print(H)
# [[1.649, 1.649],
#  [1.649, 1.649]]   ← true = e·[[1,1],[1,1]]
```

Uses exactly **3 function evaluations per direction** — all interior to the domain.

---

## The Three Things You Can Rely On

| Property | What it means | Theorem |
|---|---|---|
| **Interior-only stencil** | All evaluation points stay inside Ω — works near boundaries | 4.7 |
| **Uniqueness Principle** | Equal defect field ⟹ equal Hessian (no FD analogue) | 4.3 |
| **Stability Certificate** | ‖Φ_f‖ ≤ ε ⟹ ‖f‖_∞ ≤ C(Ω,n)·ε | 4.6 |

---

## Use Cases

### 1. Structural monitoring — bending moment near supports

```python
from kr_defect import beam_bending_moment

# Displacement sensor function (can be noisy measurements)
u_sensor = lambda x: float(x[0])**2 * (1-float(x[0]))**2 / 24.0

# Sensor positions — including right next to the support at x=0
x_sensors = np.array([0.01, 0.02, 0.05, 0.10, 0.50])

result = beam_bending_moment(u_sensor, x_sensors, step=0.04)

for x, M, fd_ok in zip(result['x'], result['M_kr'], result['fd_valid']):
    fd_str = "OK" if fd_ok else "NOT APPLICABLE"
    print(f"x={x:.3f}  M(x)={M:.5f}  FD: {fd_str}")

# Output:
# x=0.010  M(x)=0.06601  FD: NOT APPLICABLE   ← K–R works here!
# x=0.020  M(x)=0.06463  FD: NOT APPLICABLE   ← K–R works here!
# x=0.050  M(x)=0.04956  FD: OK
# x=0.100  M(x)=0.03184  FD: OK
# x=0.500  M(x)=-0.04004  FD: OK
```

### 2. Scattered sensor array

```python
from kr_defect import KRHessian

# Sensors at irregular positions
sensor_pts = np.random.uniform(0.1, 0.9, (500, 2))
f = lambda x: np.exp(x[0]) * np.sin(x[1])

kr = KRHessian(f, step=0.06, n_kvals=9)

# Estimate Hessian at each sensor position
hessians = [kr.compute(p) for p in sensor_pts]
print(f"Recovered {len(hessians)} Hessians")
print(f"Each required {kr.n_evaluations(2)} function evaluations")
```

### 3. Check stencil validity before computing

```python
from kr_defect import AnnulusDomain

domain = AnnulusDomain(r_in=0.3, r_out=0.9)
x_near_hole = np.array([0.31, 0.0])
h_vec = np.array([0.15, 0.0])

status = domain.check_stencil(x_near_hole, h_vec, K=0.5)
print(f"K–R valid:  {status['kr_valid']}")   # True
print(f"FD  valid:  {status['cfd_valid']}")  # False — x-h is in the hole
```

### 4. Noise robustness test

```python
from kr_defect.utils import monte_carlo_error

result = monte_carlo_error(
    f        = lambda x: np.exp(x[0]+x[1]),
    x        = np.array([0.5, 0.5]),
    H_true   = np.exp(1.0) * np.ones((2,2)),
    sigma    = 1e-3,
    n_trials = 1000,
    noise_type = 'gaussian',
)
print(f"Mean RMSE: {result['mean']:.4f} ± {result['std']:.4f}")
print(f"95% CI:    {result['ci_95']}")
```

---

## API Reference

### `KRHessian(f, step, n_kvals, k_min)`
Main class. Call `.compute(x)` to get the full n×n Hessian.

### `kr_phi(f, x, h, K)` → float
Single normalised K–R defect: Φ_f(x, h; K, R).

### `kr_defect(f, x, h, K)` → float
Unnormalised K–R defect: D_f(x, h; K, R).

### `beam_bending_moment(u_func, x_positions, ...)` → dict
Recover EI·u'' at sensor positions including near-support.

### `KRScattered(pts, f_vals, ...)` → `.compute(x)` → np.ndarray
Hessian from pre-measured scattered sensor values.

### `convergence_order(f, x, H_true, ...)` → dict
Empirical convergence order measurement.

### `monte_carlo_error(f, x, H_true, sigma, n_trials, ...)` → dict
Noise robustness: mean, std, 95% CI across trials.

### Domain classes
`BoxDomain`, `BallDomain`, `AnnulusDomain`, `RectangleWithHoleDomain`
— each has `.contains(x)` and `.check_stencil(x, h, K)`.

---

## Run the Examples

```bash
python examples/example1_quickstart.py   # basic Hessian recovery
python examples/example2_beam.py         # beam bending moments
python examples/example3_scattered.py    # scattered sensors + noise
```

## Run the Tests

```bash
python tests/test_core.py        # prints PASS/FAIL for each theorem
# or:
python -m pytest tests/ -v
```

---

## When to Use K–R vs Finite Differences

| Situation | Use K–R | Use FD |
|---|:---:|:---:|
| Near domain boundary | ✅ | ❌ |
| Scattered/irregular data | ✅ | ❌ |
| Missing sensor data | ✅ | ❌ |
| Non-convex domain (annulus) | ✅ | ❌ |
| Need uniqueness certificate | ✅ | ❌ |
| Smooth interior grid, best accuracy | — | ✅ |
| Lowest noise sensitivity | — | ✅ |
| Fastest runtime | — | ✅ |

---

## Citation

```bibtex
@article{Pasupuleti2026,
  author  = {Pasupuleti, RamaKrishna},
  title   = {Multivariable K--R Defect Theory: Hessian Recovery with
             Interior-Only Sampling, Uniqueness, and Applications to
             Boundary-Constrained Problems},
  journal = {Journal of Computational and Applied Mathematics},
  year    = {2026},
  note    = {Submitted},
  doi     = {10.5281/zenodo.21339639}
}
```

## License

MIT © 2026 RamaKrishna Pasupuleti
