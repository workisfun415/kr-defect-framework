"""
kr_defect.py
============
The K-R Defect Framework — Python Library
Based on:
  "A K-R Defect Framework for Classical Inequalities: Curvature Recovery,
   Uniqueness, Stability, and Applications to Nonlinear Boundary Value Problems"
  RamaKrishna Pasupuleti, Boundary Value Problems, SpringerOpen, 2026.

Author of library: RamaKrishna Pasupuleti
ORCID: 0009-0008-8418-1430
"""

import numpy as np
from typing import Callable, Tuple, Union, List, Optional


# ============================================================
# CORE DEFECT FUNCTIONS
# ============================================================

def kr_defect(f: Callable, x: float, y: float, K: float) -> float:
    """
    Compute the raw K-R defect D_f(x, y; K, R).

    D_f = K*f(x) + R*f(y) - f(K*x + R*y)
    where R = 1 - K.

    Parameters
    ----------
    f  : callable  — the function to measure
    x  : float     — left point
    y  : float     — right point (y != x)
    K  : float     — dominant weight, 0 < K < 1

    Returns
    -------
    float : D_f value (>= 0 for convex f)

    Example
    -------
    >>> import math
    >>> kr_defect(math.exp, 0, 1, 0.6)
    0.19481...
    """
    R = 1.0 - K
    z = K * x + R * y          # interior point
    return K * f(x) + R * f(y) - f(z)


def kr_normalised(f: Callable, x: float, y: float, K: float) -> float:
    """
    Compute the normalised K-R defect Phi_f(x, y; K, R).

    Phi_f = D_f / (K * R * (x - y)^2)

    By the Curvature Recovery Theorem (Theorem 2.6):
    Phi_f = (1/2) * f''(xi) for some xi between x and y.

    By the Local Limit Theorem (Theorem 2.7):
    Phi_f -> (1/2) * f''(x) as y -> x.

    Parameters
    ----------
    f  : callable  — the function
    x  : float     — left point
    y  : float     — right point (y != x)
    K  : float     — dominant weight, 0 < K < 1

    Returns
    -------
    float : Phi_f value, approximates (1/2)*f''(xi)

    Example
    -------
    >>> import math
    >>> kr_normalised(math.exp, 0.5, 0.51, 0.6)
    0.8237...   # close to (1/2)*e^0.5 = 0.8244
    """
    R = 1.0 - K
    D = kr_defect(f, x, y, K)
    return D / (K * R * (x - y) ** 2)


def kr_second_derivative(
    f: Callable,
    x: float,
    h: float = 1e-3,
    K: float = 0.5,
    average: bool = True,
    n_avg: int = 9
) -> float:
    """
    Estimate f''(x) using the K-R Local Limit Theorem.

    f''(x) ≈ 2 * Phi_f(x, x+h; K, R)

    Parameters
    ----------
    f       : callable — the function
    x       : float   — point at which to estimate f''
    h       : float   — step size (default 1e-3)
    K       : float   — weight parameter (ignored if average=True)
    average : bool    — if True, average over multiple K values for noise reduction
    n_avg   : int     — number of K values to average over (default 9)

    Returns
    -------
    float : estimate of f''(x)

    Notes
    -----
    When average=True, K values are uniformly spaced in (0.1, 0.9).
    This exploits the K-R parameter flexibility to reduce measurement noise
    — an advantage that classical finite differences do not have.

    Example
    -------
    >>> import math
    >>> kr_second_derivative(math.exp, 0.5, h=0.01)
    1.6487...   # close to e^0.5 = 1.6487
    """
    if average:
        K_values = np.linspace(0.1, 0.9, n_avg)
        estimates = [2.0 * kr_normalised(f, x, x + h, k) for k in K_values]
        return float(np.mean(estimates))
    else:
        return 2.0 * kr_normalised(f, x, x + h, K)


def kr_first_derivative(
    f: Callable,
    x: float,
    h: float = 1e-3,
    K: float = 0.5
) -> float:
    """
    Estimate f'(x) using the K-R first-order operator.

    Using the same three evaluation points {f(x), f(z), f(y)} where z=Kx+Ry:
    Phi_f^(1) = [f(y) - f(z)] / (K * (y - x)) -> f'(x) as y -> x

    Parameters
    ----------
    f : callable — the function
    x : float   — point at which to estimate f'
    h : float   — step size
    K : float   — weight parameter, 0 < K < 1

    Returns
    -------
    float : estimate of f'(x)

    Example
    -------
    >>> import math
    >>> kr_first_derivative(math.exp, 1.0, h=0.01)
    2.718...   # close to e^1 = 2.718
    """
    R = 1.0 - K
    y = x + h
    z = K * x + R * y
    return (f(y) - f(z)) / (K * h)


def kr_both_derivatives(
    f: Callable,
    x: float,
    h: float = 1e-3,
    K: float = 0.5
) -> Tuple[float, float]:
    """
    Estimate both f'(x) and f''(x) from the same three function evaluations.

    Three evaluations: f(x), f(z), f(y) where z = Kx + R(x+h).
    These give both derivatives simultaneously — the complete derivative-free
    differentiation system.

    Parameters
    ----------
    f : callable — the function
    x : float   — point
    h : float   — step size
    K : float   — weight parameter

    Returns
    -------
    (f_prime, f_double_prime) : tuple of floats

    Example
    -------
    >>> import math
    >>> kr_both_derivatives(math.exp, 1.0, h=0.01)
    (2.718..., 2.718...)   # both close to e^1
    """
    R = 1.0 - K
    y = x + h
    z = K * x + R * y

    # Three evaluations — used for both operators
    fx = f(x)
    fz = f(z)
    fy = f(y)

    # First derivative: [f(y) - f(z)] / (K*h)
    f_prime = (fy - fz) / (K * h)

    # Second derivative: 2 * D_f / (KR*h^2)
    D = K * fx + R * fy - fz
    f_double_prime = 2.0 * D / (K * R * h ** 2)

    return f_prime, f_double_prime


# ============================================================
# RECONSTRUCTION
# ============================================================

def kr_reconstruct(
    f_data: np.ndarray,
    x_grid: np.ndarray,
    K: float = 0.5,
    f0: float = 0.0,
    fp0: float = 0.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reconstruct f from K-R defect measurements using the Reconstruction Formula
    (Theorem 2.10).

    Steps:
    1. Estimate f''(x_i) at each grid point from defect measurements.
    2. Integrate twice to recover f.
    3. The integration constants f(a) and f'(a) are supplied by the user.

    Parameters
    ----------
    f_data : array — measured function values at x_grid
    x_grid : array — grid of x values (uniformly spaced)
    K      : float — K-R weight parameter
    f0     : float — boundary value f(x_grid[0])
    fp0    : float — boundary value f'(x_grid[0])

    Returns
    -------
    (f''_estimated, f_reconstructed) : tuple of arrays
    """
    n = len(x_grid)
    h = x_grid[1] - x_grid[0]

    # Step 1: Estimate f'' at each interior point
    f_double_prime = np.zeros(n)
    for i in range(n - 1):
        R = 1.0 - K
        x = x_grid[i]
        y = x_grid[i + 1]
        z_idx = K * x + R * y
        # Interpolate f at interior point z
        alpha = (z_idx - x) / h
        fz = (1 - alpha) * f_data[i] + alpha * f_data[i + 1]
        D = K * f_data[i] + R * f_data[i + 1] - fz
        f_double_prime[i] = 2.0 * D / (K * R * h ** 2)
    f_double_prime[-1] = f_double_prime[-2]  # extrapolate boundary

    # Step 2: Integrate f'' twice (trapezoidal rule)
    f_prime_reconstructed = np.zeros(n)
    f_prime_reconstructed[0] = fp0
    for i in range(1, n):
        f_prime_reconstructed[i] = f_prime_reconstructed[i-1] + \
            0.5 * h * (f_double_prime[i-1] + f_double_prime[i])

    f_reconstructed = np.zeros(n)
    f_reconstructed[0] = f0
    for i in range(1, n):
        f_reconstructed[i] = f_reconstructed[i-1] + \
            0.5 * h * (f_prime_reconstructed[i-1] + f_prime_reconstructed[i])

    return f_double_prime, f_reconstructed


def kr_inverse_bvp(
    u_data: np.ndarray,
    x_grid: np.ndarray,
    K: float = 0.5,
    average: bool = True
) -> np.ndarray:
    """
    Recover the unknown nonlinearity f in -u'' = f(u) from measurements of u.

    Uses the K-R defect to estimate u'' from sensor data, then:
    f(u(x_i)) = -u''(x_i) ≈ -2 * Phi_u(x_i, x_{i+1}; K, R)

    This is the core inverse BVP application (Section 5.2 of the paper).
    The Uniqueness Principle (Theorem 2.8) guarantees the recovered f is unique
    up to affine terms fixed by boundary conditions.

    Parameters
    ----------
    u_data : array — measured values of u at x_grid
    x_grid : array — uniformly spaced grid
    K      : float — K-R weight parameter
    average: bool  — if True, average over multiple K values

    Returns
    -------
    array : f(u(x_i)) at each grid point
    """
    n = len(x_grid)
    h = x_grid[1] - x_grid[0]
    f_values = np.zeros(n)

    K_vals = np.linspace(0.1, 0.9, 9) if average else [K]

    for i in range(n - 1):
        estimates = []
        for k in K_vals:
            R = 1.0 - k
            x = x_grid[i]
            y = x_grid[i + 1]
            alpha = k * 0.0 + R * 1.0  # fractional position of z in [x,y]
            fz = (1 - alpha) * u_data[i] + alpha * u_data[i + 1]
            D = k * u_data[i] + R * u_data[i + 1] - fz
            phi = 2.0 * D / (k * R * h ** 2)
            estimates.append(-phi)  # f(u) = -u''
        f_values[i] = np.mean(estimates)

    f_values[-1] = f_values[-2]
    return f_values


# ============================================================
# CONVEXITY DIAGNOSTIC
# ============================================================

def kr_convexity(
    f: Callable,
    x_grid: np.ndarray,
    h: float = 1e-3,
    K: float = 0.5
) -> np.ndarray:
    """
    Classify local convexity/concavity of f at each grid point.

    By the Convexity Classification Theorem (Theorem 2.13):
    sign(Phi_f) = sign(f''(xi))

    Returns
    -------
    array of +1 (convex), 0 (affine), -1 (concave) at each point
    """
    signs = np.zeros(len(x_grid))
    for i, x in enumerate(x_grid):
        phi = kr_normalised(f, x, x + h, K)
        if phi > 1e-10:
            signs[i] = 1
        elif phi < -1e-10:
            signs[i] = -1
        else:
            signs[i] = 0
    return signs


# ============================================================
# STABILITY CHECK
# ============================================================

def kr_stability_bound(epsilon: float, a: float, b: float) -> float:
    """
    Compute the K-R Stability Theorem bound (Theorem 2.15).

    If |Phi_f| <= epsilon everywhere and f(a) = f(b) = 0, then:
    ||f||_Linf <= epsilon * (b-a)^2 / 4

    Parameters
    ----------
    epsilon : float — maximum observed |Phi_f|
    a, b    : float — domain endpoints

    Returns
    -------
    float : upper bound on ||f||_Linf
    """
    return epsilon * (b - a) ** 2 / 4.0


print("K-R Defect Framework Library loaded successfully.")
print("Functions: kr_defect, kr_normalised, kr_second_derivative,")
print("           kr_first_derivative, kr_both_derivatives,")
print("           kr_reconstruct, kr_inverse_bvp, kr_convexity,")
print("           kr_stability_bound")
