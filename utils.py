"""
beam.py — Euler–Bernoulli beam bending moment recovery.

Application from Section 5.1 of the paper and Experiment 2 (Table 3).

The Euler–Bernoulli beam BVP:
  EI u''''(x) = q(x),   u(0)=u(L)=u'(0)=u'(L)=0

The bending moment is:
  M(x) = EI u''(x)

Given displacement measurements u(xᵢ) at sensor positions xᵢ ∈ [0,L],
K–R recovers M(xᵢ) from three evaluations:
  M(xᵢ) ≈ 2 × Φ_u(xᵢ, xᵢ+h; K, R)

This works at xᵢ = 0.01 (near the left support), where central FD
needs u(-h) < 0, which is outside the beam domain [0, L].
"""

import numpy as np
from typing import Callable, Optional
from .core import KRHessian


def beam_bending_moment(
    u_func: Callable,
    x_positions: np.ndarray,
    L: float = 1.0,
    EI: float = 1.0,
    step: float = 0.04,
    n_kvals: int = 9,
) -> dict:
    """
    Recover bending moments EI·u''(x) at given sensor positions.

    Uses the K–R framework (3 evaluations per position), which is valid
    even near supports where central FD requires evaluations outside [0,L].

    Parameters
    ----------
    u_func : callable
        Displacement function u : [0,L] → R.
        Called as u_func(np.array([x])) returning a scalar.
    x_positions : np.ndarray, shape (N,)
        Sensor positions within [0, L].
    L : float
        Beam length. Default 1.0.
    EI : float
        Flexural rigidity. Default 1.0.
    step : float
        K–R step size h. Default 0.04.
        Must satisfy xᵢ + h ≤ L for all positions.
    n_kvals : int
        K-values for averaging. Default 9.

    Returns
    -------
    dict with keys:
        'x'         : np.ndarray — sensor positions
        'M_kr'      : np.ndarray — K–R bending moment estimates
        'fd_valid'  : np.ndarray of bool — whether central FD is applicable
        'M_fd'      : np.ndarray — central FD estimates (NaN if not applicable)
        'evals_kr'  : int — total K–R function evaluations
        'evals_fd'  : int — total FD function evaluations (for valid positions)

    Examples
    --------
    >>> import numpy as np
    >>> from kr_defect import beam_bending_moment
    >>> # Exact solution for q=1, EI=1, L=1
    >>> u_exact = lambda x: x[0]**2 * (1-x[0])**2 / 24.0
    >>> x_pos = np.array([0.01, 0.02, 0.05, 0.10, 0.50])
    >>> result = beam_bending_moment(u_exact, x_pos)
    >>> print(result['M_kr'])    # near-support positions recovered
    """
    x_positions = np.asarray(x_positions, dtype=float)

    # 1D wrapper so KRHessian works (Hessian is scalar = u''(x))
    def u_1d(pt):
        return float(u_func(np.atleast_1d(pt)))

    kr = KRHessian(u_1d, step=step, n_kvals=n_kvals)

    M_kr  = np.zeros(len(x_positions))
    M_fd  = np.full(len(x_positions), np.nan)
    fd_ok = np.zeros(len(x_positions), dtype=bool)

    n_fd_evals = 0

    for i, xi in enumerate(x_positions):
        xpt = np.array([xi])

        # K–R: valid as long as xi + step ≤ L (interior points stay in [0,L])
        if xi + step <= L:
            H = kr.compute(xpt)
            M_kr[i] = EI * H[0, 0]
        else:
            # If step too large near right end, reduce it
            s_adj = (L - xi) * 0.9
            if s_adj > 1e-6:
                H = kr.compute(xpt, step=s_adj)
                M_kr[i] = EI * H[0, 0]
            else:
                M_kr[i] = np.nan

        # Central FD: needs xi - step ≥ 0
        if xi - step >= 0.0 and xi + step <= L:
            fd_ok[i] = True
            u_plus  = u_1d(np.array([xi + step]))
            u_mid   = u_1d(np.array([xi]))
            u_minus = u_1d(np.array([xi - step]))
            M_fd[i] = EI * (u_plus - 2.0*u_mid + u_minus) / step**2
            n_fd_evals += 3
        else:
            fd_ok[i] = False

    return {
        'x'        : x_positions,
        'M_kr'     : M_kr,
        'fd_valid' : fd_ok,
        'M_fd'     : M_fd,
        'evals_kr' : 3 * n_kvals * len(x_positions),
        'evals_fd' : n_fd_evals,
    }


def beam_exact_solution(x: float, q: float = 1.0, L: float = 1.0, EI: float = 1.0):
    """
    Exact solution of EI u'''' = q with clamped boundary conditions.

    u(x)   = q x² (L-x)² / (24 EI)
    u''(x) = q (L² - 6Lx + 6x²) / (12 EI)   [bending moment / EI]
    """
    u    = q * x**2 * (L - x)**2 / (24.0 * EI)
    u_pp = q * (L**2 - 6.0*L*x + 6.0*x**2) / (12.0 * EI)
    return {'u': u, 'u_pp': u_pp, 'M': EI * u_pp}
