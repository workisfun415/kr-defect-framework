"""
core.py — K–R defect operator and Hessian recovery.

Mathematical definitions (from the paper):

  D_f(x, h; K, R) = K·f(x) + R·f(x+h) - f(x+R·h)          Eq.(1)

  Φ_f(x, h; K, R) = D_f / (K·R·‖h‖²)                        Eq.(2)

  K+R = 1,   K > 0,   R > 0                          (Admissible set A)

  Theorem 4.1 (Curvature Recovery):
    Φ_f(x, sv; K, R) = ½ v^T H_f(ξ) v + E
    |E| ≤ ω_H(s) / (2δ)  for  K ≥ δ

  Theorem 4.2 (Local Limit):
    lim_{s→0} Φ_f(x, sv; K, R) = ½ v^T H_f(x) v
    uniformly on A_δ = {K ≥ δ > 0}
"""

import numpy as np
from typing import Callable, Optional, Union


# ─────────────────────────────────────────────────────────────────────────────
# Low-level scalar functions
# ─────────────────────────────────────────────────────────────────────────────

def kr_defect(
    f: Callable,
    x: np.ndarray,
    h: np.ndarray,
    K: float,
) -> float:
    """
    Compute the unnormalised K–R defect D_f(x, h; K, R).

    D_f = K·f(x) + R·f(x+h) - f(x+R·h)

    Parameters
    ----------
    f : callable
        Function f : R^n → R. Called as f(x) where x is a 1-D array.
    x : np.ndarray, shape (n,)
        Base evaluation point.
    h : np.ndarray, shape (n,)
        Displacement vector. Must satisfy x+h ∈ Ω.
    K : float
        Weight in (0,1). R is computed as 1-K.

    Returns
    -------
    float
        The defect D_f(x, h; K, 1-K).

    Notes
    -----
    The interior sampling point is z = x + R·h = K·x + R·(x+h),
    which lies strictly inside the segment [x, x+h] (Lemma 2.1).
    This keeps all three evaluation points inside the domain when
    Ω is convex and x, x+h ∈ Ω. See Theorem 4.7 (Boundary Interior).
    """
    R = 1.0 - K
    return K * f(x) + R * f(x + h) - f(x + R * h)


def kr_phi(
    f: Callable,
    x: np.ndarray,
    h: np.ndarray,
    K: float,
) -> float:
    """
    Compute the normalised K–R defect Φ_f(x, h; K, R).

    Φ_f = D_f / (K·R·‖h‖²)

    This is the directional curvature estimator.  As ‖h‖ → 0,
    Φ_f → ½ v^T H_f(x) v  where v = h/‖h‖.
    (Theorem 4.2, Local Limit).

    Parameters
    ----------
    f : callable
        Function f : R^n → R.
    x : np.ndarray
        Base point.
    h : np.ndarray
        Displacement vector (h ≠ 0).
    K : float
        Weight in (0,1).

    Returns
    -------
    float
        The normalised defect.

    Raises
    ------
    ValueError
        If h is zero or K ∉ (0,1).
    """
    if K <= 0.0 or K >= 1.0:
        raise ValueError(f"K must be in (0,1), got {K}")
    h_sq = float(np.dot(h, h))
    if h_sq == 0.0:
        raise ValueError("Displacement h must be non-zero.")
    R = 1.0 - K
    D = K * f(x) + R * f(x + h) - f(x + R * h)
    return D / (K * R * h_sq)


# ─────────────────────────────────────────────────────────────────────────────
# KRHessian — full Hessian recovery at a single point
# ─────────────────────────────────────────────────────────────────────────────

class KRHessian:
    """
    Recover the full Hessian matrix H_f(x) at a single point x.

    Uses Algorithm 1 from the paper:
    - Diagonal entries  : 2 × Φ_f(x, s·eᵢ; K, R)
    - Off-diagonal i<j  : 2 × Φ_f(x, s·vᵢⱼ; K, R) − ½ Hᵢᵢ − ½ Hⱼⱼ
      where vᵢⱼ = (eᵢ + eⱼ) / √2  (polarisation identity, Lemma 2.6)

    Total measurements: n + C(n,2) = n(n+1)/2 = dim(Sym_n(R)).
    This is both necessary and sufficient (Theorem 4.4).

    Parameters
    ----------
    f : callable
        Function f : R^n → R. Called as f(array of shape (n,)).
    step : float, optional
        Step size s > 0. Default 0.05. Smaller s → closer to true
        Hessian but noisier if f has measurement noise.
    n_kvals : int, optional
        Number of K values to average over. Default 9 (K = 0.1, …, 0.9).
        Set to 1 (K = 0.5 only) for speed; set to 9 for noise reduction.
    k_min : float, optional
        Minimum K value for averaging range. Default 0.1 (= δ).
        Uniformity of Theorem 4.2 requires K ≥ δ > 0.

    Examples
    --------
    >>> import numpy as np
    >>> from kr_defect import KRHessian
    >>> f = lambda x: x[0]**2 + 3*x[1]**2 + x[0]*x[1]
    >>> kr = KRHessian(f, step=0.05)
    >>> H = kr.compute(np.array([0.0, 0.0]))
    >>> print(H)   # expect [[2, 1], [1, 6]]
    """

    def __init__(
        self,
        f: Callable,
        step: float = 0.05,
        n_kvals: int = 9,
        k_min: float = 0.1,
    ):
        self.f       = f
        self.step    = float(step)
        self.n_kvals = int(n_kvals)
        self.k_min   = float(k_min)

        if n_kvals < 1:
            raise ValueError("n_kvals must be >= 1")
        if step <= 0:
            raise ValueError("step must be > 0")
        if not (0.0 < k_min < 0.5):
            raise ValueError("k_min must be in (0, 0.5)")

        # Build K-value grid (symmetric around 0.5)
        if n_kvals == 1:
            self._kvals = np.array([0.5])
        else:
            self._kvals = np.linspace(k_min, 1.0 - k_min, n_kvals)

    def _phi_averaged(self, x: np.ndarray, h: np.ndarray) -> float:
        """Average Φ_f over the K grid for noise reduction."""
        return float(np.mean([kr_phi(self.f, x, h, K) for K in self._kvals]))

    def compute(
        self,
        x: np.ndarray,
        step: Optional[float] = None,
    ) -> np.ndarray:
        """
        Compute the Hessian H_f(x).

        Parameters
        ----------
        x : np.ndarray, shape (n,)
            Point at which to estimate the Hessian.
        step : float, optional
            Override the default step size for this call only.

        Returns
        -------
        np.ndarray, shape (n, n)
            Symmetric matrix approximating H_f(x).
            Entry-wise error ≤ ω_{H_f}(s) / 2  by eq. (error_bound).
        """
        x = np.asarray(x, dtype=float)
        s = float(step) if step is not None else self.step
        n = len(x)
        H = np.zeros((n, n))

        # ── Diagonal entries: (H_f)_ii = 2 × Φ_f(x, s·eᵢ; K,R) ──────────
        for i in range(n):
            ei = np.zeros(n)
            ei[i] = 1.0
            H[i, i] = 2.0 * self._phi_averaged(x, s * ei)

        # ── Off-diagonal: polarisation identity (Lemma 2.6) ─────────────────
        for i in range(n):
            for j in range(i + 1, n):
                ei = np.zeros(n); ei[i] = 1.0
                ej = np.zeros(n); ej[j] = 1.0
                vij = (ei + ej) / np.sqrt(2.0)
                phi_ij = 2.0 * self._phi_averaged(x, s * vij)
                H[i, j] = phi_ij - 0.5 * H[i, i] - 0.5 * H[j, j]
                H[j, i] = H[i, j]

        return H

    def curvature(
        self,
        x: np.ndarray,
        v: np.ndarray,
        step: Optional[float] = None,
    ) -> float:
        """
        Estimate the directional curvature ½ v^T H_f(x) v.

        This is the fundamental K–R measurement (Theorem 4.2).
        Uses only 3 function evaluations per K-value.

        Parameters
        ----------
        x : np.ndarray
            Base point.
        v : np.ndarray
            Direction vector (will be normalised).
        step : float, optional
            Step size override.

        Returns
        -------
        float
            Estimate of ½ v^T H_f(x) v.
        """
        x = np.asarray(x, dtype=float)
        v = np.asarray(v, dtype=float)
        v = v / np.linalg.norm(v)
        s = float(step) if step is not None else self.step
        return self._phi_averaged(x, s * v)

    def n_evaluations(self, n: int) -> int:
        """Total function evaluations for a full Hessian in dimension n."""
        return 3 * self.n_kvals * (n * (n + 1) // 2)

    def __repr__(self) -> str:
        return (
            f"KRHessian(step={self.step}, n_kvals={self.n_kvals}, "
            f"k_min={self.k_min})"
        )
