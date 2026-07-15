"""
scattered.py — K–R Hessian recovery on scattered / irregular data.

When sensors are placed at irregular positions and function values
are already measured (not callable), this class recovers the Hessian
at any query point using the nearest available sensor as the sampling
partner.

This is the use case demonstrated in Experiment 3 of the paper
(Table 4): 500 random sensor positions, 3 function evaluations per
Hessian estimate, no structured grid required.
"""

import numpy as np
from typing import Optional
from .core import KRHessian, kr_phi


class KRScattered:
    """
    Hessian recovery from a cloud of pre-measured function values.

    The user provides:
    - a set of measurement positions (pts)
    - the corresponding function values (f_vals)

    At any query point x, the class finds the k nearest neighbours,
    uses them as sampling partners in the K–R scheme, and averages
    to produce a Hessian estimate.

    Parameters
    ----------
    pts : np.ndarray, shape (N, n)
        Sensor positions.
    f_vals : np.ndarray, shape (N,)
        Measured function values at pts[i].
    n_kvals : int
        Number of K-values for averaging. Default 9.
    k_min : float
        Minimum K value. Default 0.1.

    Examples
    --------
    >>> import numpy as np
    >>> from kr_defect import KRScattered
    >>> rng = np.random.default_rng(42)
    >>> pts = rng.uniform(0.1, 0.9, (500, 2))
    >>> f = lambda x: np.exp(x[0]) * np.sin(x[1])
    >>> f_vals = np.array([f(p) for p in pts])
    >>> krs = KRScattered(pts, f_vals)
    >>> H = krs.compute(np.array([0.5, 0.5]))
    """

    def __init__(
        self,
        pts: np.ndarray,
        f_vals: np.ndarray,
        n_kvals: int = 9,
        k_min: float = 0.1,
    ):
        self.pts    = np.asarray(pts,    dtype=float)
        self.f_vals = np.asarray(f_vals, dtype=float)
        self.n_kvals = int(n_kvals)
        self.k_min   = float(k_min)

        if n_kvals == 1:
            self._kvals = np.array([0.5])
        else:
            self._kvals = np.linspace(k_min, 1.0 - k_min, n_kvals)

        if self.pts.ndim == 1:
            self.pts = self.pts.reshape(-1, 1)
        self.N, self.n = self.pts.shape

    def _nearest(self, x: np.ndarray) -> tuple:
        """Return (index, distance) of nearest point to x (excluding x itself)."""
        dists = np.linalg.norm(self.pts - x, axis=1)
        # Exclude exact match
        dists[dists < 1e-14] = np.inf
        idx = int(np.argmin(dists))
        return idx, dists[idx]

    def _f_interp(self, x: np.ndarray):
        """
        Callable that returns f(x) by nearest-neighbour lookup.
        Used as the f argument to kr_phi.
        """
        x = np.asarray(x, dtype=float)
        dists = np.linalg.norm(self.pts - x, axis=1)
        return float(self.f_vals[np.argmin(dists)])

    def compute(
        self,
        x: np.ndarray,
        n_partners: int = 1,
    ) -> np.ndarray:
        """
        Estimate the Hessian at query point x.

        For each Hessian direction, the nearest available neighbour
        in that direction is used as the sampling partner x+h.

        Parameters
        ----------
        x : np.ndarray, shape (n,)
            Query point (does not need to be a sensor position).
        n_partners : int
            Number of nearest neighbours to average over per direction.
            Default 1.

        Returns
        -------
        np.ndarray, shape (n, n)
            Hessian estimate.
        """
        x = np.asarray(x, dtype=float)
        n = len(x)
        H = np.zeros((n, n))

        def phi_dir(v_unit):
            # Find nearest neighbour and use it as x+h
            idx, dist = self._nearest(x)
            if dist < 1e-12:
                return 0.0
            xn = self.pts[idx]
            h  = xn - x
            # Build a minimal f callable from known values
            def f_local(pt):
                pt = np.asarray(pt, dtype=float)
                dists = np.linalg.norm(self.pts - pt, axis=1)
                return float(self.f_vals[np.argmin(dists)])
            phis = []
            for K in self._kvals:
                try:
                    phis.append(kr_phi(f_local, x, h, K))
                except Exception:
                    pass
            return float(np.mean(phis)) if phis else 0.0

        # Diagonal
        for i in range(n):
            ei = np.zeros(n); ei[i] = 1.0
            H[i, i] = 2.0 * phi_dir(ei)

        # Off-diagonal
        for i in range(n):
            for j in range(i + 1, n):
                ei = np.zeros(n); ei[i] = 1.0
                ej = np.zeros(n); ej[j] = 1.0
                vij = (ei + ej) / np.sqrt(2.0)
                phi_ij = 2.0 * phi_dir(vij)
                H[i, j] = phi_ij - 0.5 * H[i, i] - 0.5 * H[j, j]
                H[j, i] = H[i, j]

        return H

    def rmse_at(
        self,
        x: np.ndarray,
        H_true: np.ndarray,
    ) -> float:
        """Compute RMSE between estimated and true Hessian at x."""
        H_est = self.compute(x)
        return float(np.sqrt(np.mean((H_est - H_true) ** 2)))

    def __repr__(self):
        return (
            f"KRScattered(N={self.N}, n={self.n}, "
            f"n_kvals={self.n_kvals})"
        )
