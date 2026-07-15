"""
domain.py — Domain classes and interior-point verification.

The Boundary Interior Property (Theorem 4.7) guarantees that for
convex Ω, all three K–R sampling points {x, x+Rh, x+h} lie in Ω.
These classes let you verify that property and define which domains
are supported.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Union


class ConvexDomain(ABC):
    """
    Abstract base class for convex domains.

    Every subclass must implement `contains(x)`.
    The K–R Boundary Interior Property (Theorem 4.7) guarantees:
      if x ∈ Ω and x+h ∈ Ω and Ω is convex,
      then x+Rh ∈ Ω for all (K,R) ∈ A.
    """

    @abstractmethod
    def contains(self, x: np.ndarray) -> bool:
        """Return True if x is inside the domain."""
        ...

    def check_stencil(
        self,
        x: np.ndarray,
        h: np.ndarray,
        K: float,
    ) -> dict:
        """
        Check whether all three K–R sampling points lie in the domain.

        Returns a dict with keys:
          'x_in'    : bool — base point x ∈ Ω
          'xh_in'   : bool — endpoint x+h ∈ Ω
          'z_in'    : bool — interior point x+Rh ∈ Ω  (K–R point)
          'xm_in'   : bool — backward point x-h ∈ Ω   (needed by central FD)
          'x2h_in'  : bool — forward point x+2h ∈ Ω   (needed by 1-sided FD)
          'kr_valid': bool — K–R stencil valid (x, z, x+h all in Ω)
          'cfd_valid': bool — central FD valid (x-h, x, x+h all in Ω)
        """
        R = 1.0 - K
        z   = x + R * h
        xm  = x - h
        x2h = x + 2.0 * h
        r = {
            'x_in'    : self.contains(x),
            'xh_in'   : self.contains(x + h),
            'z_in'    : self.contains(z),
            'xm_in'   : self.contains(xm),
            'x2h_in'  : self.contains(x2h),
        }
        r['kr_valid']  = r['x_in'] and r['xh_in'] and r['z_in']
        r['cfd_valid'] = r['x_in'] and r['xh_in'] and r['xm_in']
        return r


# ─────────────────────────────────────────────────────────────────────────────
# Concrete domain implementations
# ─────────────────────────────────────────────────────────────────────────────

class BoxDomain(ConvexDomain):
    """
    Axis-aligned box [lo, hi]^n (convex).

    Parameters
    ----------
    lo, hi : float or array-like
        Lower and upper bounds per dimension.
    """
    def __init__(self, lo, hi):
        self.lo = np.asarray(lo, dtype=float)
        self.hi = np.asarray(hi, dtype=float)

    def contains(self, x: np.ndarray) -> bool:
        x = np.asarray(x, dtype=float)
        return bool(np.all(x >= self.lo) and np.all(x <= self.hi))

    def __repr__(self):
        return f"BoxDomain(lo={self.lo}, hi={self.hi})"


class BallDomain(ConvexDomain):
    """
    Euclidean ball ‖x - centre‖ ≤ radius (convex).
    """
    def __init__(self, centre, radius: float):
        self.centre = np.asarray(centre, dtype=float)
        self.radius = float(radius)

    def contains(self, x: np.ndarray) -> bool:
        return bool(np.linalg.norm(np.asarray(x) - self.centre) <= self.radius)

    def __repr__(self):
        return f"BallDomain(centre={self.centre}, radius={self.radius})"


class AnnulusDomain:
    """
    Annulus r_in < ‖x‖ < r_out in R^2 (NOT convex).

    This domain demonstrates the need for the K–R interior-only
    stencil: central FD points x-h can fall inside the inner hole.
    K–R points x+Rh stay inside the annulus for small h, but
    Theorem 4.7 does NOT apply because the annulus is not convex.
    Use check_stencil() to verify case-by-case.
    """
    def __init__(self, r_in: float = 0.3, r_out: float = 0.9):
        self.r_in  = float(r_in)
        self.r_out = float(r_out)

    def contains(self, x: np.ndarray) -> bool:
        r = float(np.linalg.norm(x))
        return self.r_in < r < self.r_out

    def check_stencil(self, x, h, K=0.5):
        """Check K–R vs FD stencil validity (case-by-case, since non-convex)."""
        R = 1.0 - K
        z   = x + R * h
        xm  = x - h
        x2h = x + 2.0 * h
        r = {
            'x_in'    : self.contains(x),
            'xh_in'   : self.contains(x + h),
            'z_in'    : self.contains(z),
            'xm_in'   : self.contains(xm),
            'x2h_in'  : self.contains(x2h),
        }
        r['kr_valid']  = r['x_in'] and r['xh_in'] and r['z_in']
        r['cfd_valid'] = r['x_in'] and r['xh_in'] and r['xm_in']
        return r

    def __repr__(self):
        return f"AnnulusDomain(r_in={self.r_in}, r_out={self.r_out})"


class RectangleWithHoleDomain:
    """
    Rectangle [-a,a]^2 with circular hole ‖x‖ < r_hole removed.
    NOT convex.
    """
    def __init__(self, a: float = 0.9, r_hole: float = 0.2):
        self.a      = float(a)
        self.r_hole = float(r_hole)

    def contains(self, x: np.ndarray) -> bool:
        x = np.asarray(x, dtype=float)
        in_rect = bool(np.all(np.abs(x) <= self.a))
        in_hole = bool(np.linalg.norm(x) < self.r_hole)
        return in_rect and not in_hole

    def check_stencil(self, x, h, K=0.5):
        R = 1.0 - K
        z   = x + R * h
        xm  = x - h
        r = {
            'x_in'    : self.contains(x),
            'xh_in'   : self.contains(x + h),
            'z_in'    : self.contains(z),
            'xm_in'   : self.contains(xm),
        }
        r['kr_valid']  = r['x_in'] and r['xh_in'] and r['z_in']
        r['cfd_valid'] = r['x_in'] and r['xh_in'] and r['xm_in']
        return r

    def __repr__(self):
        return f"RectangleWithHoleDomain(a={self.a}, r_hole={self.r_hole})"


# ─────────────────────────────────────────────────────────────────────────────
# Utility
# ─────────────────────────────────────────────────────────────────────────────

def is_in_domain(domain, x: np.ndarray) -> bool:
    """Check whether point x lies inside domain."""
    return domain.contains(np.asarray(x, dtype=float))
