"""
kr_defect — Multivariable K–R Defect Framework for Hessian Recovery
====================================================================
Author  : RamaKrishna Pasupuleti
ORCID   : 0009-0008-8418-1430
License : MIT
Paper   : Multivariable K–R Defect Theory (JCAM, submitted)
GitHub  : https://github.com/workisfun415/kr-defect-framework

Quick start
-----------
>>> import numpy as np
>>> from kr_defect import KRHessian
>>> f = lambda x: np.exp(x[0] + x[1])
>>> kr = KRHessian(f, step=0.05)
>>> H = kr.compute(np.array([0.5, 0.5]))
>>> print(H)
"""

from .core       import kr_defect, kr_phi, KRHessian
from .domain     import ConvexDomain, AnnulusDomain, RectangleWithHoleDomain, is_in_domain
from .scattered  import KRScattered
from .beam       import beam_bending_moment
from .utils      import convergence_order, monte_carlo_error

__version__ = "1.0.0"
__author__  = "RamaKrishna Pasupuleti"

__all__ = [
    "kr_defect", "kr_phi", "KRHessian",
    "ConvexDomain", "AnnulusDomain", "RectangleWithHoleDomain",
    "is_in_domain", "KRScattered", "beam_bending_moment",
    "convergence_order", "monte_carlo_error",
]
