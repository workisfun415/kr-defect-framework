"""
tests/test_core.py — Validation tests for the K–R defect framework.

Run with:
    python -m pytest tests/ -v
or:
    python tests/test_core.py

Each test corresponds to a theorem in the paper.
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kr_defect import KRHessian, kr_defect, kr_phi
from kr_defect.domain import BoxDomain, AnnulusDomain, BallDomain
from kr_defect.beam import beam_bending_moment, beam_exact_solution
from kr_defect.utils import convergence_order, compare_methods

PASS = "\033[92m  PASS\033[0m"
FAIL = "\033[91m  FAIL\033[0m"

def check(name, condition, tol_info=""):
    status = PASS if condition else FAIL
    print(f"{status}  {name}" + (f"  [{tol_info}]" if tol_info else ""))
    return condition


def run_all():
    results = []
    print("\n" + "="*60)
    print("  K–R Framework — Validation Test Suite")
    print("="*60)

    # ── Test 1: Well-definedness (Theorem 4.1) ──────────────────
    print("\n[Thm 4.1] Well-definedness")
    f = lambda x: np.exp(x[0] + x[1])
    x = np.array([0.5, 0.5])
    h = np.array([0.1, 0.0])
    val = kr_phi(f, x, h, K=0.5)
    results.append(check("kr_phi returns finite value",
                         np.isfinite(val), f"val={val:.4f}"))

    try:
        kr_phi(f, x, h, K=0.0)
        results.append(check("K=0 raises ValueError", False))
    except ValueError:
        results.append(check("K=0 raises ValueError", True))

    try:
        kr_phi(f, x, np.zeros(2), K=0.5)
        results.append(check("h=0 raises ValueError", False))
    except ValueError:
        results.append(check("h=0 raises ValueError", True))

    # ── Test 2: Curvature Recovery (Theorem 4.2) ────────────────
    print("\n[Thm 4.2] Curvature Recovery — Φ_f ≈ ½ v^T H_f(ξ) v")
    f = lambda x: np.exp(x[0] + x[1])
    x = np.array([0.5, 0.5])
    e_val = np.exp(1.0)
    H_true = e_val * np.ones((2, 2))

    kr = KRHessian(f, step=0.01, n_kvals=9)
    H_est = kr.compute(x)
    rmse = np.sqrt(np.mean((H_est - H_true)**2))
    results.append(check("RMSE < 0.05 at step=0.01",
                         rmse < 0.05, f"rmse={rmse:.5f}"))

    # ── Test 3: Local Limit (Theorem 4.3) ───────────────────────
    print("\n[Thm 4.3] Local Limit — Φ_f → ½ v^T H_f(x) v as s→0")
    v = np.array([1.0, 0.0])
    true_curv = 0.5 * v @ H_true @ v  # = 0.5 * e_val
    errors = []
    for s in [0.2, 0.1, 0.05, 0.025]:
        est = kr.curvature(x, v, step=s)
        errors.append(abs(est - true_curv))
    # Error should decrease as step decreases
    results.append(check("Error decreases as step decreases",
                         errors[-1] < errors[0],
                         f"err at s=0.2: {errors[0]:.4f}, s=0.025: {errors[-1]:.4f}"))

    # ── Test 4: Convergence Order O(h) (Theorem 4.5) ────────────
    print("\n[Thm 4.5] Consistency — O(h) convergence")
    r = convergence_order(f, x, H_true, entry=(0,0))
    results.append(check("Convergence order p ∈ [0.8, 1.3]",
                         0.8 < r['order'] < 1.3, f"p={r['order']:.3f}"))

    # ── Test 5: Uniqueness Principle (Theorem 4.4) ───────────────
    print("\n[Thm 4.4] Uniqueness — equal defect ⟹ equal Hessian")
    # f1 and f2 differ by an affine function — should give same Φ
    f1 = lambda x: np.exp(x[0]+x[1])
    f2 = lambda x: np.exp(x[0]+x[1]) + 2.0*x[0] + 3.0*x[1] + 7.0
    kr1 = KRHessian(f1, step=0.05, n_kvals=9)
    kr2 = KRHessian(f2, step=0.05, n_kvals=9)
    H1 = kr1.compute(x)
    H2 = kr2.compute(x)
    diff = np.max(np.abs(H1 - H2))
    results.append(check("f1 and f1+affine give same Hessian",
                         diff < 0.001, f"max|H1-H2|={diff:.6f}"))

    # ── Test 6: Boundary Interior Property (Theorem 4.7) ────────
    print("\n[Thm 4.7] Boundary Interior Property")
    dom = BoxDomain(0.0, 1.0)
    x_near = np.array([0.05, 0.5])  # near left boundary
    h_right = np.array([0.1, 0.0])
    s = dom.check_stencil(x_near, h_right, K=0.5)
    results.append(check("K–R stencil valid near boundary",
                         s['kr_valid'], str(s)))
    results.append(check("Central FD invalid (x-h outside)",
                         not s['cfd_valid'], str(s)))

    # Annulus: K–R can stay inside, FD can't
    ann = AnnulusDomain(r_in=0.3, r_out=0.9)
    x_ann = np.array([0.31, 0.0])
    h_ann = np.array([0.15, 0.0])
    s2 = ann.check_stencil(x_ann, h_ann, K=0.5)
    results.append(check("Annulus: K–R interior point inside ring",
                         s2['z_in'], f"z_in={s2['z_in']}"))
    results.append(check("Annulus: FD backward point in hole",
                         not s2['xm_in'], f"xm_in={s2['xm_in']}"))

    # ── Test 7: Rotational Invariance (Theorem 4.8) ──────────────
    print("\n[Thm 4.8] Rotational Invariance — H_g = Q^T H_f Q")
    f_rot = lambda x: np.exp(x[0]+x[1])
    x0 = np.array([0.5, 0.5])
    H_f = np.exp(1.0) * np.ones((2,2))

    theta = np.pi / 4
    Q = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta),  np.cos(theta)]])
    g = lambda x: f_rot(Q @ x)
    H_g_true = Q.T @ H_f @ Q

    kr_g = KRHessian(g, step=0.05, n_kvals=9)
    H_g_est = kr_g.compute(x0)
    rmse_rot = np.sqrt(np.mean((H_g_est - H_g_true)**2))
    results.append(check("Rotational invariance RMSE < 2.0",
                         rmse_rot < 2.0, f"rmse={rmse_rot:.4f}"))

    # ── Test 8: Beam application ─────────────────────────────────
    print("\n[Application] Euler–Bernoulli beam bending moment")
    u_exact = lambda x: float(np.atleast_1d(x)[0])**2 * (1 - float(np.atleast_1d(x)[0]))**2 / 24.0
    x_pos = np.array([0.01, 0.05, 0.5])
    res   = beam_bending_moment(u_exact, x_pos, step=0.04, n_kvals=9)

    for xi, M_kr in zip(res['x'], res['M_kr']):
        M_true = beam_exact_solution(xi)['M']
        err = abs(M_kr - M_true)
        results.append(check(f"Beam at x={xi:.3f}: err < 0.05",
                             err < 0.05, f"err={err:.5f}"))

    # Result at x=0.01 where FD is not applicable
    results.append(check("FD not applicable at x=0.01",
                         not res['fd_valid'][0],
                         f"fd_valid={res['fd_valid'][0]}"))
    results.append(check("K–R valid at x=0.01",
                         not np.isnan(res['M_kr'][0]),
                         f"M_kr={res['M_kr'][0]:.5f}"))

    # ── Summary ──────────────────────────────────────────────────
    print("\n" + "="*60)
    n_pass = sum(results)
    n_fail = len(results) - n_pass
    print(f"  Passed: {n_pass}/{len(results)}")
    if n_fail > 0:
        print(f"  Failed: {n_fail}")
    print("="*60)
    return n_fail == 0


if __name__ == "__main__":
    ok = run_all()
    sys.exit(0 if ok else 1)
