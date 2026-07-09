"""
generate_plots.py — Visual benchmark plots for the K-R Defect Framework
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import math, sys
sys.path.insert(0, '/home/claude/kr_library')
from kr_defect import kr_normalised, kr_second_derivative, kr_both_derivatives

np.random.seed(42)
plt.rcParams.update({'font.size':11, 'font.family':'DejaVu Serif',
                     'axes.titlesize':12, 'axes.labelsize':11})

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
fig.suptitle("K–R Defect Framework: Computational Benchmarks", 
             fontsize=14, fontweight='bold', y=0.98)

# ── PLOT 1: Local Limit Theorem convergence ──────────────────────────────────
ax = axes[0,0]
x0 = 0.5
true_val = 0.5 * math.exp(x0)
h_vals = np.logspace(-3, 0, 40)
K_values = [0.3, 0.5, 0.7]
colors = ['#1f77b4','#ff7f0e','#2ca02c']

for K, col in zip(K_values, colors):
    errors = [abs(kr_normalised(math.exp, x0, x0+h, K) - true_val) for h in h_vals]
    ax.loglog(h_vals, errors, '-', color=col, label=f'K={K}', linewidth=1.8)

# O(h) reference line
ax.loglog(h_vals, h_vals, 'k--', alpha=0.5, label='O(h) slope')
ax.set_xlabel('Step size h')
ax.set_ylabel('|Φ_f − (1/2)f″(x)|')
ax.set_title('Benchmark 1: Local Limit Convergence\nf(x)=eˣ, x=0.5')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# ── PLOT 2: Near-boundary comparison ─────────────────────────────────────────
ax = axes[0,1]
positions = np.linspace(0.02, 0.98, 50)
h = 0.05
true_fpp = [-math.sin(x) for x in positions]

kr_errors, cd_errors, cd_fail = [], [], []
for x in positions:
    kr_est = kr_second_derivative(math.sin, x, h=h, K=0.5, average=False)
    kr_errors.append(abs(kr_est - (-math.sin(x))))
    if x - h >= 0:
        cd_est = (math.sin(x+h) - 2*math.sin(x) + math.sin(x-h)) / h**2
        cd_errors.append(abs(cd_est - (-math.sin(x))))
        cd_fail.append(False)
    else:
        cd_errors.append(np.nan)
        cd_fail.append(True)

ax.semilogy(positions, kr_errors, 'b-', label='K–R Defect', linewidth=2)
ax.semilogy(positions, cd_errors, 'r--', label='Central Difference', linewidth=2)
ax.axvspan(0, h, alpha=0.15, color='red', label='CD fails here')
ax.axvline(x=h, color='red', linestyle=':', alpha=0.7)
ax.set_xlabel('Position x')
ax.set_ylabel('|Estimated f″ − True f″|')
ax.set_title('Benchmark 2: Near-Boundary Estimation\nf(x)=sin(x), h=0.05')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# ── PLOT 3: Noise robustness ──────────────────────────────────────────────────
ax = axes[0,2]
sigmas = np.logspace(-5, -1, 20)
n_trials = 300
h = 0.05
x0 = 0.5
true_fpp = math.exp(x0)

cd_means, kr_means = [], []
for sigma in sigmas:
    cd_e, kr_e = [], []
    for _ in range(n_trials):
        noise = lambda: np.random.normal(0, sigma)
        fn = lambda x: math.exp(x) + noise()
        cd = (fn(x0+h) - 2*fn(x0) + fn(x0-h)) / h**2
        cd_e.append(abs(cd - true_fpp))
        K_v = np.linspace(0.1,0.9,9)
        kr = np.mean([2*kr_normalised(fn, x0, x0+h, k) for k in K_v])
        kr_e.append(abs(kr - true_fpp))
    cd_means.append(np.mean(cd_e))
    kr_means.append(np.mean(kr_e))

ax.loglog(sigmas, cd_means, 'r-o', ms=4, label='Central Difference', linewidth=1.8)
ax.loglog(sigmas, kr_means, 'b-s', ms=4, label='K–R (avg 9 K-values)', linewidth=1.8)
ax.set_xlabel('Noise level σ')
ax.set_ylabel('Mean |f̂″ − f″|')
ax.set_title('Benchmark 3: Noise Robustness\nf(x)=eˣ, x=0.5, h=0.05')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# ── PLOT 4: Inverse BVP recovery ─────────────────────────────────────────────
ax = axes[1,0]
N = 100
x_grid = np.linspace(0, 1, N+2)[1:-1]
u_true = np.sin(np.pi * x_grid)
f_true = np.pi**2 * u_true
h_grid = x_grid[1] - x_grid[0]

# Recover f using K-R
K_vals = np.linspace(0.1, 0.9, 9)
f_recovered = np.zeros(len(x_grid))
for i in range(len(x_grid)-1):
    ests = []
    for k in K_vals:
        R = 1-k
        x, y = x_grid[i], x_grid[i+1]
        alpha = R
        fz = (1-alpha)*u_true[i] + alpha*u_true[i+1]
        D = k*u_true[i] + R*u_true[i+1] - fz
        ests.append(-2.0*D/(k*R*h_grid**2))
    f_recovered[i] = np.mean(ests)
f_recovered[-1] = f_recovered[-2]

ax.plot(u_true, f_true, 'b-', label='True f(u) = π²u', linewidth=2)
ax.plot(u_true[:-1], f_recovered[:-1], 'r--', label='K–R Recovered f(u)', linewidth=1.8)
ax.set_xlabel('u (solution values)')
ax.set_ylabel('f(u) (nonlinearity)')
ax.set_title('Benchmark 4: Inverse BVP\n−u″=f(u), recovering f from u')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# ── PLOT 5: Uniqueness Principle verification ─────────────────────────────────
ax = axes[1,1]
x_pts = np.linspace(-2, 2, 30)
y_pts = np.linspace(-2, 2, 30)
K_pts = np.linspace(0.1, 0.9, 10)

f3 = lambda x: x**3
g3 = lambda x: x**3 + 2*x + 5

diffs = []
for xi in x_pts:
    for yi in y_pts:
        if abs(yi-xi) > 0.1:
            for k in K_pts:
                pf = kr_normalised(f3, xi, yi, k)
                pg = kr_normalised(g3, xi, yi, k)
                diffs.append(abs(pf-pg))

ax.hist(np.log10(np.array(diffs)+1e-16), bins=40, color='#2196F3', 
        edgecolor='white', alpha=0.8)
ax.axvline(x=-14, color='red', linestyle='--', label='Machine precision')
ax.set_xlabel('log₁₀|Φ_f − Φ_g|')
ax.set_ylabel('Count')
ax.set_title('Benchmark 5: Uniqueness Principle\nf=x³, g=x³+2x+5 (differ by affine)')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# ── PLOT 6: Dual derivative recovery ─────────────────────────────────────────
ax = axes[1,2]
x_vals = np.linspace(0.1, 2.0, 50)
f4   = lambda x: x**4 + 2*x**3 + x
f4p  = lambda x: 4*x**3 + 6*x**2 + 1
f4pp = lambda x: 12*x**2 + 12*x

fp_true  = [f4p(x)  for x in x_vals]
fpp_true = [f4pp(x) for x in x_vals]
fp_est   = [kr_both_derivatives(f4, x, h=0.01, K=0.6)[0] for x in x_vals]
fpp_est  = [kr_both_derivatives(f4, x, h=0.01, K=0.6)[1] for x in x_vals]

ax.plot(x_vals, fp_true,  'b-',  label="True f'",   linewidth=2)
ax.plot(x_vals, fp_est,   'b--', label="K–R f̂'",  linewidth=1.5, alpha=0.8)
ax.plot(x_vals, fpp_true, 'r-',  label='True f″',   linewidth=2)
ax.plot(x_vals, fpp_est,  'r--', label='K–R f̂″', linewidth=1.5, alpha=0.8)
ax.set_xlabel('x')
ax.set_ylabel('Derivative value')
ax.set_title('Benchmark 7: Complete Dual Derivative\nf(x)=x⁴+2x³+x, h=0.01, K=0.6')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout(rect=[0,0,1,0.96])
plt.savefig('/home/claude/kr_benchmarks.png', dpi=150, bbox_inches='tight')
plt.close()
print("Plots saved to kr_benchmarks.png")
