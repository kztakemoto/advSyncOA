"""
check_kramers_slope.py

Compare the slopes of the Kramers plot with the ratio of separatrix R*².
Conditions without a separatrix (e.g., K/Kc=1.5) are automatically excluded.

Theoretical prediction: slope ratio = R*² ratio

Usage:
    python check_kramers_slope.py --input_dir results_kuramers
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import json, glob, os, argparse
from scipy.optimize import brentq
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', type=str, default='results_kuramers')
parser.add_argument('--threshold', type=float, default=0.05)
parser.add_argument('--output', type=str, default='fig_kramers_slope_check.png')
args = parser.parse_args()

# ── OA functions ─────────────────────────────────────────────────
Delta = 0.5
tau   = 0.3
Kc    = 2 * Delta

def S(R):
    if R < 1e-10: return 1.0/np.pi
    return (1 - R**2) * np.arctanh(R) / (np.pi * R)

def kick(R, eps):
    return max(0.0, R * np.cos(eps) + 2 * np.sin(eps) * S(R))

def oa_step(R, K, dt=0.001, steps=300):
    for _ in range(steps):
        dR = -Delta*R + (K/2)*R*(1 - R**2)
        R = max(0.0, min(1.0, R + dt*dR))
    return R

def hybrid_map(R, K, eps):
    return kick(oa_step(R, K), eps)

def find_separatrix(K, eps, R_low=1e-4, R_high=0.99):
    """
    Find the unstable fixed point (separatrix) of the hybrid map.
    Returns None if no negative-to-positive zero crossing is found.
    """
    def f(R):
        return hybrid_map(R, K, eps) - R
    R_arr = np.linspace(R_low, R_high, 1000)
    f_arr = np.array([f(r) for r in R_arr])
    for i in range(len(f_arr)-1):
        if f_arr[i] < 0 and f_arr[i+1] > 0:  # unstable fixed point
            try:
                return brentq(f, R_arr[i], R_arr[i+1])
            except:
                return None
    return None  # no separatrix

# ── Load data ────────────────────────────────────────────────────
files = sorted(glob.glob(os.path.join(args.input_dir, '*.json')))
if not files:
    print(f"No JSON files found in {args.input_dir}")
    exit(1)

records = []
for f in files:
    with open(f) as fh:
        d = json.load(fh)
    Rs = d['R_runs']
    d['P_escape'] = float(np.mean(np.array(Rs) >= args.threshold))
    records.append(d)

# ── Compute slopes from Kramers plot ────────────────────────────
groups = defaultdict(list)
for r in records:
    key = (r['eps'], r['K_ratio'])
    groups[key].append(r)
for key in groups:
    groups[key].sort(key=lambda x: x['N'])

eps_vals      = sorted(set(r['eps']      for r in records))
K_ratio_vals  = sorted(set(r['K_ratio'] for r in records))

slopes = {}
Rstar2 = {}

print(f"{'eps':>6}  {'K/Kc':>6}  {'R*':>10}  {'slope':>12}  {'status':>20}")
print("-" * 65)

for eps in eps_vals:
    for ratio in K_ratio_vals:
        key = (eps, ratio)
        if key not in groups:
            continue

        # Check for separatrix
        K = ratio * Kc
        Rs = find_separatrix(K, eps)
        if Rs is None:
            print(f"{eps:>6.2f}  {ratio:>6.1f}  {'N/A':>10}  {'N/A':>12}  "
                  f"{'no separatrix (skip)':>20}")
            continue

        # Compute Kramers slope
        recs  = groups[key]
        N_arr = np.array([r['N']        for r in recs])
        P_arr = np.array([r['P_escape'] for r in recs])
        mask  = (P_arr > 0.0) & (P_arr < 1.0)
        if mask.sum() < 2:
            print(f"{eps:>6.2f}  {ratio:>6.1f}  {Rs:>10.4f}  {'N/A':>12}  "
                  f"{'insufficient data':>20}")
            continue

        y      = np.log(-np.log(1 - P_arr[mask]))
        coeffs = np.polyfit(N_arr[mask], y, 1)
        slopes[key] = coeffs[0]
        Rstar2[key] = Rs**2
        print(f"{eps:>6.2f}  {ratio:>6.1f}  {Rs:>10.4f}  "
              f"{coeffs[0]:>12.4e}  {'OK':>20}")

# ── Slope ratio vs R*² ratio ─────────────────────────────────────
print("\n=== Slope ratio vs R*² ratio (reference: K/Kc=3.0 for each eps) ===")
print(f"{'eps':>6}  {'K/Kc':>6}  {'slope ratio':>12}  {'R*² ratio':>12}  {'agreement':>10}")
print("-" * 55)

for eps in eps_vals:
    ref_key = (eps, 3.0)
    if ref_key not in slopes:
        continue
    for ratio in K_ratio_vals:
        if ratio == 3.0:
            continue
        key = (eps, ratio)
        if key not in slopes:
            continue
        sr = slopes[key] / slopes[ref_key]
        rr = Rstar2[key] / Rstar2[ref_key]
        print(f"{eps:>6.2f}  {ratio:>6.1f}  {sr:>12.4f}  {rr:>12.4f}  {sr/rr:>10.3f}")

# ── Color/marker scheme ──────────────────────────────────────────
# eps encoded by color, K/Kc encoded by marker
valid_eps    = sorted(set(k[0] for k in slopes))
valid_ratios = sorted(set(k[1] for k in slopes))
eps_colors   = dict(zip(valid_eps,
                        [cm.Blues(v) for v in [0.85, 0.65, 0.45]]))
K_markers    = {2.0: 's', 3.0: '^'}

# ── Global style (consistent with previous version) ───────────────
plt.rcParams.update({
    'font.size':        14,
    'axes.labelsize':   15,
    'axes.titlesize':   14,
    'legend.fontsize':  12,
    'xtick.labelsize':  12,
    'ytick.labelsize':  12,
})

# ── Figure: 2 panels (a) Kramers plot  (b) slope vs R*² ──────────
fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(13, 6))

# ===== Panel (a): Kramers plot =====
for eps in valid_eps:
    col = eps_colors[eps]
    for ratio in valid_ratios:
        key = (eps, ratio)
        if key not in slopes:
            continue
        mk   = K_markers.get(ratio, 'o')
        recs = groups[key]
        N_arr = np.array([r['N']        for r in recs])
        P_arr = np.array([r['P_escape'] for r in recs])
        mask  = (P_arr > 0.0) & (P_arr < 1.0)
        if mask.sum() < 2:
            continue
        y = np.log(-np.log(1 - P_arr[mask]))
        ax_a.plot(N_arr[mask], y, marker=mk, color=col,
                  ms=9, lw=0, zorder=5)
        # fit line
        coeffs = np.polyfit(N_arr[mask], y, 1)
        N_fit  = np.linspace(N_arr[mask].min(), N_arr[mask].max(), 50)
        ax_a.plot(N_fit, np.polyval(coeffs, N_fit), '-', color=col,
                  lw=2.0, alpha=0.85)

# dummy entries for legend (eps by color)
for eps_i, col in eps_colors.items():
    ax_a.plot([], [], color=col, lw=2.5,
              label=rf'$\varepsilon={eps_i:+.2f}$')
# dummy entries for legend (K/Kc by marker)
for ratio_i, mk in K_markers.items():
    if ratio_i in valid_ratios:
        ax_a.plot([], [], marker=mk, color='gray', lw=0, ms=9,
                  label=rf'$K={ratio_i:.1f}K_c$')

ax_a.set_xlabel(r'$N$')
ax_a.set_ylabel(r'$\log(-\log(1 - P_{\rm escape}))$')
ax_a.set_title('(a)')
ax_a.legend(ncol=2, loc='upper right')

# ===== Panel (b): slope vs R*² =====
Rstar2_arr = []
slope_arr  = []
for eps in valid_eps:
    for ratio in valid_ratios:
        key = (eps, ratio)
        if key not in slopes:
            continue
        col = eps_colors[eps]
        mk  = K_markers.get(ratio, 'o')
        ax_b.scatter(Rstar2[key], abs(slopes[key]),
                     s=180, color=col, marker=mk, zorder=5)
        Rstar2_arr.append(Rstar2[key])
        slope_arr.append(abs(slopes[key]))

Rstar2_arr = np.array(Rstar2_arr)
slope_arr  = np.array(slope_arr)

# linear fit through the origin
c_fit = np.dot(Rstar2_arr, slope_arr) / np.dot(Rstar2_arr, Rstar2_arr)
x_fit = np.linspace(0, Rstar2_arr.max() * 1.15, 50)
ax_b.plot(x_fit, c_fit * x_fit, 'k--', lw=2.0,
          label=rf'$c = {c_fit:.2f}$')
print(f"\nLinear fit through origin: slope = {c_fit:.4f} * R*²")

# dummy entries for legend (eps by color)
for eps_i, col in eps_colors.items():
    ax_b.scatter([], [], color=col, marker='o', s=120,
                 label=rf'$\varepsilon={eps_i:+.2f}$')
# dummy entries for legend (K/Kc by marker)
for ratio_i, mk in K_markers.items():
    if ratio_i in valid_ratios:
        ax_b.scatter([], [], color='gray', marker=mk, s=120,
                     label=rf'$K={ratio_i:.1f}K_c$')

ax_b.set_xlabel(r'$R^{*2}$ (from OA theory)')
ax_b.set_ylabel(r'$|\mathrm{slope}|$ of Kramers plot')
ax_b.set_title('(b)')
ax_b.legend(ncol=2, loc='upper left')
ax_b.set_xlim(left=0)
ax_b.set_ylim(bottom=0)

fig.tight_layout()
fig.savefig(args.output, dpi=150, bbox_inches='tight')
base = os.path.splitext(args.output)[0]
fig.savefig(base + '.pdf', bbox_inches='tight')
fig.savefig(base + '.eps', bbox_inches='tight')
print(f"Saved: {args.output}, {base}.pdf, {base}.eps")