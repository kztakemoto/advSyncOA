"""
Fig.1: Fixed-point structure of the hybrid map
"""

import numpy as np
import matplotlib.pyplot as plt

# ── Parameters ──────────────────────────────────────────────────
Delta = 0.5
tau   = 0.3
Kc    = 2 * Delta  # = 1.0

# ── Core functions ───────────────────────────────────────────────
def S(R):
    R = np.asarray(R, dtype=float)
    safe_R = np.where(R < 1e-10, 1e-10, R)
    val = (1 - safe_R**2) * np.arctanh(safe_R) / (np.pi * safe_R)
    return np.where(R < 1e-10, 1.0 / np.pi, val)

def kick(R, eps):
    R_new = R * np.cos(eps) + 2 * np.sin(eps) * S(R)
    return np.clip(R_new, 0.0, 1.0)

def oa_flow(R, K, tau, Delta):
    dt = 0.001
    steps = int(tau / dt)
    r = R.copy()
    for _ in range(steps):
        k1 = (-Delta * r + K/2 * r * (1 - r**2))
        k2 = (-Delta * (r + dt/2*k1) + K/2 * (r + dt/2*k1) * (1 - (r + dt/2*k1)**2))
        k3 = (-Delta * (r + dt/2*k2) + K/2 * (r + dt/2*k2) * (1 - (r + dt/2*k2)**2))
        k4 = (-Delta * (r + dt*k3)   + K/2 * (r + dt*k3)   * (1 - (r + dt*k3)**2))
        r = r + dt/6*(k1 + 2*k2 + 2*k3 + k4)
        r = np.clip(r, 0.0, 1.0)
    return r

def hybrid_map(R, K, eps, tau, Delta):
    R_after_flow = oa_flow(R, K, tau, Delta)
    return kick(R_after_flow, eps)

# ── Plot settings ────────────────────────────────────────────────
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'legend.fontsize': 9.5,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
})

R_arr = np.linspace(0, 0.99, 500)

fig, axes = plt.subplots(2, 1, figsize=(5.5, 8))

# ════════════════════════════════════════════════════════════════
# Panel (a): Enhancement  eps > 0
# ════════════════════════════════════════════════════════════════
ax = axes[0]

eps    = 0.05
K_vals = [0.7 * Kc, 0.9 * Kc, 1.0 * Kc, 1.2 * Kc]
colors = ['#2166ac', '#74add1', '#f46d43', '#d73027']
labels = [r'$K=0.7K_c$', r'$K=0.9K_c$', r'$K=K_c$', r'$K=1.2K_c$']

for K, col, lab in zip(K_vals, colors, labels):
    dR = hybrid_map(R_arr, K, eps, tau, Delta) - R_arr
    ax.plot(R_arr, dR, color=col, lw=2, label=lab)

ax.axhline(0, color='k', lw=0.8, ls='--')
ax.axvline(0, color='k', lw=0.5, ls=':')

# Mark stable fixed points
for K, col in zip(K_vals, colors):
    dR = hybrid_map(R_arr, K, eps, tau, Delta) - R_arr
    sign_changes = np.where(np.diff(np.sign(dR)))[0]
    for idx in sign_changes:
        slope = dR[idx+1] - dR[idx]
        if slope < 0:  # stable
            ax.plot(R_arr[idx], 0, 'o', color=col, ms=8, zorder=5)

ax.set_xlabel(r'$R$')
ax.set_ylabel(r'$R_{\rm new} - R$  (per cycle)')
ax.set_title(r'(a) Enhancement ($\epsilon = +0.05$)')
ax.set_xlim(0, 1)

from matplotlib.lines import Line2D
symbol_handles_a = [
    Line2D([0], [0], color='gray', lw=0, marker='o',
           ms=8, label='stable fixed point'),
]
leg_a = ax.legend(loc='lower left', framealpha=0.9)
ax.add_artist(leg_a)
ax.legend(handles=symbol_handles_a, loc='upper right', framealpha=0.9)

# ════════════════════════════════════════════════════════════════
# Panel (b): Suppression  eps < 0
# ════════════════════════════════════════════════════════════════
ax = axes[1]

eps     = -0.05
K_vals2 = [1.0 * Kc, 1.5 * Kc, 2.0 * Kc, 3.0 * Kc]
colors2 = ['#2166ac', '#74add1', '#f46d43', '#d73027']
labels2 = [r'$K=K_c$', r'$K=1.5K_c$', r'$K=2.0K_c$', r'$K=3.0K_c$']

for K, col, lab in zip(K_vals2, colors2, labels2):
    dR = hybrid_map(R_arr, K, eps, tau, Delta) - R_arr
    ax.plot(R_arr, dR, color=col, lw=2, label=lab)

ax.axhline(0, color='k', lw=0.8, ls='--')

# Mark fixed points and separatrix
for K, col in zip(K_vals2, colors2):
    dR = hybrid_map(R_arr, K, eps, tau, Delta) - R_arr
    sign_changes = np.where(np.diff(np.sign(dR)))[0]
    for idx in sign_changes:
        R_fp_interp = (R_arr[idx]
                       - dR[idx] * (R_arr[idx+1] - R_arr[idx])
                       / (dR[idx+1] - dR[idx]))
        slope = dR[idx+1] - dR[idx]
        if slope > 0:  # unstable = separatrix
            ax.plot(R_fp_interp, 0, 's', color=col, ms=8, zorder=5,
                    markerfacecolor='white', markeredgewidth=1.8)
        else:          # stable upper fixed point
            ax.plot(R_fp_interp, 0, 'o', color=col, ms=8, zorder=5)

# R=0 absorbing state (symbol only, not in K legend)
ax.plot(0, 0, 'k^', ms=9, zorder=6)

ax.set_xlabel(r'$R$')
ax.set_ylabel(r'$R_{\rm new} - R$  (per cycle)')
ax.set_title(r'(b) Suppression ($\epsilon = -0.05$)')
ax.set_xlim(0, 1)

# Combined legend: K values + symbol meanings
from matplotlib.lines import Line2D
symbol_handles = [
    Line2D([0], [0], color='gray', lw=0, marker='s',
           markerfacecolor='white', markeredgecolor='gray',
           markeredgewidth=1.8, ms=8, label=r'separatrix $R^*$'),
    Line2D([0], [0], color='gray', lw=0, marker='o',
           ms=8, label='stable fixed point'),
    Line2D([0], [0], color='k', lw=0, marker='^',
           ms=9, label=r'$R=0$ (absorbing)'),
]
leg1 = ax.legend(loc='lower left', bbox_to_anchor=(0.0, 0.0), framealpha=0.9)
ax.add_artist(leg1)
ax.legend(handles=symbol_handles, loc='lower left', bbox_to_anchor=(0.35, 0.0), framealpha=0.9)

# ── Final layout ─────────────────────────────────────────────────
fig.tight_layout()
fig.savefig('fig_fixedpoint.png', dpi=150, bbox_inches='tight')
fig.savefig('fig_fixedpoint.pdf', bbox_inches='tight')
fig.savefig('fig_fixedpoint.eps', bbox_inches='tight')
print("Saved: fig_fixedpoint.png / .pdf / .eps")
