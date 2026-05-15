"""
Degree-resolved R_k for BA network at K = Kc,
comparing epsilon=0, 0.03, 0.05, 0.07.
Single panel: R_k vs k
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.size'] = 11

# ── OA helper functions ───────────────────────────────────────────────

def S_func(R):
    if R < 1e-12:
        return 1.0 / np.pi
    if R > 1 - 1e-12:
        return 0.0
    return (1 - R**2) * np.arctanh(R) / (np.pi * R)

def R_kick(R, eps):
    return R * np.cos(eps) + 2 * np.sin(eps) * S_func(R)

def get_ba_coarse(m=3, n_bins=20):
    k_max = 200
    k_all = np.arange(m, k_max + 1).astype(float)
    Pk_all = 2.0 * m * (m + 1) / (k_all * (k_all + 1) * (k_all + 2))
    Pk_all /= Pk_all.sum()
    edges = np.unique(
        np.logspace(np.log10(m - 0.5), np.log10(k_max + 0.5), n_bins + 1
                    ).astype(int))
    k_rep, Pk_rep = [], []
    for i in range(len(edges) - 1):
        mask = (k_all >= edges[i]) & (k_all < edges[i + 1])
        if mask.sum() > 0 and Pk_all[mask].sum() > 1e-10:
            k_rep.append(np.average(k_all[mask], weights=Pk_all[mask]))
            Pk_rep.append(Pk_all[mask].sum())
    k_rep  = np.array(k_rep)
    Pk_rep = np.array(Pk_rep)
    Pk_rep /= Pk_rep.sum()
    return k_rep, Pk_rep

def find_ss(k_arr, Pk, K, Delta, eps, tau, n_iter=2000, tol=1e-7):
    n      = len(k_arr)
    k_mean = np.sum(k_arr * Pk)
    R_k    = np.full(n, 0.1)
    for _ in range(n_iter):
        R_old = R_k.copy()
        n_steps = 30
        dt = tau / n_steps
        for _ in range(n_steps):
            H   = np.sum(k_arr * Pk * R_k) / k_mean
            dR  = -Delta * R_k + (k_arr * K / 2) * H * (1 - R_k**2)
            R_k = np.clip(R_k + dt * dR, 0, 1)
        for i in range(n):
            R_k[i] = max(0.0, min(1.0, R_kick(R_k[i], eps)))
        if np.max(np.abs(R_k - R_old)) < tol:
            break
    R_global = np.sum(Pk * R_k)
    return R_global, R_k

# ── Parameters ───────────────────────────────────────────────────────

Delta = 0.5
tau   = 0.3
m     = 6          # BA minimum degree

eps_vals   = [0.03, 0.05, 0.07]
eps_colors = ['#FFAAAA', '#EE4444', '#AA0000']  # light -> dark red
eps_labels = [r'$\epsilon = +0.03$', r'$\epsilon = +0.05$', r'$\epsilon = +0.07$']

k_ba, Pk_ba = get_ba_coarse(m)
k_mean_ba   = np.sum(k_ba * Pk_ba)
k2_ba       = np.sum(k_ba**2 * Pk_ba)
Kc_ba       = 2 * Delta * k_mean_ba / k2_ba

print(f"BA: Kc = {Kc_ba:.4f},  <k> = {k_mean_ba:.2f},  <k^2> = {k2_ba:.2f}")

_, R_k0 = find_ss(k_ba, Pk_ba, Kc_ba, Delta, 0.00, tau)

results = []
for eps in eps_vals:
    _, R_k = find_ss(k_ba, Pk_ba, Kc_ba, Delta, eps, tau)
    results.append(R_k)

# ── Plot ─────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(5.5, 4))

ax.plot(k_ba, R_k0, 'o-', color='black', ms=5, lw=1.5,
        label=r'$\epsilon = 0$')
for R_k, color, label in zip(results, eps_colors, eps_labels):
    ax.plot(k_ba, R_k, 's-', color=color, ms=5, lw=1.5, label=label)

ax.set_xlabel('Degree $k$')
ax.set_ylabel(r'$R_k$')
ax.set_xscale('log')
ax.set_xlim(m - 0.5, 200)
ax.set_ylim(-0.02, 1.02)
ax.legend(loc='center right', fontsize=9)

plt.tight_layout()
plt.savefig('fig_Rk_BA.pdf', bbox_inches='tight')
fig.savefig('fig_Rk_BA.eps', bbox_inches='tight')
plt.savefig('fig_Rk_BA.png', dpi=200, bbox_inches='tight')
print("Saved fig_Rk_BA.pdf / .png")
