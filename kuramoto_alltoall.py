"""
kuramoto_alltoall.py

Fast simulation of the all-to-all Kuramoto model using mean-field O(N) computation.
Uses a Lorentzian frequency distribution for direct comparison with OA theory.

=== Coupling normalization ===

Uses K/N normalization (consistent with the standard form of OA theory):
  dθ_i/dt = ω_i + (K/N) * Σ_j sin(θ_j - θ_i)
           = ω_i + K * R * sin(ψ - θ_i)

OA equations:
  dR/dt = -Δ*R + (K/2)*R*(1 - R²)
  K_c = 2Δ

Note: The original paper (Nagahama et al., 2025) and the network version (kuramoto_sim.py)
use unnormalized coupling (K * Σ_j A_ij sin(...)). For the network case, the OA theory
gives K_c = 2Δ<k>/<k²> without normalization.

=== Usage examples ===

# Kc = 2*Delta = 1.0 (with Delta=0.5) under K/N normalization

# eps scan (K = 0.9, i.e. K/Kc = 0.9)
python kuramoto_alltoall.py --N 5000 --K 0.9 --eps 0.05 --Delta 0.5 --mode eps_scan

# Transition curve
python kuramoto_alltoall.py --N 5000 --Delta 0.5 --eps 0.05 --mode K_scan
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
import time
import os
import json
from multiprocessing import Pool, cpu_count

# =============================================
# Arguments
# =============================================
parser = argparse.ArgumentParser()
parser.add_argument('--N', type=int, default=5000, help='number of oscillators')
parser.add_argument('--K', type=float, default=None, help='coupling constant (unnormalized)')
parser.add_argument('--eps', type=float, default=0.05, help='perturbation strength (for single/eps_scan modes)')
parser.add_argument('--eps_list', type=str, default='0.03,0.05,0.07',
                    help='comma-separated eps values for K_scan mode (positive values; negative counterparts auto-added)')
parser.add_argument('--Delta', type=float, default=0.5, help='Lorentzian half-width')
parser.add_argument('--tmax', type=float, default=80.0, help='max simulation time')
parser.add_argument('--tau', type=float, default=0.3, help='perturbation interval')
parser.add_argument('--dt', type=float, default=0.01, help='integration time step')
parser.add_argument('--num_runs', type=int, default=50, help='number of runs')
parser.add_argument('--seed', type=int, default=42, help='random seed')
parser.add_argument('--output_dir', type=str, default='results_alltoall')
parser.add_argument('--random', action='store_true', help='random perturbation control')
parser.add_argument('--mode', type=str, default='single',
                    help='single: single (K,eps) run; eps_scan: scan eps; K_scan: scan K')
args = parser.parse_args()


# =============================================
# All-to-all Kuramoto model (mean-field O(N) computation)
# =============================================

def run_alltoall(N, K, eps, Delta, tmax, tau, dt, random_attack=False, seed=42):
    """
    All-to-all Kuramoto model with K/N normalization:
      dθ_i/dt = ω_i + (K/N) * Σ_j sin(θ_j - θ_i)
              = ω_i + K * R * sin(ψ - θ_i)

    Mean-field formulation gives O(N) per step.
    """
    np.random.seed(seed)

    # Lorentzian (Cauchy) frequency distribution
    omega = np.random.standard_cauchy(N) * Delta
    omega = np.clip(omega, -50 * Delta, 50 * Delta)

    theta = np.random.uniform(0, 2 * np.pi, N)

    def rhs(th):
        z = np.mean(np.exp(1j * th))
        return omega + K * np.abs(z) * np.sin(np.angle(z) - th)

    t = 0.0
    R_late = []

    while t < tmax:
        t_end = min(t + tau, tmax)
        n_steps = max(int((t_end - t) / dt), 1)
        h = (t_end - t) / n_steps

        for _ in range(n_steps):
            k1 = rhs(theta)
            k2 = rhs(theta + 0.5 * h * k1)
            k3 = rhs(theta + 0.5 * h * k2)
            k4 = rhs(theta + h * k3)
            theta = theta + (h / 6) * (k1 + 2*k2 + 2*k3 + k4)

        theta = np.mod(theta, 2 * np.pi)

        # Collect R in steady state (last 50%)
        if t > tmax * 0.5:
            R_late.append(np.abs(np.mean(np.exp(1j * theta))))

        # Adversarial perturbation
        if t_end < tmax and eps != 0:
            psi = np.angle(np.mean(np.exp(1j * theta)))
            if random_attack:
                theta = theta + np.random.choice([-eps, eps], size=N)
            else:
                theta = theta + eps * np.sign(np.sin(psi - theta))

        t = t_end

    return np.mean(R_late) if R_late else 0.0


def run_single_wrapper(params):
    """Wrapper for multiprocessing"""
    run_id, N, K, eps, Delta, tmax, tau, dt, random_attack, base_seed = params
    seed = base_seed + run_id * 97
    R = run_alltoall(N, K, eps, Delta, tmax, tau, dt, random_attack, seed)
    return run_id, R


def run_multi(N, K, eps, Delta, tmax, tau, dt, num_runs, random_attack=False, seed=42):
    """Run in parallel and return mean and standard deviation"""
    params_list = [(i, N, K, eps, Delta, tmax, tau, dt, random_attack, seed)
                   for i in range(num_runs)]

    n_proc = min(cpu_count(), num_runs)
    with Pool(n_proc) as pool:
        results = pool.map(run_single_wrapper, params_list)

    Rs = [r[1] for r in sorted(results)]
    return np.mean(Rs), np.std(Rs), Rs


# =============================================
# OA theory (compatible with K/N normalization)
# =============================================

def S_func(R):
    if R < 1e-12: return 1.0 / np.pi
    if R > 1 - 1e-12: return 0.0
    return (1 - R**2) * np.arctanh(R) / (np.pi * R)

def R_kick(R, eps):
    return R * np.cos(eps) + 2 * np.sin(eps) * S_func(R)

def oa_steady_state(K, Delta, eps, tau, n_iter=3000, tol=1e-8):
    """
    Steady state of the homogeneous OA hybrid map.
    K/N normalization: dR/dt = -Δ*R + (K/2)*R*(1 - R²)
    K_c = 2Δ
    """
    R = 0.01
    for it in range(n_iter):
        R_old = R
        # OA flow (Euler)
        n_steps = 50
        dt_oa = tau / n_steps
        for _ in range(n_steps):
            dR = -Delta * R + (K / 2) * R * (1 - R**2)
            R = max(0.0, min(1.0, R + dt_oa * dR))
        # Kick
        R = max(0.0, min(1.0, R_kick(R, eps)))
        if abs(R - R_old) < tol:
            break
    return R


# =============================================
# Main
# =============================================

if __name__ == "__main__":
    t_start = time.time()
    os.makedirs(args.output_dir, exist_ok=True)

    N = args.N
    Delta = args.Delta
    K_c = 2 * Delta  # critical coupling under K/N normalization

    print(f"All-to-all Kuramoto model (K/N normalization)")
    print(f"  N = {N}")
    print(f"  Δ = {Delta}")
    print(f"  K_c = 2Δ = {K_c:.4f}")

    if args.mode == 'single':
        # Single condition
        K = args.K if args.K is not None else 0.9 * K_c
        print(f"\n--- Single run: K={K:.4f} (K/K_c={K/K_c:.2f}), eps={args.eps} ---")

        R_mean, R_std, _ = run_multi(N, K, args.eps, Delta, args.tmax, args.tau,
                                      args.dt, args.num_runs, args.random, args.seed)
        R_theory = oa_steady_state(K, Delta, args.eps, args.tau)

        print(f"  Simulation: R = {R_mean:.4f} ± {R_std:.4f}")
        print(f"  OA theory:  R = {R_theory:.4f}")

    elif args.mode == 'eps_scan':
        # eps scan
        K = args.K if args.K is not None else 0.9 * K_c
        eps_values = np.array([0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2])

        print(f"\n--- Eps scan: K={K:.4f} (K/K_c={K/K_c:.2f}) ---")

        R_sim_list = []
        R_sim_std_list = []
        R_theory_list = []

        for eps in eps_values:
            R_mean, R_std, _ = run_multi(N, K, eps, Delta, args.tmax, args.tau,
                                          args.dt, args.num_runs, args.random, args.seed)
            R_th = oa_steady_state(K, Delta, eps, args.tau)
            R_sim_list.append(R_mean)
            R_sim_std_list.append(R_std)
            R_theory_list.append(R_th)
            print(f"  eps={eps:.4f}: sim={R_mean:.4f}±{R_std:.4f}, theory={R_th:.4f}")

        R_sim = np.array(R_sim_list)
        R_th = np.array(R_theory_list)

        # Fit exponent
        mask = R_sim > 0.02
        if mask.sum() >= 2:
            c_sim = np.polyfit(np.log(eps_values[mask]), np.log(R_sim[mask]), 1)
            print(f"\n  Simulation exponent: α = {c_sim[0]:.4f}")
        mask_th = R_th > 0.02
        if mask_th.sum() >= 2:
            c_th = np.polyfit(np.log(eps_values[mask_th]), np.log(R_th[mask_th]), 1)
            print(f"  Theory exponent:     α = {c_th[0]:.4f}")

        # Plot
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.errorbar(eps_values, R_sim, yerr=R_sim_std_list, fmt='o', color='C0',
                     capsize=3, ms=7, label='Simulation')
        ax.plot(eps_values, R_th, 's--', color='C3', ms=6, label='OA theory')
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_xlabel(r'$\varepsilon$', fontsize=14)
        ax.set_ylabel(r'$R_\mathrm{ss}$', fontsize=14)
        ax.set_title(f'All-to-all, N={N}, K/Kc={K/K_c:.2f}, ' + r'$\Delta$' + f'={Delta}')
        ax.legend(fontsize=11)
        plt.tight_layout()
        fname = os.path.join(args.output_dir, f'eps_scan_N{N}_KoverKc{K/K_c:.2f}.png')
        plt.savefig(fname, dpi=150, bbox_inches='tight')
        print(f"\n  Saved: {fname}")

        # Save data
        data = {
            'N': N, 'K': K, 'K_c': K_c, 'K_over_Kc': K/K_c, 'Delta': Delta,
            'tau': args.tau, 'num_runs': args.num_runs,
            'eps': eps_values.tolist(),
            'R_sim': R_sim.tolist(), 'R_sim_std': np.array(R_sim_std_list).tolist(),
            'R_theory': R_th.tolist()
        }
        jname = os.path.join(args.output_dir, f'eps_scan_N{N}_KoverKc{K/K_c:.2f}.json')
        with open(jname, 'w') as f:
            json.dump(data, f, indent=2)

    elif args.mode == 'K_scan':
        # K scan (transition curve): supports multiple eps values
        K_ratios = np.linspace(0.3, 3.0, 30)
        eps_pos_list = [float(x) for x in args.eps_list.split(',')]
        # Full list: eps=0 and ±eps
        eps_all = [0.0]
        for e in eps_pos_list:
            eps_all.append(e)
            eps_all.append(-e)

        print(f"\n--- K scan: eps values = {eps_all} ---")

        # Result dict: holds a list of R per eps value
        results_Kscan = {'K_ratio': [], 'eps_values': eps_all}
        for eps_v in eps_all:
            key = f'R_eps{eps_v:+.3f}'
            key_th = f'R_th_eps{eps_v:+.3f}'
            results_Kscan[key] = []
            results_Kscan[key_th] = []

        n_runs_scan = max(args.num_runs // 3, 5)

        for ratio in K_ratios:
            K = ratio * K_c
            results_Kscan['K_ratio'].append(ratio)

            line_parts = [f"  K/Kc={ratio:.2f}:"]
            for eps_v in eps_all:
                # Simulation
                R_sim, _, _ = run_multi(N, K, eps_v, Delta, args.tmax, args.tau,
                                         args.dt, n_runs_scan, False, args.seed)
                # Theory
                R_th = oa_steady_state(K, Delta, eps_v, args.tau)

                key = f'R_eps{eps_v:+.3f}'
                key_th = f'R_th_eps{eps_v:+.3f}'
                results_Kscan[key].append(R_sim)
                results_Kscan[key_th].append(R_th)

                line_parts.append(f"eps={eps_v:+.2f}({R_sim:.3f}/{R_th:.3f})")

            print(' '.join(line_parts))

        # --- Plot ---
        # Color scheme: eps=0 black, positive eps red, negative eps blue
        import matplotlib.cm as cm
        n_pos = len(eps_pos_list)
        red_colors = [cm.Reds(0.4 + 0.5 * i / max(n_pos - 1, 1)) for i in range(n_pos)]
        blue_colors = [cm.Blues(0.4 + 0.5 * i / max(n_pos - 1, 1)) for i in range(n_pos)]

        fig, ax = plt.subplots(figsize=(10, 7))
        K_ratio_arr = np.array(results_Kscan['K_ratio'])

        # eps=0
        key0 = 'R_eps+0.000'
        key0_th = 'R_th_eps+0.000'
        ax.plot(K_ratio_arr, results_Kscan[key0_th], 'k-', lw=2, label=r'OA $\varepsilon=0$')
        ax.plot(K_ratio_arr, results_Kscan[key0], 'ko', ms=5, alpha=0.7)

        # Positive eps
        for i, eps_v in enumerate(eps_pos_list):
            key = f'R_eps+{eps_v:.3f}'
            key_th = f'R_th_eps+{eps_v:.3f}'
            c = red_colors[i]
            ax.plot(K_ratio_arr, results_Kscan[key_th], '-', color=c, lw=2,
                    label=fr'OA $\varepsilon=+{eps_v}$')
            ax.plot(K_ratio_arr, results_Kscan[key], 'o', color=c, ms=5, alpha=0.7)

        # Negative eps
        for i, eps_v in enumerate(eps_pos_list):
            key = f'R_eps-{eps_v:.3f}'
            key_th = f'R_th_eps-{eps_v:.3f}'
            c = blue_colors[i]
            ax.plot(K_ratio_arr, results_Kscan[key_th], '-', color=c, lw=2,
                    label=fr'OA $\varepsilon=-{eps_v}$')
            ax.plot(K_ratio_arr, results_Kscan[key], 'o', color=c, ms=5, alpha=0.7)

        ax.axvline(1.0, color='gray', ls=':', alpha=0.5)
        ax.set_xlabel(r'$K / K_c$', fontsize=14)
        ax.set_ylabel('$R$', fontsize=14)
        ax.set_title(f'All-to-all, N={N}, ' + r'$\Delta$' + f'={Delta}')
        ax.legend(fontsize=9, ncol=2)
        ax.set_ylim(-0.02, 1.02)
        plt.tight_layout()

        eps_tag = '_'.join([f'{e:.2f}' for e in eps_pos_list])
        fname = os.path.join(args.output_dir, f'K_scan_N{N}_eps{eps_tag}.png')
        plt.savefig(fname, dpi=150, bbox_inches='tight')
        print(f"\n  Saved: {fname}")

        # Save JSON
        jname = os.path.join(args.output_dir, f'K_scan_N{N}_eps{eps_tag}.json')
        with open(jname, 'w') as f:
            json.dump(results_Kscan, f, indent=2)
        print(f"  Saved: {jname}")

    elapsed = time.time() - t_start
    print(f"\nTotal: {elapsed:.1f}s")