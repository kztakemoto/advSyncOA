"""
Usage:
    python kuramoto_alltoall_kramers.py \
        --N 1000 --K 1.5 --eps -0.05 --Delta 0.5 \
        --tau 0.3 --tmax 80 --num_runs 100 --seed 42 \
        --output results_kramers/N1000_Kratio1.5_eps-0.05.json
"""

import numpy as np
import json
import argparse
import time
from multiprocessing import Pool, cpu_count

parser = argparse.ArgumentParser()
parser.add_argument('--N',        type=int,   default=1000)
parser.add_argument('--K',        type=float, required=True)
parser.add_argument('--eps',      type=float, required=True)
parser.add_argument('--Delta',    type=float, default=0.5)
parser.add_argument('--tau',      type=float, default=0.3)
parser.add_argument('--tmax',     type=float, default=80.0)
parser.add_argument('--dt',       type=float, default=0.01)
parser.add_argument('--num_runs', type=int,   default=100)
parser.add_argument('--seed',     type=int,   default=42)
parser.add_argument('--output',   type=str,   required=True)
args = parser.parse_args()


def run_alltoall(N, K, eps, Delta, tmax, tau, dt, seed=42):
    np.random.seed(seed)
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
        if t > tmax * 0.5:
            R_late.append(float(np.abs(np.mean(np.exp(1j * theta)))))
        if t_end < tmax and eps != 0:
            psi = np.angle(np.mean(np.exp(1j * theta)))
            theta = theta + eps * np.sign(np.sin(psi - theta))
        t = t_end
    return float(np.mean(R_late)) if R_late else 0.0


def wrapper(params):
    run_id, N, K, eps, Delta, tmax, tau, dt, base_seed = params
    seed = base_seed + run_id * 97
    R = run_alltoall(N, K, eps, Delta, tmax, tau, dt, seed)
    return run_id, R


# ── Run ──────────────────────────────────────────────────────────
t0 = time.time()
params_list = [
    (i, args.N, args.K, args.eps, args.Delta,
     args.tmax, args.tau, args.dt, args.seed)
    for i in range(args.num_runs)
]

n_proc = min(cpu_count(), args.num_runs)
with Pool(n_proc) as pool:
    results = pool.map(wrapper, params_list)

Rs = [r[1] for r in sorted(results)]
R_mean = float(np.mean(Rs))
R_std  = float(np.std(Rs))

print(f"N={args.N}, K={args.K:.4f}, eps={args.eps}: "
      f"R = {R_mean:.4f} ± {R_std:.4f}  ({time.time()-t0:.1f}s)")

# ── Save ─────────────────────────────────────────────────────────
data = {
    'N':        args.N,
    'K':        args.K,
    'K_ratio':  args.K / (2 * args.Delta),
    'eps':      args.eps,
    'Delta':    args.Delta,
    'tau':      args.tau,
    'tmax':     args.tmax,
    'num_runs': args.num_runs,
    'R_mean':   R_mean,
    'R_std':    R_std,
    'R_runs':   Rs,   # R values from individual runs
}
with open(args.output, 'w') as f:
    json.dump(data, f, indent=2)
print(f"  Saved: {args.output}")