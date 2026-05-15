"""
Usage examples:
  # ER network with Lorentzian distribution (half-width Delta=0.5), K=0.15, eps=0.05
  python kuramoto_sim.py --network ER --N 1000 --kave 6 --K 0.15 --eps 0.05 \
      --freq_dist lorentzian --freq_width 0.5 --tmax 80 --num_runs 100

  # Gaussian distribution (std=1.0), same as before
  python kuramoto_sim.py --network ER --N 1000 --kave 6 --K 0.20 --eps 0.05 \
      --tmax 80 --num_runs 100
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from numba import jit
import argparse
from multiprocessing import Pool, cpu_count
import time
import os
import pandas as pd

#### Parameters #############
parser = argparse.ArgumentParser()
parser.add_argument('--network', type=str, default='ER')
parser.add_argument('--N', type=int, default=1000)
parser.add_argument('--kave', type=float, default=6.0)
parser.add_argument('--K', type=float, default=0.4)
parser.add_argument('--eps', type=float, default=0.05)
parser.add_argument('--tmax', type=float, default=50.0)
parser.add_argument('--attack_type', type=str, default='node')
parser.add_argument('--t_interval', type=float, default=0.3)
parser.add_argument('--random', action='store_true')
parser.add_argument('--seed', type=int, default=123)
parser.add_argument('--num_runs', type=int, default=100)
parser.add_argument('--output_dir', type=str, default='results')
parser.add_argument('--attack_start', type=float, default=0.0)
parser.add_argument('--attack_end', type=float, default=None)
# === Additional options ===
parser.add_argument('--freq_dist', type=str, default='gaussian',
                    help='frequency distribution: gaussian or lorentzian')
parser.add_argument('--freq_width', type=float, default=1.0,
                    help='distribution width (std for gaussian, half-width Delta for lorentzian)')
args = parser.parse_args()

real_network_data = ['power-1138-bus', 'bn-mouse']

@jit(nopython=True)
def kuramoto_numba(theta, K, omega, adj_matrix):
    N = len(theta)
    dtheta = np.zeros(N)
    for i in range(N):
        sum_sin = 0.0
        for j in range(N):
            if adj_matrix[i, j] != 0:
                sum_sin += adj_matrix[i, j] * np.sin(theta[j] - theta[i])
        dtheta[i] = omega[i] + K * sum_sin
    return dtheta

def kuramoto_wrapper(t, theta, K, omega, adj_matrix):
    return kuramoto_numba(theta, K, omega, adj_matrix)

def calculate_order_parameter(theta):
    return np.abs(np.mean(np.exp(1j * theta)))

def generate_network(network_type, N, kave, seed):
    if network_type == 'BA':
        g = nx.barabasi_albert_graph(N, int(kave / 2), seed=seed)
    elif network_type == 'ER':
        g = nx.gnm_random_graph(N, int(kave * N / 2), directed=False, seed=seed)
    elif network_type == 'WS':
        pws = 0.05
        g = nx.watts_strogatz_graph(N, int(kave), pws, seed=seed)
    elif network_type == 'RR':
        d = int(kave)
        if (d * N) % 2 != 0:
            raise ValueError(f"RR network requires d * N to be even, but got d={d}, N={N} (d * N = {d * N})")
        g = nx.random_regular_graph(d, N, seed=seed)
    elif network_type in real_network_data:
        df = pd.read_csv(f"./network_data/{network_type}.txt", sep=r'\s+', header=None)
        g = nx.from_pandas_edgelist(df, source=0, target=1)
        g = nx.Graph(g)
        g.remove_edges_from(nx.selfloop_edges(g))
        lcc = max(nx.connected_components(g), key=len)
        g = g.subgraph(lcc)
    else:
        raise ValueError(f"Invalid network type: {network_type}")
    return nx.to_numpy_array(g)

def run_kuramoto_simulation(params):
    (run_id, dt, network_type, N, kave, eps, K, tmax, t_interval, attack_type,
     random_attack, attack_start, attack_end, seed, freq_dist, freq_width) = params

    run_seed = seed + run_id * 100
    np.random.seed(run_seed)

    adj_matrix = generate_network(network_type, N, kave, seed=run_seed)

    if network_type in real_network_data:
        N = len(adj_matrix)

    # === Select frequency distribution ===
    if freq_dist == 'lorentzian':
        omega = np.random.standard_cauchy(N) * freq_width
        omega = np.clip(omega, -50 * freq_width, 50 * freq_width)
    else:
        omega = np.random.normal(0, freq_width, N)

    theta0 = np.random.uniform(0, 2 * np.pi, N)

    _ = kuramoto_numba(theta0, 0.1, omega, adj_matrix)

    t_current = 0
    theta_current = theta0.copy()
    adj_matrix_current = adj_matrix.copy()

    times = []
    R_values = []

    while t_current < tmax:
        t_end = np.round(min(t_current + t_interval, tmax), 6)
        t_eval = np.round(np.arange(t_current, t_end, dt), 6)

        sol = solve_ivp(
            kuramoto_wrapper,
            [t_current, t_end],
            theta_current,
            t_eval=t_eval,
            args=(K, omega, adj_matrix_current)
        )

        for i, t in enumerate(sol.t):
            times.append(t)
            R = calculate_order_parameter(sol.y[:, i])
            R_values.append(R)

        theta_current = np.mod(sol.y[:, -1], 2 * np.pi)
        del sol

        actual_attack_end = tmax if attack_end is None else attack_end
        if t_end < tmax and eps != 0.0 and attack_start <= t_current < actual_attack_end:
            psi = np.angle(np.mean(np.exp(1j * theta_current)))

            if attack_type == 'node':
                if random_attack:
                    theta_current = theta_current + np.random.choice([-eps, eps], size=len(theta_current))
                else:
                    theta_current = theta_current + eps * np.sign(np.sin(psi - theta_current))
            else:
                raise ValueError(f"Invalid attack type: {attack_type}.")

        t_current = t_end

    return np.array(times), np.array(R_values), run_id

def run_parallel_simulations(num_runs, dt=0.01):
    num_processes = cpu_count()

    run_params = []
    for run_id in range(num_runs):
        run_params.append((
            run_id, dt,
            args.network, args.N, args.kave, args.eps, args.K, args.tmax,
            args.t_interval, args.attack_type, args.random,
            args.attack_start, args.attack_end, args.seed,
            args.freq_dist, args.freq_width  # added
        ))

    with Pool(processes=num_processes) as pool:
        results = pool.map(run_kuramoto_simulation, run_params)

    return results, num_processes

if __name__ == "__main__":
    start_time = time.time()

    os.makedirs(args.output_dir, exist_ok=True)

    # Generate filename (include freq_dist info)
    freq_tag = f"_{args.freq_dist}_w{args.freq_width}" if args.freq_dist != 'gaussian' else ""

    if args.eps == 0:
        if args.network in real_network_data:
            filename_base = f"results_{args.network}_K{args.K}_tmax{args.tmax}_nbruns{args.num_runs}_seed{args.seed}{freq_tag}"
        else:
            filename_base = f"results_{args.network}_N{args.N}_kave{args.kave}_K{args.K}_tmax{args.tmax}_nbruns{args.num_runs}_seed{args.seed}{freq_tag}"
    else:
        if args.network in real_network_data:
            filename_base = f"results_{args.network}_K{args.K}_tmax{args.tmax}_{args.attack_type}_attack_eps{args.eps}_interval{args.t_interval}_nbruns{args.num_runs}_seed{args.seed}{freq_tag}"
        else:
            filename_base = f"results_{args.network}_N{args.N}_kave{args.kave}_K{args.K}_tmax{args.tmax}_{args.attack_type}_attack_eps{args.eps}_interval{args.t_interval}_nbruns{args.num_runs}_seed{args.seed}{freq_tag}"

        if args.attack_start > 0:
            filename_base += f"_start{args.attack_start}"
        if args.attack_end is not None:
            filename_base += f"_end{args.attack_end}"
        if args.random:
            filename_base += "_random"

    combined_csv_filename = os.path.join(args.output_dir, f"{filename_base}_all_runs.csv")

    if os.path.exists(combined_csv_filename):
        print(f"Results already exist: {combined_csv_filename}")
        print("Delete the file to recalculate.")
        exit(0)

    np.random.seed(args.seed)

    results, num_processes = run_parallel_simulations(num_runs=args.num_runs)

    print(f"Completed {args.num_runs} simulations:")
    print(f"- Network: {args.network}, N={args.N}, kave={args.kave}")
    print(f"- K={args.K}, eps={args.eps}")
    print(f"- Frequency distribution: {args.freq_dist} (width={args.freq_width})")
    print(f"- Processes: {num_processes}")

    ref_times = results[0][0]
    data_dict = {'time': ref_times}

    for times, R_values, run_id in results:
        if len(times) != len(ref_times) or not np.allclose(times, ref_times):
            from scipy.interpolate import interp1d
            f = interp1d(times, R_values, bounds_error=False, fill_value="extrapolate")
            data_dict[f'R_run{run_id}'] = f(ref_times)
        else:
            data_dict[f'R_run{run_id}'] = R_values

    combined_df = pd.DataFrame(data_dict)
    R_columns = [col for col in combined_df.columns if col.startswith('R_run')]
    combined_df['R_mean'] = combined_df[R_columns].mean(axis=1)

    combined_df.to_csv(combined_csv_filename, index=False)
    print(f"Saved: {combined_csv_filename}")

    plt.plot(ref_times, combined_df['R_mean'], 'b-', linewidth=2)
    plt.xlabel('Time')
    plt.ylabel('Order Parameter R (Mean)')
    plt.ylim(0, 1)
    plt_filename = os.path.join(args.output_dir, f"R_vs_t_{filename_base}.png")
    plt.savefig(plt_filename, dpi=300, bbox_inches='tight')
    print(f"Saved plot: {plt_filename}")

    print(f"Total runtime: {time.time() - start_time:.2f}s")