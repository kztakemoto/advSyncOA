#!/bin/bash
#
# run_oa_verification.sh
#
# OA theory verification: compute R vs K transition curves via simulation
# and compare with theory. Three cases: all-to-all (K/N normalization),
# ER, and BA (no normalization).
#
# Usage:
#   chmod +x run_oa_verification.sh
#   ./run_oa_verification.sh                    # default
#   ./run_oa_verification.sh --N_net 1000       # change N for ER/BA
#   ./run_oa_verification.sh --kave 12          # change mean degree
#   ./run_oa_verification.sh --skip_alltoall    # skip all-to-all
#
# Requirements:
#   - kuramoto_alltoall.py (all-to-all, K/N normalization)
#   - kuramoto_networks.py (network version, no normalization, supports --freq_dist lorentzian)

set -e

# ============================================================
# Default parameters
# ============================================================
N_NET=1000
N_ALLTOALL=5000
KAVE=6
NRUNS=100
SEED=42
TMAX=80.0
TAU=0.3
DELTA=0.5
EPS_LIST="0.03,0.05,0.07"
SKIP_ALLTOALL=0

# ============================================================
# Parse command-line arguments
# ============================================================
while [[ $# -gt 0 ]]; do
    case $1 in
        --N_net)      N_NET="$2";      shift 2 ;;
        --N_alltoall) N_ALLTOALL="$2"; shift 2 ;;
        --kave)       KAVE="$2";       shift 2 ;;
        --nruns)      NRUNS="$2";      shift 2 ;;
        --seed)       SEED="$2";       shift 2 ;;
        --tmax)       TMAX="$2";       shift 2 ;;
        --tau)        TAU="$2";        shift 2 ;;
        --Delta)      DELTA="$2";      shift 2 ;;
        --eps_list)   EPS_LIST="$2";   shift 2 ;;
        --skip_alltoall) SKIP_ALLTOALL=1; shift 1 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done


OUTDIR="results_oa_N${N_NET}_kave${KAVE}"
mkdir -p ${OUTDIR}

# ============================================================
# Auto-compute Kc and K scan range
# ============================================================

python3 -c "
import numpy as np
from scipy.stats import poisson

Delta = ${DELTA}
kave = ${KAVE}
N = ${N_NET}

# ER: Poisson distribution (N-independent)
k_max = int(kave + 5*np.sqrt(kave))
k = np.arange(1, k_max+1)
Pk = poisson.pmf(k, kave)
Pk /= Pk.sum()
Kc_er = 2 * Delta * np.sum(k*Pk) / np.sum(k**2*Pk)

# BA: sample degree distribution directly from NetworkX (finite-size)
m = int(kave / 2)
import networkx as nx
all_deg = []
for s in range(30):
    g = nx.barabasi_albert_graph(N, m, seed=s*137)
    all_deg.extend([d for _, d in g.degree()])
all_deg = np.array(all_deg, dtype=float)
k_mean_ba = np.mean(all_deg)
k2_ba = np.mean(all_deg**2)
Kc_ba = 2 * Delta * k_mean_ba / k2_ba

def make_range(Kc, n=20):
    return ' '.join(f'{v:.5f}' for v in np.linspace(0.3*Kc, 3.0*Kc, n))

with open('${OUTDIR}/_params.sh', 'w') as f:
    f.write(f'KC_ER={Kc_er:.6f}\n')
    f.write(f'KC_BA={Kc_ba:.6f}\n')
    f.write(f'K_RANGE_ER=\"{make_range(Kc_er)}\"\n')
    f.write(f'K_RANGE_BA=\"{make_range(Kc_ba)}\"\n')

print(f'Kc(ER) = {Kc_er:.6f}')
print(f'Kc(BA) = {Kc_ba:.6f}  (<k^2>={k2_ba:.1f}, N={N})')
print(f'<k> = {kave}, Delta = {Delta}')
"

# Load parameters
source ${OUTDIR}/_params.sh

echo ""
echo "============================================================"
echo " OA theory verification simulation"
echo " ER/BA: N = ${N_NET}, <k> = ${KAVE}"
echo " All-to-all: N = ${N_ALLTOALL}"
echo " Delta = ${DELTA}, tau = ${TAU}"
echo " Kc(ER) = ${KC_ER}"
echo " Kc(BA) = ${KC_BA}"
echo " eps = ${EPS_LIST}"
echo " Output: ${OUTDIR}"
echo " ${NRUNS} runs per condition"
echo "============================================================"

FREQ_OPT="--freq_dist lorentzian --freq_width ${DELTA}"
EPS_SCAN="0.0 0.03 -0.03 0.05 -0.05 0.07 -0.07"

# ================================================================
# PART 1: All-to-all model (K/N normalization, K_c = 2*Delta)
# ================================================================

echo ""
echo "=== PART 1: All-to-all ==="
echo ""

ALLTOALL_JSON=$(ls ${OUTDIR}/K_scan_N*.json 2>/dev/null | head -1)
if [ "${SKIP_ALLTOALL}" = "1" ]; then
    echo "  Skipping (--skip_alltoall specified)"
elif [ -n "${ALLTOALL_JSON}" ]; then
    echo "  Using existing results: ${ALLTOALL_JSON}"
    echo "  (Delete ${ALLTOALL_JSON} to recompute)"
else
    python kuramoto_alltoall.py \
        --N ${N_ALLTOALL} \
        --Delta ${DELTA} \
        --eps_list ${EPS_LIST} \
        --tmax ${TMAX} \
        --tau ${TAU} \
        --num_runs ${NRUNS} \
        --seed ${SEED} \
        --output_dir ${OUTDIR} \
        --mode K_scan
    echo "  Done"
fi

# ================================================================
# PART 2: ER network (no normalization)
# ================================================================

echo ""
echo "=== PART 2: ER (Kc = ${KC_ER}) ==="
echo ""

for K in ${K_RANGE_ER}; do
    for EPS in ${EPS_SCAN}; do
        PATTERN="${OUTDIR}/results_ER_N${N_NET}_kave${KAVE}*_K${K}_*eps${EPS}_*_all_runs.csv"
        if ls ${PATTERN} 1>/dev/null 2>&1; then
            echo "  ER: K=${K}, eps=${EPS} ... skip (existing)"
            continue
        fi
        echo "  ER: K=${K}, eps=${EPS}"
        python kuramoto_networks.py \
            --network ER \
            --N ${N_NET} \
            --kave ${KAVE} \
            --K ${K} \
            --eps ${EPS} \
            --tmax ${TMAX} \
            --t_interval ${TAU} \
            --num_runs ${NRUNS} \
            --seed ${SEED} \
            --output_dir ${OUTDIR} \
            ${FREQ_OPT}
    done
done

echo "  Done"

# ================================================================
# PART 3: BA network (no normalization)
# ================================================================

echo ""
echo "=== PART 3: BA (Kc = ${KC_BA}) ==="
echo ""

for K in ${K_RANGE_BA}; do
    for EPS in ${EPS_SCAN}; do
        PATTERN="${OUTDIR}/results_BA_N${N_NET}_kave${KAVE}*_K${K}_*eps${EPS}_*_all_runs.csv"
        if ls ${PATTERN} 1>/dev/null 2>&1; then
            echo "  BA: K=${K}, eps=${EPS} ... skip (existing)"
            continue
        fi
        echo "  BA: K=${K}, eps=${EPS}"
        python kuramoto_networks.py \
            --network BA \
            --N ${N_NET} \
            --kave ${KAVE} \
            --K ${K} \
            --eps ${EPS} \
            --tmax ${TMAX} \
            --t_interval ${TAU} \
            --num_runs ${NRUNS} \
            --seed ${SEED} \
            --output_dir ${OUTDIR} \
            ${FREQ_OPT}
    done
done

echo "  Done"


# ================================================================
# Completion message
# ================================================================

echo ""
echo "============================================================"
echo " All simulations complete"
echo " Results: ${OUTDIR}/"
echo "============================================================"
echo ""
echo "Next step:"
echo "  python analyze_oa_verification.py --results_dir ${OUTDIR} --kave ${KAVE} --N_net ${N_NET}"
echo ""
echo "Kc values:"
echo "  All-to-all (K/N normalization):  Kc = $(echo "2 * ${DELTA}" | bc)"
echo "  ER (no normalization):           Kc = ${KC_ER}"
echo "  BA (no normalization):           Kc = ${KC_BA}"