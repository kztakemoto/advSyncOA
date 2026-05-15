#!/bin/bash
# ============================================================
# N-dependence scan (saves R values from individual runs)
# Purpose: Verify N- and eps-dependence of escape probability P_escape
#
# Usage:
#   bash run_kuramers.sh
#
# Output:
#   JSON files for each condition saved under results_kuramers/
# ============================================================

SCRIPT="kuramoto_alltoall_kuramers.py"  # version that saves individual runs
OUTDIR="results_kuramers"
mkdir -p "$OUTDIR"

DELTA=0.5
TAU=0.3
TMAX=80
NUM_RUNS=10000
SEED=42

N_LIST="200 220 240 260 280 300 320 350 400 450 500 550 600 650 700 800 900 1000 1200 1500 2000 2500 3000 4000 5000"
K_RATIO_LIST="2.0 3.0"
EPS_LIST="-0.03 -0.05 -0.07"

KC=$(python3 -c "print(2 * $DELTA)")

for eps in $EPS_LIST; do
    for ratio in $K_RATIO_LIST; do
        K=$(python3 -c "print($ratio * $KC)")
        for N in $N_LIST; do
            OUT="$OUTDIR/N${N}_Kratio${ratio}_eps${eps}.json"
            echo "Running: N=$N, K/Kc=$ratio, eps=$eps ..."
            python3 "$SCRIPT" \
                --N "$N" \
                --K "$K" \
                --eps "$eps" \
                --Delta "$DELTA" \
                --tau "$TAU" \
                --tmax "$TMAX" \
                --num_runs "$NUM_RUNS" \
                --seed "$SEED" \
                --output "$OUT"
            echo "  -> Saved $OUT"
        done
    done
done

echo "Done. Results in $OUTDIR/"