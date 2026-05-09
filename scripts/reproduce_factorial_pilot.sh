#!/usr/bin/env bash
# Reproduce: Factorial ablation pilot calibration
#
# Runs the pilot calibration for the {Stationary, Non-Stationary} × {Linear, Nonlinear}
# factorial design. Tests 3 difficulty settings (A/B/C) with seeds 0-2 to find the
# setting where baseline AUROC falls in the 0.75-0.90 range.
#
# Settings (per expert specification):
#   A: coeff_scale=0.25 noise=0.20 regime=0.30 nonlinear=0.50
#   B: coeff_scale=0.20 noise=0.30 regime=0.40 nonlinear=0.75
#   C: coeff_scale=0.22 noise=0.25 regime=0.60 nonlinear=0.50
#
# All settings use d=10, T=600, lag=3.
#
# Output: results/raw/factorial_pilot_results.json
#
# Usage:
#   bash scripts/reproduce_factorial_pilot.sh
#   bash scripts/reproduce_factorial_pilot.sh --setting A --max-iter 2000

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJ_ROOT"

export PYTHONPATH="$PROJ_ROOT:${PYTHONPATH:-}"

# Resolve JRNGC path
if [ -z "${JRNGC_PATH:-}" ]; then
    if [ -d "$PROJ_ROOT/../JRNGC" ]; then
        export JRNGC_PATH="$PROJ_ROOT/../JRNGC"
    fi
fi

# Resolve results directory
if [ -z "${ISTF_RESULTS_DIR:-}" ]; then
    export ISTF_RESULTS_DIR="$PROJ_ROOT/results/raw"
fi

mkdir -p "$ISTF_RESULTS_DIR"

SETTING="${1:-all}"  # A, B, C, or all
MAX_ITER="${2:-2000}"
SEEDS="${3:-0,1,2}"

echo "=== ISTF-Mamba: Factorial Pilot Calibration ==="
echo "Setting:   $SETTING"
echo "Max iter:  $MAX_ITER"
echo "Seeds:     $SEEDS"
echo "Results:   $ISTF_RESULTS_DIR/factorial_pilot_results.json"
echo ""

python experiments/test_interaction_ablation.py \
    --setting "$SETTING" \
    --max-iter "$MAX_ITER" \
    --seeds "$SEEDS" \
    --output "$ISTF_RESULTS_DIR/factorial_pilot_results.json"

echo ""
echo "Factorial pilot complete."
