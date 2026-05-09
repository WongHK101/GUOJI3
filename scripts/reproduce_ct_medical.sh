#!/usr/bin/env bash
# Reproduce: CT_medical ISTF-Mamba 3-seed main result
#
# This is the primary real-world signal: ISTF-Mamba vs Baseline JRNGC
# on the clinical time-series dataset (d=40, T=1200, 133 edges).
#
# Output: results/raw/ct_medical_3seed_results.json
#
# Usage:
#   bash scripts/reproduce_ct_medical.sh
#   bash scripts/reproduce_ct_medical.sh --max-iter 5000 --lr 5e-4

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

# Resolve data directory
if [ -z "${ISTF_DATA_DIR:-}" ]; then
    if [ -d "$PROJ_ROOT/../JRNGC/data" ]; then
        export ISTF_DATA_DIR="$PROJ_ROOT/../JRNGC/data"
    elif [ -d "$PROJ_ROOT/data" ]; then
        export ISTF_DATA_DIR="$PROJ_ROOT/data"
    fi
fi

# Resolve results directory
if [ -z "${ISTF_RESULTS_DIR:-}" ]; then
    export ISTF_RESULTS_DIR="$PROJ_ROOT/results/raw"
fi

mkdir -p "$ISTF_RESULTS_DIR"

echo "=== ISTF-Mamba: CT_medical 3-Seed Validation ==="
echo "Project root: $PROJ_ROOT"
echo "JRNGC path:   ${JRNGC_PATH:-not found}"
echo "Data dir:     ${ISTF_DATA_DIR:-not set}"
echo "Results dir:  $ISTF_RESULTS_DIR"
echo ""

python experiments/test_ct_medical_3seed.py "$@"

echo ""
echo "CT_medical complete. Results in $ISTF_RESULTS_DIR/ct_medical_3seed_results.json"
