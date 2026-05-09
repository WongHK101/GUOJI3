#!/usr/bin/env bash
# Reproduce: Shortcut learning diagnostics (three-way mask intervention)
#
# Runs the mask intervention experiment that demonstrates Concat JRNGC
# routes prediction through auxiliary channel c rather than causal input x.
#
# Output: results/raw/mask_supplement_results.json
#
# Usage:
#   bash scripts/reproduce_diagnostics.sh
#   bash scripts/reproduce_diagnostics.sh --max-iter 2000 --device cuda:0

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

echo "=== ISTF-Mamba: Shortcut Learning Diagnostics ==="
echo "Project root: $PROJ_ROOT"
echo "JRNGC path:   ${JRNGC_PATH:-not found}"
echo "Data dir:     ${ISTF_DATA_DIR:-not set}"
echo "Results dir:  $ISTF_RESULTS_DIR"
echo ""

python experiments/test_mask_supplement.py "$@"

echo ""
echo "Diagnostics complete. Results in $ISTF_RESULTS_DIR/mask_supplement_results.json"
