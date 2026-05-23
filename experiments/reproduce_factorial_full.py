"""Reproduce full factorial post-hoc pipeline (local, no GPU required).

Steps:
  1. merge_factorial_canonical.py  — merge 3 source JSONs into canonical file
  2. factorial_stat_tests.py       — paired t-test, Wilcoxon, Cohen's d, 95% CI, Holm-Bonferroni
  3. (Diagnostics require GPU — run diagnostics_nslinear_5seed.py on cloud separately)

Usage:
    python reproduce_factorial_full.py
"""

import subprocess, sys, os

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

STEPS = [
    ("merge_factorial_canonical.py", "Merge 3 factorial JSONs into canonical file"),
    ("factorial_stat_tests.py",      "Paired statistical tests (12 comparisons)"),
]

GPU_STEPS = [
    ("factorial_diagnostics.py --setting D2 --seed 0 --max-iter 2000",
     "Full-cell diagnostics seed 0"),
    ("factorial_diagnostics.py --setting D2 --seed 3 --max-iter 2000",
     "Full-cell diagnostics seed 3"),
    ("diagnostics_nslinear_5seed.py --setting D2 --seeds 0 1 2 3 4 --max-iter 2000",
     "5-seed NS+Linear diagnostics with selectivity index"),
]


def run_step(script, description):
    print(f"\n{'='*60}")
    print(f"  {description}")
    print(f"  Script: {script}")
    print(f"{'='*60}")
    path = os.path.join(_SCRIPT_DIR, script)
    result = subprocess.run([sys.executable, path], cwd=_SCRIPT_DIR,
                            capture_output=False)
    if result.returncode != 0:
        print(f"  FAILED (exit code {result.returncode})")
        return False
    print(f"  OK")
    return True


def main():
    print("=" * 60)
    print("  Factorial Post-Hoc Reproduce Pipeline")
    print("=" * 60)

    all_ok = True
    for script, desc in STEPS:
        if not run_step(script, desc):
            all_ok = False

    print(f"\n{'='*60}")
    print("  GPU steps (run on cloud with 'conda activate jrngc_bw'):")
    print(f"{'='*60}")
    for script, desc in GPU_STEPS:
        print(f"  python -u experiments/{script} --output experiments/diagnostics_...json")

    if all_ok:
        print("\nLocal pipeline complete. Outputs in results/raw/.")
        return 0
    else:
        print("\nSome steps failed. Check logs above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
