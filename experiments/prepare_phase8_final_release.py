"""Create an external source/config authorization lock for final Phase 8."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
for path in (PROJECT_ROOT, SRC_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from phase8_coverage import build_stratified_lag_schedule, schedule_sha256  # noqa: E402
from phase8_protocol import critical_source_manifest, file_sha256  # noqa: E402
from phase8_final_protocol import validate_final_run_matrix, write_json  # noqa: E402


CRITICAL_FILES = [
    ".gitattributes",
    ".gitignore",
    "configs/phase8_final/phase8_lambda_tradeoff_config.json",
    "configs/phase8_final/phase8_lambda_tradeoff_matrix.csv",
    "docs/phase8_final_bounded_tradeoff_execution_lock.md",
    "experiments/aggregate_phase8_final.py",
    "experiments/execute_phase8_final_stage.py",
    "experiments/freeze_phase8_final_confirmation.py",
    "experiments/phase8_gpu_runner.py",
    "experiments/phase8_cpu_preflight.py",
    "experiments/prepare_phase8_final_release.py",
    "src/phase8_coverage.py",
    "src/phase8_final_protocol.py",
    "src/phase8_final_results.py",
    "src/phase8_protocol.py",
    "src/phase8_results.py",
    "src/phase8_training.py",
    "src/mamba_jrngc_pilot.py",
    "src/minimal_mamba.py",
    "src/knowledge_metrics.py",
    "src/nonstationary_var.py",
    "src/repaired_istf.py",
    "experiments/risk_mitigation_20260515/run_full_aux_penalty.py",
    "tests/test_phase8_final.py",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-matrix", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def git(*args: str) -> str:
    result = subprocess.run(["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or f"git {' '.join(args)} failed")
    return result.stdout.strip()


def main() -> int:
    args = parse_args()
    report = validate_final_run_matrix(args.config, args.run_matrix)
    if not report["passed"]:
        raise RuntimeError(f"Cannot release invalid final matrix: {report['failures']}")
    if git("status", "--porcelain"):
        raise RuntimeError("Final release requires a clean worktree")
    commit = git("rev-parse", "HEAD")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("Release-lock output directory must be empty")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = critical_source_manifest(PROJECT_ROOT, CRITICAL_FILES, release_commit=commit)
    write_json(args.output_dir / "release_source_manifest.json", manifest)
    (args.output_dir / "approved_phase8_code_commit.txt").write_text(commit + "\n", encoding="utf-8")
    schedule = build_stratified_lag_schedule(T=500, lag=1, d_out=8, max_iter=2000, seed=32001)
    authorization = {
        "authorization": "GPT_APPROVED_PHASE8_FINAL_BOUNDED_LAMBDA_TRADEOFF",
        "release_commit": commit,
        "config_sha256": file_sha256(args.config),
        "run_matrix_sha256": file_sha256(args.run_matrix),
        "estimator_schedule_sha256": schedule_sha256(schedule),
        "allowed_blocks": ["repair_lambda_tradeoff"],
        "allowed_lambdas": [0.0003, 0.001, 0.003],
        "record_count": 18,
        "comparator_rerun_allowed": False,
    }
    write_json(args.output_dir / "phase8_lambda_tradeoff_authorization.json", authorization)
    summary = {
        "release_commit": commit,
        "source_manifest_sha256": file_sha256(args.output_dir / "release_source_manifest.json"),
        "config_sha256": file_sha256(args.config),
        "run_matrix_sha256": file_sha256(args.run_matrix),
        "authorization_sha256": file_sha256(args.output_dir / "phase8_lambda_tradeoff_authorization.json"),
        "schedule_sha256": authorization["estimator_schedule_sha256"],
        "critical_file_count": len(CRITICAL_FILES),
    }
    write_json(args.output_dir / "release_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
