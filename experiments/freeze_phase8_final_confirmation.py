"""Freeze conditional Phase 8 confirmation assets after a passing pilot."""

from __future__ import annotations

import argparse
import copy
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
for path in (PROJECT_ROOT, SRC_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from phase8_coverage import build_stratified_lag_schedule, schedule_sha256  # noqa: E402
from phase8_final_protocol import (  # noqa: E402
    CONFIRMATION_DATA_SEEDS,
    CONFIRMATION_METHODS,
    CONFIRMATION_MODEL_SEEDS,
    write_csv,
    write_json,
)
from phase8_protocol import file_sha256, load_json  # noqa: E402


FIELDS = [
    "record_id", "formal_result", "phase", "block", "method", "data_seed", "model_seed",
    "perturbation_seed", "jacobian_seed", "d", "T", "K", "d_cond", "max_iter", "lambda_x",
    "lambda_c", "raw_chain_lambda", "training_attribution_horizon", "primary_graph_score",
    "coefficient_metric", "secondary_history_score", "n_min", "notes",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-report", type=Path, required=True)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def git_value(*args: str) -> str:
    result = subprocess.run(["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or f"git {' '.join(args)} failed")
    return result.stdout.strip()


def matrix_rows(selected_lambda: float):
    rows = []
    for data_seed in CONFIRMATION_DATA_SEEDS:
        for model_seed in CONFIRMATION_MODEL_SEEDS:
            for method in CONFIRMATION_METHODS:
                label = method.replace("coverage_aligned_raw_chain", "repair")
                d_cond = 0 if method == "baseline_jrngc" else 4
                lambda_x = 0.01 if method != "coverage_aligned_raw_chain" else ""
                lambda_c = ""
                raw_chain_lambda = ""
                horizon = "nominal_K"
                if method == "concat_x_only":
                    lambda_c = 0
                elif method == "full_aux_lc10":
                    lambda_c = 0.10
                elif method == "coverage_aligned_raw_chain":
                    lambda_x = ""
                    raw_chain_lambda = selected_lambda
                    horizon = "full_prefix_H499"
                rows.append({
                    "record_id": f"P8-FCON-D{data_seed}-M{model_seed}-{label}",
                    "formal_result": "true",
                    "phase": "gated_confirmation",
                    "block": "repair_confirmation",
                    "method": method,
                    "data_seed": data_seed,
                    "model_seed": model_seed,
                    "perturbation_seed": "",
                    "jacobian_seed": 33001,
                    "d": 8,
                    "T": 500,
                    "K": 1,
                    "d_cond": d_cond,
                    "max_iter": 2000,
                    "lambda_x": lambda_x,
                    "lambda_c": lambda_c,
                    "raw_chain_lambda": raw_chain_lambda,
                    "training_attribution_horizon": horizon,
                    "primary_graph_score": "S_GC_total_nominal",
                    "coefficient_metric": "coefficient_r_total_lag1",
                    "secondary_history_score": "S_reliable_history",
                    "n_min": 50,
                    "notes": "held_out_confirmation_no_further_tuning",
                })
    return rows


def main() -> int:
    args = parse_args()
    pilot = load_json(args.pilot_report)
    if pilot.get("confirmation_eligible") is not True or pilot.get("selected_lambda") is None:
        raise PermissionError("Pilot report does not authorize held-out confirmation")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("Confirmation lock output directory must be empty")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    status = git_value("status", "--porcelain")
    if status:
        raise RuntimeError(f"Source worktree must be clean before confirmation freeze:\n{status}")
    source_commit = git_value("rev-parse", "HEAD")
    selected_lambda = float(pilot["selected_lambda"])
    pilot_sha = file_sha256(args.pilot_report)

    matrix_path = args.output_dir / "phase8_confirmation_matrix.csv"
    write_csv(matrix_path, matrix_rows(selected_lambda), FIELDS)
    matrix_sha = file_sha256(matrix_path)

    config = copy.deepcopy(load_json(args.base_config))
    config.update({
        "config_name": "phase8_final_conditional_confirmation_v1",
        "execution_stage": "confirmation",
        "selected_lambda": selected_lambda,
        "source_commit": source_commit,
        "pilot_aggregate_sha256": pilot_sha,
        "run_matrix": {
            "path": matrix_path.name,
            "sha256": matrix_sha,
            "record_count": 40,
        },
    })
    config_path = args.output_dir / "phase8_confirmation_config.json"
    write_json(config_path, config)
    config_sha = file_sha256(config_path)
    schedule = build_stratified_lag_schedule(T=500, lag=1, d_out=8, max_iter=2000, seed=33001)
    schedule_hash = schedule_sha256(schedule)
    authorization = {
        "authorization": "GPT_APPROVED_PHASE8_FINAL_CONDITIONAL_CONFIRMATION",
        "release_commit": source_commit,
        "config_sha256": config_sha,
        "run_matrix_sha256": matrix_sha,
        "allowed_blocks": ["repair_confirmation"],
        "selected_lambda": selected_lambda,
        "pilot_aggregate_sha256": pilot_sha,
        "estimator_schedule_sha256": schedule_hash,
        "record_count": 40,
    }
    token = {
        "authorization": "GPT_APPROVED_PHASE8_FINAL_CONDITIONAL_CONFIRMATION",
        "pilot_go_passed": True,
        "source_commit": source_commit,
        "config_sha256": config_sha,
        "run_matrix_sha256": matrix_sha,
        "estimator_schedule_sha256": schedule_hash,
        "selected_lambda": selected_lambda,
        "pilot_aggregate_sha256": pilot_sha,
        "no_further_tuning_authorized": True,
    }
    authorization_path = args.output_dir / "phase8_confirmation_authorization.json"
    token_path = args.output_dir / "phase8_confirmation_release_token.json"
    write_json(authorization_path, authorization)
    write_json(token_path, token)
    manifest = {
        "source_commit": source_commit,
        "selected_lambda": selected_lambda,
        "files": {
            path.name: file_sha256(path)
            for path in sorted(args.output_dir.iterdir())
            if path.is_file()
        },
    }
    write_json(args.output_dir / "confirmation_lock_manifest.json", manifest)
    commands = f"""# Frozen Phase 8 held-out confirmation

```bash
python experiments/execute_phase8_final_stage.py \\
  --config {config_path} \\
  --run-matrix {matrix_path} \\
  --release-lock-dir <release-lock-dir> \\
  --authorization {authorization_path} \\
  --confirmation-token {token_path} \\
  --output-root <fresh-confirmation-root>
```
"""
    (args.output_dir / "CONFIRMATION_COMMANDS.md").write_text(commands, encoding="utf-8")
    print(f"selected_lambda={selected_lambda}")
    print(f"config_sha256={config_sha}")
    print(f"run_matrix_sha256={matrix_sha}")
    print(f"schedule_sha256={schedule_hash}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
