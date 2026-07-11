"""Run the non-evidentiary Phase 8 CPU implementation preflight.

This entry point never trains a formal record and refuses non-CPU execution.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from phase8_coverage import (  # noqa: E402
    CoverageAlignedRawChainJRNGC,
    Phase8ModelConfig,
    as_raw_bdt,
    build_stratified_lag_schedule,
    comparator_parity_audit,
    deterministic_schedule_audit,
    direct_indirect_chain_decomposition_audit,
    estimator_exact_reference_audit,
    finite_difference_total_raw_chain_audit,
    fixed_target_concat_interventions,
    make_legacy_baseline,
    make_legacy_concat,
)
from phase8_protocol import (  # noqa: E402
    critical_source_manifest,
    environment_snapshot,
    file_sha256,
    formal_estimator_schedule_report,
    git_commit,
    validate_run_matrix,
)


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "phase8" / "phase8_execution_lock.json"
DEFAULT_MATRIX = PROJECT_ROOT / "configs" / "phase8" / "phase8_run_matrix.csv"
CRITICAL_SOURCE_PATHS = [
    "src/phase8_coverage.py",
    "src/phase8_protocol.py",
    "src/phase8_results.py",
    "src/phase8_training.py",
    "experiments/phase8_cpu_preflight.py",
    "experiments/phase8_gpu_runner.py",
    "experiments/phase8_numerical_forensics.py",
    "experiments/execute_phase8_stage.py",
    "experiments/validate_phase8_gpu_preflight.py",
    "experiments/aggregate_phase8.py",
    "tests/test_phase8_coverage.py",
    "tests/test_phase8_execution.py",
    "configs/phase8/phase8_execution_lock.json",
    "configs/phase8/phase8_run_matrix.csv",
    "configs/phase8/confirmation_release_token.schema.json",
    "src/mamba_jrngc_pilot.py",
    "src/minimal_mamba.py",
    "src/repaired_istf.py",
    "src/knowledge_metrics.py",
    "src/nonstationary_var.py",
    "experiments/risk_mitigation_20260515/run_full_aux_penalty.py",
    "experiments/test_mask_supplement.py",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--release-commit", default=None)
    return parser.parse_args()


def save_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def contains_prohibited_prediction_loss_key(value: object) -> bool:
    if isinstance(value, dict):
        return any(key == "prediction_loss" or contains_prohibited_prediction_loss_key(item) for key, item in value.items())
    if isinstance(value, (list, tuple)):
        return any(contains_prohibited_prediction_loss_key(item) for item in value)
    return False


def provenance_field_validation() -> dict:
    torch.manual_seed(8141)
    rng = np.random.default_rng(8142)
    x = rng.normal(scale=0.2, size=(3, 10)).astype(np.float32)
    cfg = Phase8ModelConfig(d=3, lag=2, layers=1, hidden=5, d_cond=2, d_state=2)
    baseline = make_legacy_baseline(cfg)
    concat = make_legacy_concat(cfg)
    baseline_components = baseline.loss_components(x)
    concat_components = concat.loss_components(x)
    repair_cfg = Phase8ModelConfig(d=3, lag=1, layers=1, hidden=5, d_cond=2, d_state=2)
    repair = CoverageAlignedRawChainJRNGC(repair_cfg)
    repair_raw = as_raw_bdt(x, device=torch.device("cpu"), dtype=torch.float32, require_grad=True)
    repair_entry = build_stratified_lag_schedule(T=10, lag=1, d_out=3, max_iter=1, seed=8143)[0]
    repair_components = repair.loss_components(repair_raw, repair_entry)
    interventions = fixed_target_concat_interventions(concat, x, perturbation_seed=31101)
    required_components = {
        "fixed_target_prediction_mse",
        "jacobian_penalty",
        "total_regularized_objective",
    }
    required_intervention = {
        "fixed_target_prediction_mse",
        "fixed_target_prediction_mse_delta",
        "legacy_objective_delta",
    }
    required_repair_components = required_components | {
        "nominal_jacobian_penalty",
        "historical_jacobian_penalty",
    }
    passed = (
        set(baseline_components) == required_components
        and set(concat_components) == required_components
        and set(repair_components) == required_repair_components
        and torch.allclose(
            repair_components["nominal_jacobian_penalty"] + repair_components["historical_jacobian_penalty"],
            repair_components["jacobian_penalty"],
        )
        and required_intervention.issubset(interventions)
        and not contains_prohibited_prediction_loss_key(baseline_components)
        and not contains_prohibited_prediction_loss_key(concat_components)
        and not contains_prohibited_prediction_loss_key(interventions)
        and interventions["target_policy"] == "clean_raw_target_fixed"
    )
    return {
        "passed": bool(passed),
        "required_component_fields": sorted(required_components),
        "required_intervention_fields": sorted(required_intervention),
        "baseline_fields": sorted(baseline_components),
        "concat_fields": sorted(concat_components),
        "repair_fields": sorted(repair_components),
        "required_repair_component_fields": sorted(required_repair_components),
        "intervention_fields": sorted(interventions),
        "prohibited_prediction_loss_key_found": contains_prohibited_prediction_loss_key({
            "baseline": baseline_components,
            "concat": concat_components,
            "repair": repair_components,
            "interventions": interventions,
        }),
        "target_policy": interventions["target_policy"],
        "legacy_track_label": "legacy_objective_delta",
        "primary_intervention_label": "fixed_target_prediction_mse_delta",
    }


def main() -> int:
    args = parse_args()
    if args.device != "cpu":
        raise RuntimeError("Phase 8 implementation preflight is CPU-only")
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in {"-1", ""}:
        raise RuntimeError("Set CUDA_VISIBLE_DEVICES=-1 before CPU preflight")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    args.output_root.mkdir(parents=True, exist_ok=True)
    started = time.time()

    # Freeze provenance before any semantic or mathematical preflight runs.
    formal_schedules = formal_estimator_schedule_report()
    environment = environment_snapshot(deterministic_settings={
        "torch_use_deterministic_algorithms": True,
        "torch_num_threads": torch.get_num_threads(),
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
    })
    manifest = critical_source_manifest(
        PROJECT_ROOT,
        CRITICAL_SOURCE_PATHS,
        release_commit=args.release_commit or git_commit(PROJECT_ROOT),
    )
    save_json(args.output_root / "environment.json", environment)
    save_json(args.output_root / "release_source_manifest.json", manifest)
    (args.output_root / "approved_phase8_code_commit.txt").write_text(
        str(manifest["release_commit"]) + "\n",
        encoding="utf-8",
    )
    (args.output_root / "config_sha256.txt").write_text(file_sha256(args.config) + "\n", encoding="utf-8")
    (args.output_root / "run_matrix_sha256.txt").write_text(file_sha256(args.run_matrix) + "\n", encoding="utf-8")
    (args.output_root / "estimator_schedule_sha256.txt").write_text(
        formal_schedules["pilot"]["sha256"] + "\n",  # type: ignore[index]
        encoding="utf-8",
    )

    fd = finite_difference_total_raw_chain_audit()
    chain = direct_indirect_chain_decomposition_audit()
    exact = estimator_exact_reference_audit(draw_count=1536)
    schedule = deterministic_schedule_audit()
    parity = comparator_parity_audit()
    matrix = validate_run_matrix(args.config, args.run_matrix)
    provenance = provenance_field_validation()

    save_json(args.output_root / "finite_difference_and_chain_decomposition_report.json", {
        "passed": bool(fd["passed"] and chain["passed"]),
        "finite_difference": fd,
        "chain_decomposition": chain,
    })
    save_json(args.output_root / "estimator_exact_reference_report.json", exact)
    save_json(args.output_root / "deterministic_schedule_report.json", {
        "passed": bool(schedule["passed"]),
        "schedule_audit": schedule,
        "formal_schedule_hashes": formal_schedules,
    })
    save_json(args.output_root / "comparator_parity_report.json", parity)
    save_json(args.output_root / "run_matrix_dry_validation_report.json", matrix)
    save_json(args.output_root / "provenance_field_validation_report.json", provenance)
    passed = all([
        fd["passed"],
        chain["passed"],
        exact["passed"],
        schedule["passed"],
        parity["passed"],
        matrix["passed"],
        provenance["passed"],
    ])
    summary = {
        "passed": bool(passed),
        "formal_scientific_runs_executed": 0,
        "gpu_used": False,
        "cpu_preflight_only": True,
        "wall_time_seconds": time.time() - started,
        "reports": sorted(path.name for path in args.output_root.iterdir()),
    }
    save_json(args.output_root / "cpu_preflight_summary.json", summary)
    print(json.dumps(summary, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
