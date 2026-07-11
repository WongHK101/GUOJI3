from __future__ import annotations

import copy
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
for path in (PROJECT_ROOT, SRC_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiments.freeze_phase8_final_confirmation import FIELDS, matrix_rows  # noqa: E402
from experiments.phase8_gpu_runner import instantiate_handle, model_config  # noqa: E402
from phase8_coverage import as_raw_bdt, build_stratified_lag_schedule  # noqa: E402
from phase8_final_protocol import (  # noqa: E402
    validate_final_authorization,
    validate_final_run_matrix,
    write_csv,
)
from phase8_final_results import _pilot_gate, select_eligible_lambda  # noqa: E402
from phase8_protocol import file_sha256, load_json  # noqa: E402


CONFIG = PROJECT_ROOT / "configs" / "phase8_final" / "phase8_lambda_tradeoff_config.json"
MATRIX = PROJECT_ROOT / "configs" / "phase8_final" / "phase8_lambda_tradeoff_matrix.csv"


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_final_lambda_matrix_is_exact_and_comparator_free():
    report = validate_final_run_matrix(CONFIG, MATRIX)
    assert report["passed"], report["failures"]
    assert report["record_count"] == 18
    resolved = report["resolved_records"]
    assert {row["raw_chain_lambda"] for row in resolved} == {0.0003, 0.001, 0.003}
    assert {row["data_seed"] for row in resolved} == {12001, 12002, 12003}
    assert {row["model_seed"] for row in resolved} == {22001, 22002}
    assert {row["method"] for row in resolved} == {"coverage_aligned_raw_chain"}
    assert all(row["max_iter"] == 2000 and row["gating_checkpoint"] == "final" for row in resolved)


def _record(value: float) -> dict:
    return {
        "block": "repair_lambda_tradeoff",
        "method": "coverage_aligned_raw_chain",
        "d": 3,
        "K": 1,
        "d_cond": 2,
        "raw_chain_lambda": value,
    }


def test_lambda_is_applied_without_changing_initial_model_or_prediction():
    config = load_json(CONFIG)
    low_record = _record(0.0003)
    high_record = _record(0.003)
    assert model_config(low_record, config).jacobian_lam == pytest.approx(0.0003)
    assert model_config(high_record, config).jacobian_lam == pytest.approx(0.003)

    torch.manual_seed(917)
    low = instantiate_handle(low_record, config)
    torch.manual_seed(917)
    high = instantiate_handle(high_record, config)
    for name, tensor in low.state_dict().items():
        assert torch.equal(tensor, high.state_dict()[name])

    raw_np = np.random.default_rng(55).normal(size=(3, 20)).astype(np.float32)
    raw_low = as_raw_bdt(raw_np, device=low.device, dtype=low.dtype, require_grad=True)
    raw_high = as_raw_bdt(raw_np, device=high.device, dtype=high.dtype, require_grad=True)
    indices = np.arange(1, 20)
    assert torch.equal(low.predict_from_raw(raw_low, indices), high.predict_from_raw(raw_high, indices))
    entry = build_stratified_lag_schedule(T=20, lag=1, d_out=3, max_iter=1, seed=32001)[0]
    low_components = low.loss_components(raw_low, entry)
    high_components = high.loss_components(raw_high, entry)
    assert float(high_components["jacobian_penalty"] / low_components["jacobian_penalty"]) == pytest.approx(10.0)
    assert float(high_components["fixed_target_prediction_mse"]) == pytest.approx(
        float(low_components["fixed_target_prediction_mse"]), abs=1e-8
    )


def test_confirmation_matrix_is_exactly_40_rows_and_uses_selected_lambda(tmp_path: Path):
    matrix = tmp_path / "confirmation.csv"
    write_csv(matrix, matrix_rows(0.001), FIELDS)
    config = copy.deepcopy(load_json(CONFIG))
    config.update({
        "config_name": "test_confirmation",
        "execution_stage": "confirmation",
        "selected_lambda": 0.001,
        "run_matrix": {"path": matrix.name, "sha256": file_sha256(matrix), "record_count": 40},
    })
    config_path = tmp_path / "confirmation.json"
    _write_json(config_path, config)
    report = validate_final_run_matrix(config_path, matrix)
    assert report["passed"], report["failures"]
    assert report["record_count"] == 40
    repair = [row for row in report["resolved_records"] if row["method"] == "coverage_aligned_raw_chain"]
    assert len(repair) == 10
    assert all(row["raw_chain_lambda"] == pytest.approx(0.001) for row in repair)


def test_confirmation_authorization_requires_matching_token(tmp_path: Path):
    authorization = {
        "authorization": "GPT_APPROVED_PHASE8_FINAL_CONDITIONAL_CONFIRMATION",
        "release_commit": "abc",
        "config_sha256": "cfg",
        "run_matrix_sha256": "mat",
        "allowed_blocks": ["repair_confirmation"],
        "selected_lambda": 0.001,
        "pilot_aggregate_sha256": "pilot",
        "estimator_schedule_sha256": "schedule",
    }
    token = {
        "authorization": "GPT_APPROVED_PHASE8_FINAL_CONDITIONAL_CONFIRMATION",
        "pilot_go_passed": True,
        "source_commit": "abc",
        "config_sha256": "cfg",
        "run_matrix_sha256": "mat",
        "estimator_schedule_sha256": "schedule",
        "selected_lambda": 0.001,
        "pilot_aggregate_sha256": "pilot",
    }
    auth_path = tmp_path / "auth.json"
    token_path = tmp_path / "token.json"
    _write_json(auth_path, authorization)
    _write_json(token_path, token)
    accepted = validate_final_authorization(
        auth_path,
        release_commit="abc",
        config_sha256="cfg",
        matrix_sha256="mat",
        phase="gated_confirmation",
        block="repair_confirmation",
        confirmation_token_path=token_path,
    )
    assert accepted["selected_lambda"] == 0.001
    token["selected_lambda"] = 0.003
    _write_json(token_path, token)
    with pytest.raises(PermissionError):
        validate_final_authorization(
            auth_path,
            release_commit="abc",
            config_sha256="cfg",
            matrix_sha256="mat",
            phase="gated_confirmation",
            block="repair_confirmation",
            confirmation_token_path=token_path,
        )


def _metrics(auroc: float, auprc: float, coefficient: float, mse: float) -> dict:
    return {
        "auroc": auroc,
        "auprc": auprc,
        "coefficient_r": coefficient,
        "fixed_target_prediction_mse": mse,
        "jacobian_penalty": 0.01,
        "total_regularized_objective": mse + 0.01,
        "partial_total_pearson": 0.80,
        "partial_total_topk_jaccard": 0.70,
        "M_missing": 0.10,
        "tail_mass_mean": 0.20,
        "tail_mass_median": 0.20,
    }


def _pilot_rows(repair_mse: float) -> list:
    rows = []
    for seed in (12001, 12002, 12003):
        rows.append({
            "data_seed": seed,
            "comparators": {
                "concat_x_only": _metrics(0.50, 0.40, 0.30, 0.10),
                "baseline_jrngc": _metrics(0.70, 0.60, 0.55, 0.11),
                "full_aux_equal_lambda": _metrics(0.58, 0.50, 0.40, 0.105),
                "full_aux_lc10": _metrics(0.59, 0.51, 0.41, 0.104),
            },
            "repairs": {"lambda_0.001": _metrics(0.62, 0.54, 0.46, repair_mse)},
        })
    return rows


def test_complete_pilot_gate_includes_pure_mse_limit():
    config = load_json(CONFIG)
    mechanism = [
        {"data_seed": seed, "direct_gate_passed": True, "historical_gate_passed": True}
        for seed in (12001, 12002, 12003)
    ]
    passed = _pilot_gate(
        data_seed_rows=_pilot_rows(0.105),
        lambda_value=0.001,
        config=config,
        semantic_compute_passed=True,
        mechanism_rows=mechanism,
    )
    assert passed["eligible"]
    failed = _pilot_gate(
        data_seed_rows=_pilot_rows(0.125),
        lambda_value=0.001,
        config=config,
        semantic_compute_passed=True,
        mechanism_rows=mechanism,
    )
    assert not failed["eligible"]
    assert not failed["repair_minus_concat"]["pure_mse_gate_passed"]


def test_candidate_selection_uses_mse_then_auroc_then_smaller_lambda():
    def report(value: float, mse: float, auroc: float):
        return {
            "lambda": value,
            "eligible": True,
            "repair_minus_concat": {"means": {"relative_mse_degradation": mse, "delta_auroc": auroc}},
        }

    assert select_eligible_lambda([report(0.003, 0.04, 0.20), report(0.001, 0.03, 0.10)])["lambda"] == 0.001
    assert select_eligible_lambda([report(0.003, 0.03, 0.20), report(0.001, 0.03, 0.10)])["lambda"] == 0.003
    assert select_eligible_lambda([report(0.003, 0.03, 0.20), report(0.001, 0.03, 0.20)])["lambda"] == 0.001
    assert select_eligible_lambda([{**report(0.001, 0.01, 0.3), "eligible": False}]) is None
