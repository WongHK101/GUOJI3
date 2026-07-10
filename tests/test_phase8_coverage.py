"""Non-evidentiary CPU semantic tests for the Phase 8 execution lock."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from phase8_coverage import (  # noqa: E402
    CoverageAlignedRawChainJRNGC,
    Phase8ModelConfig,
    as_raw_bdt,
    build_balanced_lag_schedule,
    coefficient_r_total_lag1,
    comparator_parity_audit,
    deterministic_schedule_audit,
    direct_indirect_chain_decomposition_audit,
    eligible_targets_for_lag,
    estimator_exact_reference_audit,
    extract_attribution_objects,
    finite_difference_total_raw_chain_audit,
    fixed_target_concat_interventions,
    make_legacy_baseline,
    make_legacy_concat,
    make_no_aux_input_space_control,
    sampled_lag_balanced_penalty,
    target_indices,
    total_raw_chain_at_target,
)
from phase8_protocol import (  # noqa: E402
    validate_confirmation_release_token,
    validate_run_matrix,
)
from phase8_training import (  # noqa: E402
    FIXED_FINAL_POLICY,
    LEGACY_RESTORE_POLICY,
    train_with_frozen_checkpoint_policy,
)


os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")


class ToyRawChainModel(nn.Module):
    """Linear raw-history fixture with explicit h=1-first weights."""

    def __init__(self, weights: np.ndarray, lag: int):
        super().__init__()
        values = torch.as_tensor(weights, dtype=torch.float64)
        self.weights = nn.Parameter(values)
        self.d = int(values.shape[0])
        self.lag = lag

    def predict_from_raw(self, raw_bdt, indices, auxiliary_override=None):
        del auxiliary_override
        rows = []
        for u in indices:
            y = torch.zeros(self.d, dtype=raw_bdt.dtype, device=raw_bdt.device)
            for h in range(1, min(u, self.weights.shape[2]) + 1):
                y = y + self.weights[:, :, h - 1] @ raw_bdt[0, :, u - h]
            rows.append(y)
        return torch.stack(rows)

    def predict_partial_from_raw(self, raw_bdt, target_u):
        history = raw_bdt[0, :, target_u - self.lag:target_u].detach().clone().requires_grad_(True)
        y = torch.zeros(self.d, dtype=history.dtype)
        for h in range(1, self.lag + 1):
            y = y + self.weights[:, :, h - 1] @ history[:, self.lag - h]
        return y, history


def _repair_cfg(dtype="float64", lag=2):
    return Phase8ModelConfig(
        d=3,
        lag=lag,
        layers=1,
        hidden=8,
        d_cond=2,
        d_state=2,
        d_conv=2,
        dtype=dtype,
    )


def _x(seed=100, d=3, T=12, dtype=np.float64):
    return np.random.default_rng(seed).normal(scale=0.2, size=(d, T)).astype(dtype)


def test_01_finite_difference_total_raw_chain_nonzero():
    report = finite_difference_total_raw_chain_audit()
    assert report["passed"]
    assert abs(report["autograd"]) > 1e-8


def test_02_direct_plus_indirect_chain_decomposition():
    report = direct_indirect_chain_decomposition_audit()
    assert report["passed"]
    assert report["direct_gradient_norm"] > 0
    assert report["indirect_gradient_norm"] > 0


def test_03_raw_coordinate_orientation_target_source():
    weights = np.zeros((3, 3, 1), dtype=np.float64)
    weights[1, 0, 0] = 4.0
    model = ToyRawChainModel(weights, lag=1)
    result = extract_attribution_objects(model, _x(T=8), true_edge_count=1, n_min=2)
    assert result.s_gc_total[1, 0] == pytest.approx(4.0)
    assert result.s_gc_total[0, 1] == pytest.approx(0.0)


def test_04_nominal_lag_indexing_h1_first():
    weights = np.zeros((2, 2, 3), dtype=np.float64)
    weights[1, 0, 0] = 3.0
    weights[1, 0, 1] = 1.0
    model = ToyRawChainModel(weights, lag=2)
    result = extract_attribution_objects(model, _x(d=2, T=9), n_min=2)
    assert result.j_bar_total[1, 0, 0] == pytest.approx(3.0)
    assert result.j_bar_total[1, 0, 1] == pytest.approx(1.0)
    assert result.s_gc_total[1, 0] == pytest.approx(3.0)


def test_05_eligible_window_count_by_lag():
    T, K = 12, 2
    for h in range(1, T):
        assert len(eligible_targets_for_lag(T, K, h)) == T - max(K, h)


def test_06_prefix_causality_future_perturbation_invariance():
    torch.manual_seed(120)
    model = CoverageAlignedRawChainJRNGC(_repair_cfg()).eval()
    x = _x(seed=121)
    u = 8
    raw = as_raw_bdt(x, device=torch.device("cpu"), dtype=torch.float64)
    base = model.predict_from_raw(raw, [u]).detach()
    changed = x.copy()
    changed[:, u:] += 1000.0
    changed_raw = as_raw_bdt(changed, device=torch.device("cpu"), dtype=torch.float64)
    got = model.predict_from_raw(changed_raw, [u]).detach()
    torch.testing.assert_close(base, got, atol=1e-10, rtol=1e-8)


def test_07_raw_target_isolation():
    torch.manual_seed(130)
    model = CoverageAlignedRawChainJRNGC(_repair_cfg()).eval()
    x = _x(seed=131)
    u = 8
    raw = as_raw_bdt(x, device=torch.device("cpu"), dtype=torch.float64)
    pred = model.predict_from_raw(raw, [u]).detach()
    target = model.raw_targets(raw, [u]).detach()
    changed = x.copy()
    changed[:, u] += 50.0
    changed_raw = as_raw_bdt(changed, device=torch.device("cpu"), dtype=torch.float64)
    changed_pred = model.predict_from_raw(changed_raw, [u]).detach()
    changed_target = model.raw_targets(changed_raw, [u]).detach()
    torch.testing.assert_close(pred, changed_pred, atol=1e-10, rtol=1e-8)
    assert not torch.allclose(target, changed_target)


def test_08_deliberate_detach_negative_test():
    model = CoverageAlignedRawChainJRNGC(_repair_cfg()).eval()
    with pytest.raises(RuntimeError):
        total_raw_chain_at_target(model, _x(), 8, detach_raw=True)


def test_09_second_order_predictor_and_preprocessor_gradients():
    torch.manual_seed(140)
    model = CoverageAlignedRawChainJRNGC(_repair_cfg(dtype="float32", lag=1)).train()
    x = _x(seed=141, dtype=np.float32)
    raw = as_raw_bdt(x, device=torch.device("cpu"), dtype=torch.float32, require_grad=True)
    entry = {
        "iteration": 0,
        "lags": [1, 2],
        "eligible_windows": [6, 7],
        "output_targets": [0, 1],
    }
    penalty = sampled_lag_balanced_penalty(model, raw, entry, create_graph=True)
    penalty.backward()
    predictor = []
    preprocessor = []
    for name, parameter in model.named_parameters():
        if parameter.grad is None or "weight_head" in name:
            continue
        (preprocessor if name.startswith("preprocessor.") else predictor).append(parameter.grad.reshape(-1))
    assert predictor and preprocessor
    for group in [predictor, preprocessor]:
        vector = torch.cat(group)
        assert torch.isfinite(vector).all()
        assert torch.linalg.norm(vector) > 0


def test_10_baseline_partial_total_nominal_equivalence():
    torch.manual_seed(150)
    cfg = Phase8ModelConfig(d=3, lag=1, layers=1, hidden=5, d_cond=2, d_state=2)
    adapter = make_legacy_baseline(cfg)
    x = _x(seed=151, T=10, dtype=np.float32)
    partial = adapter.partial_nominal_score(x)
    total = extract_attribution_objects(adapter, x, n_min=2).s_gc_total
    assert np.max(np.abs(partial - total)) <= 1e-7


def test_11_exact_lag_balanced_objective_and_512_draw_estimator():
    report = estimator_exact_reference_audit(draw_count=512)
    assert report["passed"]
    assert report["relative_objective_error"] <= 0.05
    assert report["parameter_gradient_cosine"] >= 0.95


def test_12_deterministic_schedule_reproduction_and_factor():
    report = deterministic_schedule_audit()
    assert report["passed"]
    assert report["sampled_sum_factor"] == pytest.approx(499 / 32)


def test_13_primary_reliable_and_unrestricted_scores_are_separate():
    T = 30
    weights = np.zeros((2, 2, T - 1), dtype=np.float64)
    weights[1, 0, 0] = 1.0
    weights[1, 0, 2] = 2.0
    weights[1, 0, T - 2] = 100.0
    model = ToyRawChainModel(weights, lag=1)
    result = extract_attribution_objects(model, _x(d=2, T=T), n_min=20)
    assert result.s_gc_total[1, 0] == pytest.approx(1.0)
    assert result.s_reliable_history[1, 0] == pytest.approx(2.0)
    assert result.s_prefix_all[1, 0] == pytest.approx(100.0)
    assert result.prefix_maximizing_lag[1, 0] == T - 1
    assert result.prefix_max_outside_reliable[1, 0]


def test_14_fixed_target_intervention_semantics_and_field_names():
    torch.manual_seed(160)
    cfg = Phase8ModelConfig(d=3, lag=2, layers=1, hidden=5, d_cond=2, d_state=2)
    adapter = make_legacy_concat(cfg)
    x = _x(seed=161, T=10, dtype=np.float32)
    result = fixed_target_concat_interventions(adapter, x, perturbation_seed=31101)
    assert set(result) >= {
        "fixed_target_prediction_mse",
        "fixed_target_prediction_mse_delta",
        "legacy_total_regularized_objective",
        "legacy_objective_delta",
    }
    assert "prediction_loss" not in result
    assert result["target_policy"] == "clean_raw_target_fixed"
    repeated = fixed_target_concat_interventions(adapter, x, perturbation_seed=31101)
    assert result["fixed_target_prediction_mse"] == repeated["fixed_target_prediction_mse"]


def test_15_pure_mse_penalty_total_objective_separation():
    adapter = make_legacy_baseline(Phase8ModelConfig(d=3, lag=1, layers=1, hidden=5))
    components = adapter.loss_components(_x(T=10, dtype=np.float32))
    assert set(components) == {
        "fixed_target_prediction_mse",
        "jacobian_penalty",
        "total_regularized_objective",
    }
    torch.testing.assert_close(
        components["fixed_target_prediction_mse"] + components["jacobian_penalty"],
        components["total_regularized_objective"],
    )


def test_16_comparator_same_weight_same_input_parity_matrix():
    report = comparator_parity_audit()
    assert report["passed"]
    assert set(report["comparators"]) == {
        "baseline",
        "concat_x_only",
        "full_aux_equal",
        "full_aux_lc10",
    }


def test_17_same_seed_repeated_attribution_is_deterministic():
    torch.manual_seed(170)
    model = CoverageAlignedRawChainJRNGC(_repair_cfg()).eval()
    x = _x(seed=171, T=10)
    first = extract_attribution_objects(model, x, n_min=2)
    second = extract_attribution_objects(model, x, n_min=2)
    assert np.max(np.abs(first.s_gc_total - second.s_gc_total)) <= 1e-6
    assert np.max(np.abs(first.s_reliable_history - second.s_reliable_history)) <= 1e-6
    assert np.max(np.abs(first.s_prefix_all - second.s_prefix_all)) <= 1e-6


def test_18_lag1_coefficient_metric_does_not_use_history_maximum():
    lag1 = np.array([[0.0, 0.1], [0.8, 0.0]])
    A = np.array([[0.0, 0.0], [1.0, 0.0]])
    value = coefficient_r_total_lag1(lag1, A)
    assert value is not None and np.isfinite(value)


def test_19_run_matrix_dry_validation_all_135_records():
    report = validate_run_matrix(
        PROJECT_ROOT / "configs" / "phase8" / "phase8_execution_lock.json",
        PROJECT_ROOT / "configs" / "phase8" / "phase8_run_matrix.csv",
    )
    assert report["passed"], report["failures"]
    assert report["record_count"] == 135
    assert report["formal_count"] == 130
    assert report["non_evidentiary_count"] == 5
    assert report["confirmation_count"] == 50
    assert report["forbidden_phase7_data_seed_matches"] == []


def test_20_confirmation_requires_exact_release_token():
    expected = {
        "pilot_go_passed": True,
        "authorization": "GPT_APPROVED_PHASE8_CONFIRMATION",
        "config_sha256": "a" * 64,
        "run_matrix_sha256": "b" * 64,
        "estimator_schedule_sha256": "c" * 64,
    }
    validate_confirmation_release_token(
        expected,
        config_sha256="a" * 64,
        run_matrix_sha256="b" * 64,
        estimator_schedule_sha256="c" * 64,
    )
    bad = dict(expected)
    bad["pilot_go_passed"] = False
    with pytest.raises(PermissionError):
        validate_confirmation_release_token(
            bad,
            config_sha256="a" * 64,
            run_matrix_sha256="b" * 64,
            estimator_schedule_sha256="c" * 64,
        )


def test_21_checkpoint_and_no_aux_classification_are_frozen():
    config = json.loads((PROJECT_ROOT / "configs" / "phase8" / "phase8_execution_lock.json").read_text())
    assert config["blocks"]["capacity_replication"]["gating_checkpoint"] == "restored_legacy_best"
    assert config["blocks"]["fixed_target_interventions"]["gating_checkpoint"] == "restored_legacy_best"
    assert config["blocks"]["coefficient_replication"]["gating_checkpoint"] == "restored_legacy_best"
    assert config["blocks"]["repair_pilot"]["gating_checkpoint"] == "final"
    assert config["blocks"]["repair_confirmation"]["gating_checkpoint"] == "final"
    assert config["blocks"]["fixed_target_interventions"]["no_aux_control_status"] == (
        "new_matched_input_space_control_not_legacy_replication"
    )


def test_22_no_aux_control_is_raw_target_and_non_graph_evidence():
    torch.manual_seed(180)
    cfg = Phase8ModelConfig(d=3, lag=2, layers=1, hidden=5, d_cond=2, d_state=2)
    control = make_no_aux_input_space_control(cfg)
    x = _x(seed=181, T=10, dtype=np.float32)
    raw = as_raw_bdt(x, device=torch.device("cpu"), dtype=torch.float32)
    idx = target_indices(10, 2)
    mse = control.fixed_target_prediction_mse(raw, raw, idx)
    assert torch.isfinite(mse)
    assert control.control_status == "new_matched_input_space_control_not_legacy_replication"
    assert control.graph_recovery_evidence_allowed is False


def test_23_checkpoint_helpers_separate_legacy_restore_from_fixed_final():
    cfg = Phase8ModelConfig(d=2, lag=1, layers=1, hidden=3)
    x = _x(seed=191, d=2, T=6, dtype=np.float32)
    torch.manual_seed(190)
    restored = make_legacy_baseline(cfg)
    restored_meta = train_with_frozen_checkpoint_policy(
        restored,
        x,
        max_iter=2,
        checkpoint_policy=LEGACY_RESTORE_POLICY,
    )
    torch.manual_seed(190)
    final = make_legacy_baseline(cfg)
    final_meta = train_with_frozen_checkpoint_policy(
        final,
        x,
        max_iter=2,
        checkpoint_policy=FIXED_FINAL_POLICY,
    )
    assert restored_meta.gating_checkpoint == "restored_legacy_best"
    assert restored_meta.selected_iteration == 0
    assert final_meta.gating_checkpoint == "final"
    assert final_meta.selected_iteration == 1
