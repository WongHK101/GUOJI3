from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src", ROOT / "experiments"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from phase9_lorenz_strong_operating_preflight import (  # noqa: E402
    evaluate_method_gates,
    load_config,
    make_model_config,
)
from phase9_audit_stageb import make_adapter, state_subset_sha256  # noqa: E402


CONFIG = ROOT / "configs" / "phase9_lorenz_strong_operating_preflight_v1.json"


def test_config_is_strictly_development_only():
    config = load_config(CONFIG)
    assert config["boundaries"]["development_only"] is True
    assert config["boundaries"]["formal_result"] is False
    assert config["boundaries"]["manuscript_evidence"] is False
    assert config["boundaries"]["formal_seed_generation_authorized"] is False
    assert config["boundaries"]["autodl_execution_authorized"] is False
    assert config["dataset"]["data_seed"] == 0
    assert config["dataset"]["development_seed_previously_observed"] is True


def test_model_config_locks_lorenz_nominal_lag():
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    cfg = make_model_config(config, d=10)
    assert cfg.d == 10
    assert cfg.lag == 1
    assert cfg.layers == 5
    assert cfg.hidden == 50
    assert cfg.d_cond == 4
    assert cfg.d_state == 8
    assert cfg.attribution_horizon == 64


def test_same_architecture_initialization_is_deterministic():
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    cfg = make_model_config(config, d=10)
    seeds = config["seeds"]
    hashes = []
    for _ in range(2):
        adapter = make_adapter(
            "mamba_concat",
            cfg,
            predictor_seed=int(seeds["predictor_seed"]),
            preprocessor_seed=int(seeds["preprocessor_seed"]),
        )
        hashes.append(
            state_subset_sha256(adapter.model, prefix_excluded="preprocessor")
        )
    assert hashes[0] == hashes[1]


def test_baseline_gate_requires_strong_graph_and_score_identity():
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    summary = {
        "total_nominal_metrics": {"auroc": 0.8},
        "missing_route_relative_magnitude": 0.0,
    }
    arrays = {
        "s_total_nominal": np.eye(3),
        "s_partial_nominal": np.eye(3),
    }
    report = evaluate_method_gates(
        method="baseline",
        primary_summary=summary,
        primary_arrays=arrays,
        interventions={},
        mixing=None,
        horizon=None,
        gates=config["gates"],
    )
    assert report["passed"]
    summary["total_nominal_metrics"]["auroc"] = 0.59
    rejected = evaluate_method_gates(
        method="baseline",
        primary_summary=summary,
        primary_arrays=arrays,
        interventions={},
        mixing=None,
        horizon=None,
        gates=config["gates"],
    )
    assert not rejected["passed"]
    assert not rejected["checks"]["strong_operating_point"]


def test_concat_gate_requires_route_coordinate_and_horizon_evidence():
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    summary = {
        "missing_route_relative_magnitude": 0.4,
        "partial_total_nominal_pearson": 0.7,
        "partial_total_nominal_topk_jaccard": 0.7,
        "temporal_tail_statistics": {"median": 0.2},
    }
    interventions = {
        "fixed_target_prediction_mse_delta": {"mask_c": 0.2},
    }
    mixing = {"normalized_source_entropy": {"median": 0.9}}
    horizon = {
        "offdiagonal_cumulative_mass_ratio_vs_H128": {"64": 0.995},
        "nominal_score_max_abs_difference_vs_H128": {"64": 0.0},
    }
    report = evaluate_method_gates(
        method="mamba_concat",
        primary_summary=summary,
        primary_arrays={},
        interventions=interventions,
        mixing=mixing,
        horizon=horizon,
        gates=config["gates"],
    )
    assert report["passed"]
    interventions["fixed_target_prediction_mse_delta"]["mask_c"] = 0.0
    rejected = evaluate_method_gates(
        method="mamba_concat",
        primary_summary=summary,
        primary_arrays={},
        interventions=interventions,
        mixing=mixing,
        horizon=horizon,
        gates=config["gates"],
    )
    assert not rejected["passed"]
    assert not rejected["checks"]["mask_c"]
