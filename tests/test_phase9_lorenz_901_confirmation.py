from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src", ROOT / "experiments"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from aggregate_phase9_lorenz_901_confirmation import evaluate_aggregate_gates
from lorenz96_frozen import generate_lorenz96, lorenz96_direct_graph
from phase9_lorenz_901_confirmation import (
    generate_dataset,
    load_dataset,
    load_protocol,
    load_run_matrix,
    method_seed_bundle,
    run_identifier,
)


CONFIG_PATH = ROOT / "configs" / "phase9_lorenz_901_confirmation_v1.json"


def test_frozen_protocol_and_run_matrix_are_exact():
    config = load_protocol(CONFIG_PATH)
    matrix = load_run_matrix(config)
    assert len(matrix) == 20
    assert {row["method"] for row in matrix} == {"baseline", "mamba_concat"}
    assert {row["iterations"] for row in matrix} == {2000}
    assert len({row["data_seed"] for row in matrix}) == 5
    assert len({row["train_seed"] for row in matrix}) == 2
    assert all(row["formal_result"] for row in matrix)
    assert all(row["manuscript_evidence_pending"] for row in matrix)


def test_formal_seed_namespace_excludes_phase7_stage1b_identifiers():
    config = load_protocol(CONFIG_PATH)
    formal_seeds = set(config["formal_design"]["data_seeds"])
    assert formal_seeds.isdisjoint({4, 5, 6, 7, 8})
    assert 0 not in formal_seeds
    assert config["development_smoke"]["data_seed"] == 0


def test_seed_bundle_is_shared_across_methods_and_frozen_by_data_seed():
    config = load_protocol(CONFIG_PATH)
    bundle = method_seed_bundle(
        config=config,
        data_seed=26073111,
        train_seed=26073201,
    )
    assert bundle == {
        "master_seed": 26073201,
        "predictor_seed": 26073201,
        "preprocessor_seed": 27073201,
        "score_window_seed": 27073111,
        "perturbation_seed": 28073111,
    }


def test_run_identifier_encodes_all_pairing_keys():
    assert run_identifier(
        prefix="formal",
        run_index=7,
        data_seed=26073123,
        train_seed=26073219,
        method="mamba_concat",
    ) == (
        "formal__r007__lorenz_f40"
        "__dseed26073123__tseed26073219__mamba_concat"
    )


def test_lorenz_generator_is_deterministic_and_normalized():
    kwargs = dict(
        d=6,
        t=80,
        t_eval=0,
        forcing=40.0,
        seed=12345,
        delta_t=0.1,
        observation_noise_sd=0.1,
        burn_in=60,
    )
    left, left_eval, left_graph = generate_lorenz96(**kwargs)
    right, right_eval, right_graph = generate_lorenz96(**kwargs)
    assert np.array_equal(left, right)
    assert np.array_equal(left_eval, right_eval)
    assert np.array_equal(left_graph, right_graph)
    assert left.shape == (6, 80)
    assert left.dtype == np.float32
    assert np.max(np.abs(np.mean(left, axis=1))) < 1e-5
    assert np.max(np.abs(np.std(left, axis=1) - 1.0)) < 1e-5


def test_lorenz_graph_orientation_and_support():
    graph = lorenz96_direct_graph(10)
    assert graph.shape == (10, 10, 1)
    assert int(np.sum(graph)) == 40
    for target in range(10):
        sources = set(np.flatnonzero(graph[target, :, 0]))
        assert sources == {
            target,
            (target + 1) % 10,
            (target - 1) % 10,
            (target - 2) % 10,
        }


def test_dataset_generation_is_hash_locked_and_idempotent(tmp_path):
    config = copy.deepcopy(load_protocol(CONFIG_PATH))
    config["dataset"].update({
        "d": 6,
        "T": 80,
        "burn_in": 60,
    })
    first = generate_dataset(
        output_root=tmp_path,
        config=config,
        data_seed=12345,
        formal_result=True,
    )
    second = generate_dataset(
        output_root=tmp_path,
        config=config,
        data_seed=12345,
        formal_result=True,
    )
    assert first["x_sha256"] == second["x_sha256"]
    assert first["graph_sha256"] == second["graph_sha256"]
    assert first["formal_result"] is True
    assert first["manuscript_evidence"] is False
    x, graph, loaded = load_dataset(output_root=tmp_path, data_seed=12345)
    assert x.shape == (6, 80)
    assert graph.shape == (6, 6, 1)
    assert loaded["x_sha256"] == first["x_sha256"]


def _passing_seed_rows():
    rows = []
    for index in range(5):
        rows.append({
            "data_seed": 100 + index,
            "baseline_total_nominal_auroc": 0.90,
            "mamba_concat_missing_route_relative_magnitude": 0.40,
            "mamba_concat_mask_c_fixed_target_mse_delta": 0.20,
            "mamba_concat_partial_total_nominal_pearson": 0.85,
            "mamba_concat_partial_total_nominal_topk_jaccard": 0.70,
            "mamba_concat_coordinate_entropy_median": 0.90,
            "mamba_concat_temporal_tail_median": 0.15,
            "mamba_concat_h64_h128_mass_ratio": 0.99,
            "mamba_concat_nominal_h64_h128_max_abs": 0.0,
            "concat_vs_baseline_relative_mse": 0.02,
            "concat_discrepancy_pass": True,
        })
    return rows


def test_aggregate_gate_passes_only_with_all_frozen_dimensions():
    config = load_protocol(CONFIG_PATH)
    report = evaluate_aggregate_gates(
        data_seed_rows=_passing_seed_rows(),
        run_validation_passed=True,
        gates=config["aggregate_gates"],
    )
    assert report["passed"] is True
    assert all(report["checks"].values())


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("baseline_total_nominal_auroc", 0.1),
        ("mamba_concat_missing_route_relative_magnitude", 0.0),
        ("mamba_concat_mask_c_fixed_target_mse_delta", 0.0),
        ("mamba_concat_coordinate_entropy_median", 0.0),
        ("mamba_concat_temporal_tail_median", 0.0),
        ("mamba_concat_h64_h128_mass_ratio", 0.1),
        ("mamba_concat_nominal_h64_h128_max_abs", 1.0),
        ("concat_vs_baseline_relative_mse", 1.0),
    ],
)
def test_aggregate_gate_rejects_systematic_failure(field, value):
    config = load_protocol(CONFIG_PATH)
    rows = _passing_seed_rows()
    for row in rows:
        row[field] = value
        if field.startswith("mamba_concat_partial_total"):
            row["concat_discrepancy_pass"] = False
    report = evaluate_aggregate_gates(
        data_seed_rows=rows,
        run_validation_passed=True,
        gates=config["aggregate_gates"],
    )
    assert report["passed"] is False


def test_protocol_json_contains_no_manuscript_authorization():
    raw = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    assert raw["boundaries"]["manuscript_edit_authorized"] is False
    assert raw["boundaries"]["phase7_stage1b_authorized"] is False
    assert raw["claim_boundary"]["not_permitted"]


def test_formal_status_contract_contains_determinism_comparison_fields():
    source = (
        ROOT
        / "experiments"
        / "phase9_lorenz_901_confirmation.py"
    ).read_text(encoding="utf-8")
    for field in (
        '"config_file_sha256"',
        '"config_canonical_sha256"',
        '"git_commit"',
    ):
        assert field in source
