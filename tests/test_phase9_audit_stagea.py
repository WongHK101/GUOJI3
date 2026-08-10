from __future__ import annotations

import copy
import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src", ROOT / "experiments"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from aggregate_phase9_audit_stagea import architecture_gate  # noqa: E402
from mamba_jrngc_pilot import train_model  # noqa: E402
from phase8_coverage import Phase8ModelConfig, make_legacy_baseline  # noqa: E402
from phase9_audit_generalization import (  # noqa: E402
    AUDIT_LABELS,
    build_audit_profile,
    make_tcn_concat_adapter,
    sampled_raw_chain_audit,
)
from phase9_audit_stagea import (  # noqa: E402
    canonical_text_sha256,
    horizon_summary,
    make_adapter,
    state_subset_sha256,
    train_legacy_with_metadata,
    validate_matrix,
)


CONFIG_PATH = ROOT / "configs" / "phase9_audit_stagea_v1.json"
MATRIX_PATH = (
    ROOT
    / "paper-data"
    / "docs"
    / "phase9_audit_validation_v1"
    / "PHASE9_AUDIT_RUN_MATRIX.csv"
)


def load_fixture():
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    with MATRIX_PATH.open("r", encoding="utf-8-sig", newline="") as handle:
        matrix = list(csv.DictReader(handle))
    return config, matrix


def test_stagea_matrix_cardinality_and_boundaries():
    config, matrix = load_fixture()
    selected = validate_matrix(matrix, config)
    assert len(selected) == 56
    assert sum(
        row["evidence_role"] == "internal_prospective_go_no_go"
        for row in selected
    ) == 54
    assert sum(
        row["evidence_role"] == "non_primary_determinism_duplicate"
        for row in selected
    ) == 2
    assert all(row["execution_authorized"] == "false" for row in matrix)
    assert config["boundaries"]["stage_b_authorized"] is False
    assert config["boundaries"]["autodl_authorized"] is False


def test_canonical_text_hash_ignores_crlf(tmp_path):
    left = tmp_path / "left.csv"
    right = tmp_path / "right.csv"
    left.write_bytes(b"a,b\n1,2\n")
    right.write_bytes(b"a,b\r\n1,2\r\n")
    assert canonical_text_sha256(left) == canonical_text_sha256(right)


def test_mamba_and_tcn_predictor_initialization_is_paired():
    cfg = Phase8ModelConfig(
        d=3,
        lag=2,
        layers=1,
        hidden=6,
        d_cond=2,
        d_state=2,
        d_conv=2,
        expand=1,
        jacobian_lam=0.01,
        dtype="float32",
    )
    mamba = make_adapter(
        "mamba_concat",
        cfg,
        predictor_seed=4401,
        preprocessor_seed=4402,
    )
    tcn = make_adapter(
        "tcn_concat",
        cfg,
        predictor_seed=4401,
        preprocessor_seed=4403,
    )
    assert state_subset_sha256(
        mamba.model,
        prefix_excluded="preprocessor",
    ) == state_subset_sha256(
        tcn.model,
        prefix_excluded="preprocessor",
    )
    for name, value in mamba.model.state_dict().items():
        if name.startswith("preprocessor."):
            continue
        torch.testing.assert_close(value, tcn.model.state_dict()[name])


def test_tcn_profile_declares_auxiliary_route():
    torch.manual_seed(4410)
    adapter = make_tcn_concat_adapter(
        d=3,
        lag=2,
        layers=1,
        hidden=6,
        d_cond=2,
    )
    x = np.random.default_rng(4411).normal(size=(3, 14)).astype(np.float32)
    audit = sampled_raw_chain_audit(
        adapter,
        x,
        target_indices=[8, 10, 12],
        attribution_horizon=6,
    )
    profile = build_audit_profile(
        architecture="causal_tcn_concat_jrngc",
        sampled_audit=audit,
        has_auxiliary_route=True,
    )
    assert profile["audit_dimensions"][
        "partial_score_route_completeness"
    ] == AUDIT_LABELS["partial"]


def test_local_training_loop_matches_legacy_checkpoint_behavior():
    torch.manual_seed(4420)
    cfg = Phase8ModelConfig(
        d=2,
        lag=2,
        layers=1,
        hidden=4,
        jacobian_lam=0.01,
        dtype="float32",
    )
    first = make_legacy_baseline(cfg).model
    second = copy.deepcopy(first)
    x = np.random.default_rng(4421).normal(size=(2, 10)).astype(np.float32)
    legacy, legacy_loss = train_model(
        first,
        x,
        max_iter=3,
        lr=1e-3,
        weight_decay=0.0,
        lookback=10,
        check_every=50,
        verbose=False,
    )
    metadata = train_legacy_with_metadata(
        second,
        x,
        max_iter=3,
        learning_rate=1e-3,
        weight_decay=0.0,
        gradient_clip_norm=1.0,
        check_every=50,
        lookback=10,
    )
    assert abs(float(legacy_loss) - metadata["best_total_regularized_objective"]) < 1e-8
    assert metadata["selected_iteration"] == 0
    for name, value in legacy.state_dict().items():
        torch.testing.assert_close(value, second.state_dict()[name])


def test_horizon_summary_uses_same_targets_and_offdiagonal_mass():
    targets = np.array([128, 140], dtype=np.int64)
    base = np.zeros((2, 2, 128), dtype=np.float64)
    base[0, 1, :32] = 1.0
    base[0, 1, 32:64] = 0.5
    base[0, 1, 64:] = 0.25

    def audit(horizon):
        jac = base[:, :, :horizon].copy()
        score = np.max(jac[:, :, :3], axis=2)
        return SimpleNamespace(
            j_bar_total=jac,
            s_total_nominal=score,
            target_indices=targets,
        )

    summary = horizon_summary(
        {32: audit(32), 64: audit(64), 128: audit(128)}
    )
    assert summary["same_target_indices"] is True
    assert summary["nominal_score_max_abs_difference_vs_H128"]["32"] == 0
    assert 0 < summary["offdiagonal_cumulative_mass_ratio_vs_H128"]["64"] < 1
    assert summary["mass_beyond_H128_assessed"] is False


def test_architecture_gate_passes_only_complete_synthetic_fixture():
    _, config_matrix = load_fixture()
    del config_matrix
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    units = []
    for index in range(6):
        units.append(
            {
                "data_unit": f"unit{index}",
                "dataset_kind": "netsim" if index < 4 else "mocap",
                "missing_route_mean": 0.4,
                "missing_route_seed_pass_count": 3,
                "mask_c_delta_mean": 0.2,
                "mask_c_positive_seed_count": 3,
                "partial_total_pearson_mean": 0.7,
                "partial_total_pearson_defined_count": 3,
                "partial_total_jaccard_mean": (
                    0.5 if index < 4 else None
                ),
                "coordinate_entropy_median_mean": 0.9,
                "temporal_tail_median_mean": 0.2,
                "h64_h128_mass_ratio_mean": 0.995,
                "horizon_nominal_max_abs_mean": 0.0,
                "horizon_nominal_max_abs_max": 0.0,
            }
        )
    passed = architecture_gate(units, gates=config["gates"])
    assert passed["passed"] is True
    units[0]["h64_h128_mass_ratio_mean"] = 0.90
    failed = architecture_gate(units, gates=config["gates"])
    assert failed["passed"] is False
    assert failed["checks"]["horizon_absolute_min"] is False
