from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import pytest
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src", ROOT / "experiments"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from aggregate_phase9_audit_stageb import architecture_gate  # noqa: E402
from validate_phase9_stageb_smoke import (  # noqa: E402
    MATRIX_RUN_IDS,
    METHODS,
    compare_roots,
    inspect_root,
)
from phase9_audit_stageb import (  # noqa: E402
    release_lock_fingerprint,
    validate_matrix,
    validate_smoke_gate,
)


DOCS = ROOT / "paper-data" / "docs" / "phase9_audit_validation_v1"
CONFIG_PATH = ROOT / "configs" / "phase9_audit_stageb_v1.json"
SOURCE_MATRIX = DOCS / "PHASE9_AUDIT_RUN_MATRIX.csv"
AUTHORIZED_MATRIX = DOCS / "PHASE9_AUDIT_STAGEB_AUTHORIZED_MATRIX.csv"
STAGE_A_DECISION = DOCS / "STAGEA_GATE_DECISION_LOCK.json"


def read_csv(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def load_config():
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def test_authorized_matrix_only_flips_execution_permission():
    source = [
        row
        for row in read_csv(SOURCE_MATRIX)
        if row["stage"] == "B_AUTODL_CONFIRMATION"
    ]
    authorized = read_csv(AUTHORIZED_MATRIX)
    assert len(source) == len(authorized) == 36
    assert [row["run_id"] for row in source] == [
        row["run_id"] for row in authorized
    ]
    for sealed, released in zip(source, authorized):
        assert sealed["execution_authorized"] == "false"
        assert released["execution_authorized"] == "true"
        for field in sealed:
            if field != "execution_authorized":
                assert released[field] == sealed[field]


def test_stageb_matrix_cardinality_environment_and_seed_boundary():
    config = load_config()
    selected = validate_matrix(read_csv(AUTHORIZED_MATRIX), config)
    assert len(selected) == 36
    assert {row["method"] for row in selected} == {
        "baseline",
        "mamba_concat",
        "tcn_concat",
    }
    assert {row["data_unit"] for row in selected} == set(config["datasets"])
    assert {int(row["replicate"]) for row in selected} == {1, 2}
    assert all(
        row["environment"] == "autodl_frozen_confirmation"
        for row in selected
    )
    assert all(
        row["evidence_role"] == "conditional_external_confirmation"
        for row in selected
    )
    assert not {
        "netsim19",
        "netsim08",
        "netsim44",
        "netsim03",
    } & {row["data_unit"] for row in selected}


def test_stageb_matrix_guard_rejects_permission_or_environment_change():
    config = load_config()
    rows = read_csv(AUTHORIZED_MATRIX)
    rows[0]["execution_authorized"] = "false"
    with pytest.raises(RuntimeError, match="Unauthorized Stage B row"):
        validate_matrix(rows, config)
    rows = read_csv(AUTHORIZED_MATRIX)
    rows[0]["environment"] = "windows_rtx4090"
    with pytest.raises(RuntimeError, match="Unexpected Stage B environment"):
        validate_matrix(rows, config)


def test_stagea_decision_lock_is_exact_and_passing():
    expected = (
        "68cff029f6d192260abce0567de5f46b"
        "b9de73bf5355b61e64dfa923f8406166"
    )
    canonical = STAGE_A_DECISION.read_bytes().replace(b"\r\n", b"\n")
    assert hashlib.sha256(canonical).hexdigest() == expected
    decision = json.loads(STAGE_A_DECISION.read_text(encoding="utf-8"))
    assert decision["decision"] == "UNLOCK_STAGE_B"
    assert decision["unlock_stage_b"] is True
    assert decision["semantic_integrity_passed"] is True
    assert decision["determinism"]["passed"] is True
    assert (
        load_config()["required_stage_a_decision_canonical_sha256"]
        == expected
    )


def test_stageb_scientific_configuration_remains_stagea_matched():
    stage_a = json.loads(
        (ROOT / "configs" / "phase9_audit_stagea_v1.json").read_text(
            encoding="utf-8"
        )
    )
    stage_b = load_config()
    assert stage_b["model"] == stage_a["model"]
    assert stage_b["training"] == stage_a["training"]
    assert stage_b["evaluation"] == stage_a["evaluation"]
    for field in (
        "finite_required",
        "baseline_partial_total_max_abs",
        "baseline_missing_route_max",
        "determinism_max_abs",
        "determinism_metric_abs",
        "qualifying_unit_count",
        "unit_count",
        "missing_route_min",
        "mask_c_delta_median_min",
        "partial_total_pearson_max",
        "netsim_discrepancy_subject_count",
        "netsim_subject_count",
        "topk_jaccard_max",
        "coordinate_entropy_median_min",
        "temporal_tail_median_min",
        "nominal_horizon_max_abs",
        "h64_h128_ratio_qualifying_min",
        "h64_h128_ratio_absolute_min",
        "baseline_netsim_auroc_context_min",
    ):
        assert stage_b["gates"][field] == stage_a["gates"][field]
    assert stage_b["gates"]["qualifying_seed_count"] == 2
    assert stage_b["gates"]["seed_count"] == 2


def test_stageb_gate_requires_both_replicates_in_qualifying_units():
    gates = load_config()["gates"]
    units = []
    for index in range(6):
        units.append(
            {
                "data_unit": f"unit{index}",
                "dataset_kind": "netsim" if index < 4 else "mocap",
                "missing_route_mean": 0.4,
                "missing_route_seed_pass_count": 2,
                "mask_c_delta_mean": 0.2,
                "mask_c_positive_seed_count": 2,
                "partial_total_pearson_mean": 0.7,
                "partial_total_pearson_defined_count": 2,
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
    assert architecture_gate(units, gates=gates)["passed"] is True
    for index in (0, 1):
        units[index]["missing_route_seed_pass_count"] = 1
    failed = architecture_gate(units, gates=gates)
    assert failed["passed"] is False
    assert failed["checks"]["missing_route"] is False


def test_formal_smoke_gate_is_bound_to_the_exact_release(tmp_path):
    release_lock = {
        "approved_commit": "a" * 40,
        "release_token_sha256": "b" * 64,
        "stage_a_decision_canonical_sha256": "c" * 64,
    }
    payload = {
        "passed": True,
        "formal_scientific_evidence": False,
        "release_lock_fingerprint": release_lock_fingerprint(release_lock),
        "approved_commit": release_lock["approved_commit"],
        "release_token_sha256": release_lock["release_token_sha256"],
        "stage_a_decision_canonical_sha256": release_lock[
            "stage_a_decision_canonical_sha256"
        ],
    }
    path = tmp_path / "smoke.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert validate_smoke_gate(
        path,
        release_lock=release_lock,
    )["release_lock_fingerprint"] == release_lock_fingerprint(release_lock)
    payload["approved_commit"] = "d" * 40
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="commit mismatch"):
        validate_smoke_gate(path, release_lock=release_lock)


def test_smoke_validator_accepts_identical_roots_and_rejects_drift(
    tmp_path,
):
    release_lock = {
        "approved_commit": "a" * 40,
        "release_token_sha256": "b" * 64,
        "stage_a_decision_canonical_sha256": "c" * 64,
    }

    def make_root(name: str) -> Path:
        root = tmp_path / name
        statuses = []
        for method in METHODS:
            matrix_id = MATRIX_RUN_IDS[method]
            run_id = f"smoke_it20__{matrix_id}"
            status = {
                "status": "complete",
                "run_id": run_id,
                "matrix_run_id": matrix_id,
                "method": method,
                "device": "cuda",
                "deterministic_algorithms": True,
                "no_nan_inf": True,
                "effective_iterations": 20,
                "formal_result": False,
                "confirmation_candidate": False,
                "manuscript_evidence": False,
                "cuda_max_memory_allocated_mb": 1.0,
            }
            statuses.append(status)
            run_dir = root / "runs" / run_id
            run_dir.mkdir(parents=True)
            torch.save(
                {"model_state": {"weight": torch.tensor([1.0])}},
                run_dir / "checkpoint.pt",
            )
            score = np.array([[0.0, 1.0], [0.5, 0.0]])
            np.savez_compressed(
                run_dir / "sampled_attribution_objects.npz",
                s_total_nominal=score,
            )
            (run_dir / "sampled_attribution_audit.json").write_text(
                json.dumps(
                    {"total_nominal_metrics": {"n_true_edges": 1}}
                ),
                encoding="utf-8",
            )
        (root / "execution_summary.json").write_text(
            json.dumps(
                {
                    "requested_records": 3,
                    "completed_records": 3,
                    "smoke": True,
                    "formal_result": False,
                    "confirmation_candidate": False,
                    "statuses": statuses,
                }
            ),
            encoding="utf-8",
        )
        (root / "release_lock.json").write_text(
            json.dumps(release_lock),
            encoding="utf-8",
        )
        return root

    first = make_root("first")
    second = make_root("second")
    assert inspect_root(first)["passed"] is True
    assert inspect_root(second)["passed"] is True
    assert compare_roots(first, second)["passed"] is True
    drift = (
        second
        / "runs"
        / f"smoke_it20__{MATRIX_RUN_IDS['mamba_concat']}"
        / "checkpoint.pt"
    )
    torch.save(
        {"model_state": {"weight": torch.tensor([1.01])}},
        drift,
    )
    assert compare_roots(first, second)["passed"] is False
