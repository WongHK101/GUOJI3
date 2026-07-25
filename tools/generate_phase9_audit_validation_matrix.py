"""Generate the frozen Phase 9 audit-generality validation run matrix."""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = (
    ROOT
    / "paper-data"
    / "docs"
    / "phase9_audit_validation_v1"
    / "PHASE9_AUDIT_RUN_MATRIX.csv"
)

METHODS = ("baseline", "mamba_concat", "tcn_concat")

STAGES = {
    "A_4090_VALIDATION": {
        "environment": "windows_rtx4090",
        "authorized": False,
        "replicates": 3,
        "master_seed_base": 29001000,
        "predictor_seed_offset": 100000,
        "preprocessor_seed_offset": 200000,
        "perturbation_seed_base": 29100000,
        "score_window_seed_base": 29200000,
        "units": (
            ("netsim19", "netsim", "sim3_subject_19.npz", 0, 200),
            ("netsim08", "netsim", "sim3_subject_8.npz", 0, 200),
            ("netsim44", "netsim", "sim3_subject_44.npz", 0, 200),
            ("netsim03", "netsim", "sim3_subject_3.npz", 0, 200),
            (
                "mocap_run_holdout",
                "mocap",
                "mocap_time_series_run_angles.npz",
                728,
                1228,
            ),
            (
                "mocap_salsa_holdout",
                "mocap",
                "mocap_time_series_salsa_angles.npz",
                3000,
                3500,
            ),
        ),
    },
    "B_AUTODL_CONFIRMATION": {
        "environment": "autodl_frozen_confirmation",
        "authorized": False,
        "replicates": 2,
        "master_seed_base": 29301000,
        "predictor_seed_offset": 100000,
        "preprocessor_seed_offset": 200000,
        "perturbation_seed_base": 29400000,
        "score_window_seed_base": 29500000,
        "units": (
            ("netsim16", "netsim", "sim3_subject_16.npz", 0, 200),
            ("netsim00", "netsim", "sim3_subject_0.npz", 0, 200),
            ("netsim30", "netsim", "sim3_subject_30.npz", 0, 200),
            ("netsim10", "netsim", "sim3_subject_10.npz", 0, 200),
            (
                "mocap_run_holdout",
                "mocap",
                "mocap_time_series_run_angles.npz",
                728,
                1228,
            ),
            (
                "mocap_salsa_holdout",
                "mocap",
                "mocap_time_series_salsa_angles.npz",
                3000,
                3500,
            ),
        ),
    },
}


def rows():
    records = []
    for stage, spec in STAGES.items():
        for unit_index, unit in enumerate(spec["units"]):
            unit_id, kind, source_file, start, stop = unit
            for replicate in range(1, int(spec["replicates"]) + 1):
                master_seed = int(spec["master_seed_base"]) + unit_index * 100 + replicate
                for method in METHODS:
                    records.append(
                        {
                            "run_id": (
                                f"{stage.lower()}__{unit_id}__{method}"
                                f"__rep{replicate}"
                            ),
                            "stage": stage,
                            "execution_authorized": str(spec["authorized"]).lower(),
                            "environment": spec["environment"],
                            "data_unit": unit_id,
                            "dataset_kind": kind,
                            "source_file": source_file,
                            "segment_start_inclusive": start,
                            "segment_stop_exclusive": stop,
                            "method": method,
                            "replicate": replicate,
                            "master_seed": master_seed,
                            "predictor_seed": (
                                master_seed + int(spec["predictor_seed_offset"])
                            ),
                            "preprocessor_seed": (
                                ""
                                if method == "baseline"
                                else master_seed
                                + int(spec["preprocessor_seed_offset"])
                            ),
                            "perturbation_seed": (
                                int(spec["perturbation_seed_base"]) + unit_index
                            ),
                            "score_window_seed": (
                                int(spec["score_window_seed_base"]) + unit_index
                            ),
                            "max_iter": 1000,
                            "lag": 3,
                            "primary_audit_horizon": 64,
                            "audit_window_count": 32,
                            "checkpoint_policy": "restored_min_total_objective",
                            "evidence_role": (
                                "internal_prospective_go_no_go"
                                if stage == "A_4090_VALIDATION"
                                else "conditional_external_confirmation"
                            ),
                            "duplicate_of": "",
                        }
                    )
    for method in ("mamba_concat", "tcn_concat"):
        source = next(
            record
            for record in records
            if record["stage"] == "A_4090_VALIDATION"
            and record["data_unit"] == "netsim19"
            and record["method"] == method
            and record["replicate"] == 1
        )
        duplicate = dict(source)
        duplicate["run_id"] = source["run_id"] + "__determinism_duplicate"
        duplicate["evidence_role"] = "non_primary_determinism_duplicate"
        duplicate["duplicate_of"] = source["run_id"]
        records.append(duplicate)
    return records


def main() -> int:
    records = rows()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print(f"Wrote {len(records)} rows to {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
