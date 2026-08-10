"""Generate a non-evidentiary example Jacobian coverage report."""

from __future__ import annotations

import hashlib
from pathlib import Path
import sys

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPOSITORY_ROOT))

from src.jacobian_coverage_audit import (
    AuditReport,
    CoverageDeclaration,
    PredictiveRoute,
    SCHEMA_VERSION,
    build_audit_profile,
)


OUTPUT_DIR = Path(__file__).resolve().parent


def main() -> None:
    partial_score = np.array([[0.0, 0.20], [0.10, 0.0]], dtype=np.float64)
    total_score = np.array([[0.0, 0.50], [0.10, 0.0]], dtype=np.float64)
    partial_path = OUTPUT_DIR / "partial_nominal_example.npy"
    total_path = OUTPUT_DIR / "total_nominal_example.npy"
    np.save(partial_path, partial_score)
    np.save(total_path, total_score)

    declaration = CoverageDeclaration(
        architecture="Illustrative concat auxiliary predictor",
        graph_claim="Nominal raw-variable directed Granger-predictive graph",
        score_variables=("raw_history",),
        penalty_variables=("raw_history",),
        predictive_routes=(
            PredictiveRoute(
                route_id="raw_route",
                description="Raw history enters the predictor directly.",
                enters_prediction=True,
                interpreted_as_graph_knowledge=True,
                score_covered=True,
                penalty_covered=True,
            ),
            PredictiveRoute(
                route_id="auxiliary_route",
                description="A causal transform of raw history enters by concatenation.",
                enters_prediction=True,
                interpreted_as_graph_knowledge=True,
                score_covered=False,
                penalty_covered=False,
            ),
        ),
        coordinate_mapping="Raw score columns retain original source identities.",
        coordinate_identity_valid=True,
        primary_score_horizon=1,
        attribution_horizon=64,
        required_support_horizon=None,
        omitted_mass_beyond_horizon_assessed=False,
        score_penalty_coordinate_compatible=True,
        score_penalty_horizon_relation=(
            "The partial score and penalty use the same raw nominal support."
        ),
    )
    report = AuditReport(
        schema_version=SCHEMA_VERSION,
        declaration=declaration,
        profile=build_audit_profile(declaration),
        diagnostics={
            "documentation_fixture": True,
            "scientific_evidence": False,
            "partial_total_max_abs_difference": float(
                np.max(np.abs(partial_score - total_score))
            ),
        },
        provenance={
            "generator": "generate_example.py",
            "scope": "non-evidentiary documentation fixture",
        },
        score_object_files={
            "partial_nominal": partial_path.name,
            "total_nominal": total_path.name,
        },
    )
    json_path = report.write_json(OUTPUT_DIR / "concat_x_only_audit.json")
    csv_path = report.write_profile_csv(OUTPUT_DIR / "concat_x_only_profile.csv")

    generated = [
        Path(__file__),
        OUTPUT_DIR / "README.md",
        json_path,
        csv_path,
        partial_path,
        total_path,
    ]
    lines = []
    for path in sorted(generated, key=lambda item: item.name):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(f"{digest}  {path.name}")
    (OUTPUT_DIR / "ARTIFACT_SHA256SUMS.txt").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
        newline="\n",
    )


if __name__ == "__main__":
    main()
