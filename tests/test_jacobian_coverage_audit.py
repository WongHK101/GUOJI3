from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from jacobian_coverage_audit import (  # noqa: E402
    COORDINATE_AMBIGUOUS,
    COVERED,
    HORIZON_TRUNCATED,
    PARTIALLY_COVERED,
    SCHEMA_VERSION,
    AuditReport,
    CoverageDeclaration,
    PredictiveRoute,
    build_audit_profile,
    cross_source_leakage,
    finite_support_upper_bound,
    route_chain_rule,
    topk_stability_guaranteed,
)


def concat_declaration() -> CoverageDeclaration:
    return CoverageDeclaration(
        architecture="concat_jrngc",
        graph_claim="directed raw-variable Granger-predictive graph",
        score_variables=("x",),
        penalty_variables=("x",),
        predictive_routes=(
            PredictiveRoute(
                route_id="raw_x",
                description="raw-history predictor route",
                enters_prediction=True,
                interpreted_as_graph_knowledge=True,
                score_covered=True,
                penalty_covered=True,
            ),
            PredictiveRoute(
                route_id="aux_c",
                description="causal auxiliary predictor route derived from raw history",
                enters_prediction=True,
                interpreted_as_graph_knowledge=True,
                score_covered=False,
                penalty_covered=False,
            ),
        ),
        coordinate_mapping="auxiliary coordinates mix original source variables",
        coordinate_identity_valid=False,
        primary_score_horizon=1,
        attribution_horizon=128,
        required_support_horizon=None,
        omitted_mass_beyond_horizon_assessed=False,
        score_penalty_coordinate_compatible=True,
        score_penalty_horizon_relation="identical partial nominal support",
    )


def test_profile_keeps_completeness_alignment_coordinate_and_horizon_separate():
    profile = build_audit_profile(concat_declaration())
    assert profile.score_route_completeness == PARTIALLY_COVERED
    assert profile.penalty_route_completeness == PARTIALLY_COVERED
    assert profile.score_penalty_alignment == COVERED
    assert profile.coordinate_validity == COORDINATE_AMBIGUOUS
    assert profile.horizon_validity == HORIZON_TRUNCATED


def test_explicit_penalty_exemption_requires_reason():
    route = PredictiveRoute(
        route_id="context",
        description="declared context route",
        enters_prediction=True,
        interpreted_as_graph_knowledge=False,
        score_covered=False,
        penalty_covered=False,
        penalty_exempt=True,
    )
    with pytest.raises(ValueError, match="requires a reason"):
        route.validate()


def test_route_chain_rule_matches_autograd_double_precision():
    x = torch.tensor([0.4, -0.7], dtype=torch.float64, requires_grad=True)

    def auxiliary(raw):
        return torch.stack((raw[0] ** 2 + raw[1], raw[0] - 0.5 * raw[1] ** 2))

    def predictor(raw, condition):
        return torch.stack((
            1.2 * raw[0] - 0.3 * raw[1] + condition[0] * condition[1],
            raw[1] + torch.sin(condition[0]),
        ))

    c = auxiliary(x)
    total = torch.autograd.functional.jacobian(lambda raw: predictor(raw, auxiliary(raw)), x)
    partial_raw = torch.autograd.functional.jacobian(
        lambda raw: predictor(raw, c.detach()),
        x,
    )
    partial_aux = torch.autograd.functional.jacobian(
        lambda condition: predictor(x.detach(), condition),
        c.detach(),
    )
    aux_raw = torch.autograd.functional.jacobian(auxiliary, x)
    reconstructed = route_chain_rule(
        partial_raw.detach().numpy(),
        partial_aux.detach().numpy(),
        aux_raw.detach().numpy(),
    )
    np.testing.assert_allclose(reconstructed, total.detach().numpy(), atol=1e-10, rtol=1e-8)


def test_coordinate_preserving_and_mixed_counterexample():
    depthwise = np.zeros((3, 3, 2), dtype=np.float64)
    depthwise[0, 0] = [1.0, 0.2]
    depthwise[1, 1] = [0.9, -0.1]
    depthwise[2, 2] = [1.1, 0.3]
    assert cross_source_leakage(depthwise) < 1e-12

    mixed = depthwise.copy()
    mixed[0, 2, 0] = 0.5
    assert cross_source_leakage(mixed) > 0.05


def test_finite_horizon_closure_and_long_memory_boundary():
    assert finite_support_upper_bound(nominal_lag=3, transform_support=4) == 6
    with pytest.raises(ValueError):
        finite_support_upper_bound(nominal_lag=0, transform_support=4)

    alpha = 0.9
    omitted_after_32 = alpha**32
    assert omitted_after_32 > 0.01


def test_topk_ranking_stability_fixture_and_boundary():
    reference = np.array([1.0, 0.8, 0.3, 0.1])
    stable = np.array([0.98, 0.82, 0.31, 0.09])
    unstable = np.array([1.0, 0.48, 0.62, 0.1])
    assert topk_stability_guaranteed(reference, stable, k=2)
    assert not topk_stability_guaranteed(reference, unstable, k=2)


def test_report_serialization_and_machine_readable_profile(tmp_path):
    declaration = concat_declaration()
    report = AuditReport(
        schema_version=SCHEMA_VERSION,
        declaration=declaration,
        profile=build_audit_profile(declaration),
        diagnostics={
            "missing_route_relative_magnitude": 0.71,
            "partial_total_nominal_pearson": 0.63,
            "temporal_tail_mass": {"median": 0.24, "maximum": 0.41},
        },
        provenance={
            "code_commit": "0397e8af27c4f396d7713b129e0d7307da732681",
            "config_sha256": "fixture",
        },
        score_object_files={
            "partial_nominal": "scores/partial_nominal.npy",
            "total_nominal": "scores/total_nominal.npy",
        },
    )
    json_path = report.write_json(tmp_path / "audit_report.json")
    csv_path = report.write_profile_csv(tmp_path / "audit_profile.csv")
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["audit_profile"]["score_route_completeness"] == PARTIALLY_COVERED
    assert csv_path.read_text(encoding="utf-8").count("\n") == 6
    assert b"\r\n" not in json_path.read_bytes()
    assert b"\r\n" not in csv_path.read_bytes()


def test_report_rejects_nonfinite_diagnostic(tmp_path):
    declaration = concat_declaration()
    report = AuditReport(
        schema_version=SCHEMA_VERSION,
        declaration=declaration,
        profile=build_audit_profile(declaration),
        diagnostics={"invalid": float("nan")},
        provenance={"code_commit": "fixture"},
        score_object_files={"total_nominal": "score.npy"},
    )
    with pytest.raises(ValueError, match="nonfinite"):
        report.write_json(tmp_path / "invalid.json")
