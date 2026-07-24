from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from phase8_coverage import Phase8ModelConfig, make_legacy_baseline, make_legacy_concat  # noqa: E402
from phase9_audit_generalization import (  # noqa: E402
    AUDIT_LABELS,
    build_audit_profile,
    condition_coordinate_mixing_audit,
    deterministic_audit_targets,
    make_tcn_concat_adapter,
    sampled_raw_chain_audit,
)


def fixture_config() -> Phase8ModelConfig:
    return Phase8ModelConfig(
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


def test_deterministic_targets_respect_horizon():
    first = deterministic_audit_targets(
        T=30,
        lag=2,
        attribution_horizon=8,
        count=5,
        seed=91,
    )
    second = deterministic_audit_targets(
        T=30,
        lag=2,
        attribution_horizon=8,
        count=5,
        seed=91,
    )
    np.testing.assert_array_equal(first, second)
    assert np.min(first) >= 8
    assert len(first) == 5


def test_baseline_partial_and_total_sampled_scores_coincide():
    torch.manual_seed(92)
    adapter = make_legacy_baseline(fixture_config())
    x = np.random.default_rng(93).normal(size=(3, 14)).astype(np.float32)
    audit = sampled_raw_chain_audit(
        adapter,
        x,
        target_indices=[8, 10, 12],
        attribution_horizon=6,
    )
    np.testing.assert_allclose(
        audit.s_total_nominal,
        audit.s_partial_nominal,
        atol=1e-7,
        rtol=1e-6,
    )
    assert audit.missing_route_relative_magnitude is not None
    assert audit.missing_route_relative_magnitude < 1e-7
    profile = build_audit_profile(
        architecture="baseline_jrngc",
        sampled_audit=audit,
        has_auxiliary_route=False,
    )
    assert profile["audit_dimensions"]["partial_score_route_completeness"] == (
        AUDIT_LABELS["covered"]
    )


def test_concat_reports_missing_route_and_coordinate_ambiguity():
    torch.manual_seed(94)
    adapter = make_legacy_concat(fixture_config())
    x = np.random.default_rng(95).normal(size=(3, 14)).astype(np.float32)
    targets = [8, 10, 12]
    audit = sampled_raw_chain_audit(
        adapter,
        x,
        target_indices=targets,
        attribution_horizon=6,
    )
    assert audit.missing_route_relative_magnitude is not None
    assert audit.missing_route_relative_magnitude > 0
    mixing = condition_coordinate_mixing_audit(
        adapter,
        x,
        target_indices=targets,
        attribution_horizon=6,
    )
    assert mixing["architecture_label"] == AUDIT_LABELS["coordinate_ambiguous"]
    assert mixing["defined_coordinate_time_count"] > 0
    profile = build_audit_profile(
        architecture="legacy_concat_jrngc",
        sampled_audit=audit,
        has_auxiliary_route=True,
    )
    assert profile["audit_dimensions"]["partial_score_route_completeness"] == (
        AUDIT_LABELS["partial"]
    )
    assert profile["audit_dimensions"]["bounded_total_raw_chain_horizon"] == (
        AUDIT_LABELS["horizon_truncated"]
    )


def test_tcn_auxiliary_route_is_causal_and_auditable():
    torch.manual_seed(96)
    adapter = make_tcn_concat_adapter(
        d=3,
        lag=2,
        layers=1,
        hidden=6,
        d_cond=2,
    )
    x = np.random.default_rng(97).normal(size=(3, 14)).astype(np.float32)
    raw = torch.as_tensor(x).unsqueeze(0).requires_grad_(True)
    condition = adapter.condition_sequence(raw)
    before = condition[:, :8].detach().clone()
    changed = raw.detach().clone()
    changed[:, :, 8:] += 10.0
    after = adapter.condition_sequence(changed)[:, :8].detach()
    torch.testing.assert_close(before, after)
    loss = condition.square().mean()
    output_projection_gradient = torch.autograd.grad(
        loss,
        adapter.model.preprocessor.tcn.out_proj.weight,
        retain_graph=True,
    )[0]
    assert torch.linalg.norm(output_projection_gradient).item() > 0
    audit = sampled_raw_chain_audit(
        adapter,
        x,
        target_indices=[8, 10, 12],
        attribution_horizon=6,
    )
    assert audit.missing_route_relative_magnitude is not None
    assert audit.missing_route_relative_magnitude > 0
