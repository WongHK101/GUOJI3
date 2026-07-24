"""Development utilities for cross-domain Jacobian coverage audits.

The routines intentionally separate direct nominal-lag graph attribution from
bounded historical route diagnostics. A sampled H-lag audit is always labelled
HORIZON-TRUNCATED; it is not a replacement for full-prefix attribution.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from phase8_coverage import (
    EPS,
    LegacyBaselineAdapter,
    LegacyComparatorAdapter,
    LegacyConcatXOnlyAdapter,
    as_raw_bdt,
    exact_topk_jaccard,
    partial_raw_chain_at_target,
    pearson_with_reason,
)
from repaired_istf import canonical_metric_adapter


AUDIT_LABELS = {
    "covered": "COVERED",
    "partial": "PARTIALLY COVERED",
    "coordinate_ambiguous": "COORDINATE-AMBIGUOUS",
    "horizon_truncated": "HORIZON-TRUNCATED",
    "unassessed": "UNASSESSED",
}


@dataclass(frozen=True)
class SampledAuditResult:
    j_bar_total: np.ndarray
    j_bar_partial: np.ndarray
    j_bar_missing: np.ndarray
    s_total_nominal: np.ndarray
    s_partial_nominal: np.ndarray
    s_total_bounded_history: np.ndarray
    target_indices: np.ndarray
    attribution_horizon: int
    nominal_lag: int
    missing_route_relative_magnitude: Optional[float]
    missing_route_undefined_reason: Optional[str]
    partial_total_nominal_pearson: Optional[float]
    partial_total_nominal_pearson_reason: Optional[str]
    partial_total_nominal_topk_jaccard: Optional[float]
    temporal_tail_statistics: Mapping[str, object]
    total_nominal_metrics: Optional[Mapping[str, float]]
    partial_nominal_metrics: Optional[Mapping[str, float]]

    def summary(self) -> Dict[str, object]:
        return {
            "target_indices": self.target_indices.tolist(),
            "attribution_horizon": int(self.attribution_horizon),
            "nominal_lag": int(self.nominal_lag),
            "attribution_scope": "deterministic_sampled_bounded_raw_chain",
            "horizon_label": AUDIT_LABELS["horizon_truncated"],
            "missing_route_relative_magnitude": self.missing_route_relative_magnitude,
            "missing_route_undefined_reason": self.missing_route_undefined_reason,
            "partial_total_nominal_pearson": self.partial_total_nominal_pearson,
            "partial_total_nominal_pearson_reason": self.partial_total_nominal_pearson_reason,
            "partial_total_nominal_topk_jaccard": self.partial_total_nominal_topk_jaccard,
            "temporal_tail_statistics": dict(self.temporal_tail_statistics),
            "total_nominal_metrics": (
                None if self.total_nominal_metrics is None else dict(self.total_nominal_metrics)
            ),
            "partial_nominal_metrics": (
                None if self.partial_nominal_metrics is None else dict(self.partial_nominal_metrics)
            ),
        }

    def arrays(self) -> Dict[str, np.ndarray]:
        return {
            "j_bar_total": self.j_bar_total,
            "j_bar_partial": self.j_bar_partial,
            "j_bar_missing": self.j_bar_missing,
            "s_total_nominal": self.s_total_nominal,
            "s_partial_nominal": self.s_partial_nominal,
            "s_total_bounded_history": self.s_total_bounded_history,
            "target_indices": self.target_indices,
        }


def deterministic_audit_targets(
    *,
    T: int,
    lag: int,
    attribution_horizon: int,
    count: int,
    seed: int,
) -> np.ndarray:
    if attribution_horizon < lag:
        raise ValueError("attribution_horizon must be >= nominal lag")
    eligible = np.arange(max(lag, attribution_horizon), T, dtype=np.int64)
    if eligible.size == 0:
        raise ValueError("No targets are eligible for the requested audit horizon")
    if count <= 0 or count >= eligible.size:
        return eligible
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(eligible, size=count, replace=False))


def _offdiagonal_mass(values: np.ndarray) -> float:
    d = values.shape[0]
    mask = ~np.eye(d, dtype=bool)
    return float(np.sum(values[mask, :]))


def _distribution_summary(values: Sequence[float]) -> Dict[str, object]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {
            "defined_count": 0,
            "mean": None,
            "median": None,
            "p95": None,
            "maximum": None,
        }
    return {
        "defined_count": int(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p95": float(np.quantile(arr, 0.95)),
        "maximum": float(np.max(arr)),
    }


def sampled_raw_chain_audit(
    model: LegacyComparatorAdapter,
    x_full,
    *,
    target_indices: Sequence[int],
    attribution_horizon: int,
    graph: Optional[np.ndarray] = None,
) -> SampledAuditResult:
    """Compute partial and total raw-chain audit objects on fixed windows.

    Absolute Jacobians are formed per window and then averaged. The source-lag
    axis stores h=1 at position zero. The diagonal remains in the raw tensors
    and is excluded only in graph metrics and off-diagonal diagnostics.
    """

    indices = np.asarray(list(target_indices), dtype=np.int64)
    if indices.size == 0 or int(np.min(indices)) < attribution_horizon:
        raise ValueError("Every target must support the requested attribution horizon")
    d = int(model.d)
    lag = int(model.lag)
    device = model.device
    raw = as_raw_bdt(x_full, device=device, dtype=torch.float32, require_grad=True)
    total_windows: List[np.ndarray] = []
    partial_windows: List[np.ndarray] = []
    missing_windows: List[np.ndarray] = []
    tail_ratios: List[float] = []
    tail_undefined = 0
    offdiag = ~np.eye(d, dtype=bool)

    for target_u in indices.tolist():
        prediction = model.predict_from_raw(raw, [target_u])[0]
        total_rows = []
        for output in range(d):
            gradient = torch.autograd.grad(
                prediction[output],
                raw,
                create_graph=False,
                retain_graph=True,
                allow_unused=False,
            )[0]
            total_rows.append(torch.stack([
                gradient[0, :, target_u - h]
                for h in range(1, attribution_horizon + 1)
            ], dim=1))
        total = torch.stack(total_rows, dim=0).detach().cpu().numpy().astype(np.float64)
        partial = (
            partial_raw_chain_at_target(model, x_full, target_u)
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64)
        )
        partial_full = np.zeros_like(total)
        partial_full[:, :, :lag] = partial
        missing = total - partial_full
        total_windows.append(np.abs(total))
        partial_windows.append(np.abs(partial_full))
        missing_windows.append(np.abs(missing))
        total_mass = float(np.sum(np.abs(total)[offdiag, :]))
        tail_mass = float(np.sum(np.abs(total[:, :, lag:])[offdiag, :]))
        if total_mass <= EPS:
            tail_undefined += 1
        else:
            tail_ratios.append(tail_mass / total_mass)

    j_total = np.mean(np.stack(total_windows, axis=0), axis=0)
    j_partial = np.mean(np.stack(partial_windows, axis=0), axis=0)
    j_missing = np.mean(np.stack(missing_windows, axis=0), axis=0)
    s_total_nominal = np.max(j_total[:, :, :lag], axis=2)
    s_partial_nominal = np.max(j_partial[:, :, :lag], axis=2)
    s_history = np.max(j_total, axis=2)
    total_mass = _offdiagonal_mass(j_total)
    missing_mass = _offdiagonal_mass(j_missing)
    if total_mass <= EPS:
        missing_ratio = None
        missing_reason = "zero_total_offdiagonal_raw_chain_mass"
    else:
        missing_ratio = float(missing_mass / (total_mass + EPS))
        missing_reason = None
    pearson, pearson_reason = pearson_with_reason(
        s_partial_nominal,
        s_total_nominal,
    )
    if graph is None:
        edge_count = None
        topk = None
        total_metrics = None
        partial_metrics = None
    else:
        graph_array = np.asarray(graph)
        graph_2d = np.any(graph_array != 0, axis=2) if graph_array.ndim == 3 else graph_array != 0
        graph_2d = graph_2d.copy()
        np.fill_diagonal(graph_2d, 0)
        edge_count = int(np.sum(graph_2d))
        topk = exact_topk_jaccard(
            s_partial_nominal,
            s_total_nominal,
            edge_count,
        )
        total_metrics = canonical_metric_adapter(graph_array, s_total_nominal)
        partial_metrics = canonical_metric_adapter(graph_array, s_partial_nominal)
    tail = _distribution_summary(tail_ratios)
    tail["undefined_count"] = int(tail_undefined)
    tail["nominal_lag"] = lag
    tail["bounded_horizon"] = int(attribution_horizon)
    return SampledAuditResult(
        j_bar_total=j_total,
        j_bar_partial=j_partial,
        j_bar_missing=j_missing,
        s_total_nominal=s_total_nominal,
        s_partial_nominal=s_partial_nominal,
        s_total_bounded_history=s_history,
        target_indices=indices,
        attribution_horizon=attribution_horizon,
        nominal_lag=lag,
        missing_route_relative_magnitude=missing_ratio,
        missing_route_undefined_reason=missing_reason,
        partial_total_nominal_pearson=pearson,
        partial_total_nominal_pearson_reason=pearson_reason,
        partial_total_nominal_topk_jaccard=topk,
        temporal_tail_statistics=tail,
        total_nominal_metrics=total_metrics,
        partial_nominal_metrics=partial_metrics,
    )


def condition_coordinate_mixing_audit(
    adapter: LegacyConcatXOnlyAdapter,
    x_full,
    *,
    target_indices: Sequence[int],
    attribution_horizon: int,
) -> Dict[str, object]:
    """Quantify how broadly each auxiliary coordinate depends on raw sources."""

    indices = [int(value) for value in target_indices]
    raw = as_raw_bdt(
        x_full,
        device=adapter.device,
        dtype=torch.float32,
        require_grad=True,
    )
    condition = adapter.condition_sequence(raw)
    max_source_shares: List[float] = []
    normalized_entropies: List[float] = []
    effective_source_counts: List[float] = []
    undefined = 0
    source_profiles = []
    for target_u in indices:
        condition_time = target_u - 1
        start = max(0, target_u - attribution_horizon)
        for coordinate in range(adapter.d_cond):
            gradient = torch.autograd.grad(
                condition[0, condition_time, coordinate],
                raw,
                create_graph=False,
                retain_graph=True,
                allow_unused=False,
            )[0][0, :, start:target_u]
            source_mass = torch.sum(torch.abs(gradient), dim=1)
            total = float(torch.sum(source_mass).detach())
            if total <= EPS:
                undefined += 1
                continue
            shares = (source_mass / total).detach().cpu().numpy().astype(np.float64)
            entropy = float(-np.sum(shares * np.log(shares + EPS)))
            normalized = entropy / math.log(adapter.d) if adapter.d > 1 else 0.0
            max_source_shares.append(float(np.max(shares)))
            normalized_entropies.append(normalized)
            effective_source_counts.append(float(np.exp(entropy)))
            source_profiles.append({
                "target_index": target_u,
                "condition_time": condition_time,
                "condition_coordinate": coordinate,
                "source_shares": shares.tolist(),
            })
    return {
        "architecture_label": AUDIT_LABELS["coordinate_ambiguous"],
        "reason": (
            "auxiliary coordinates are learned mixtures and do not have a declared "
            "one-to-one mapping to original source variables"
        ),
        "target_indices": indices,
        "attribution_horizon": int(attribution_horizon),
        "defined_coordinate_time_count": len(max_source_shares),
        "undefined_coordinate_time_count": int(undefined),
        "max_source_share": _distribution_summary(max_source_shares),
        "normalized_source_entropy": _distribution_summary(normalized_entropies),
        "effective_source_count": _distribution_summary(effective_source_counts),
        "source_profiles": source_profiles,
    }


def fixed_target_baseline_interventions(
    adapter: LegacyBaselineAdapter,
    x_full,
    *,
    perturbation_seed: int,
) -> Dict[str, object]:
    raw = as_raw_bdt(x_full, device=adapter.device, dtype=torch.float32)
    indices = np.arange(adapter.lag, raw.shape[2], dtype=np.int64)
    clean_target = adapter.raw_targets(raw, indices).detach()
    zero = torch.zeros_like(raw)
    shuffled = raw.detach().clone()
    rng = np.random.default_rng(perturbation_seed)
    for source in range(adapter.d):
        permutation = torch.as_tensor(
            rng.permutation(raw.shape[2]),
            device=raw.device,
            dtype=torch.long,
        )
        shuffled[0, source] = raw[0, source, permutation]
    conditions = {"clean": raw, "mask_x": zero, "shuffle_x": shuffled}
    mse = {}
    for name, route in conditions.items():
        prediction = adapter.predict_from_raw(route, indices)
        mse[name] = float(torch.mean((prediction - clean_target) ** 2).detach().cpu())
    return {
        "fixed_target_prediction_mse": mse,
        "fixed_target_prediction_mse_delta": {
            name: value - mse["clean"] for name, value in mse.items()
        },
        "target_policy": "clean_raw_target_fixed",
        "mask_value": 0.0,
        "shuffle_axis": "time_within_each_source_variable",
        "perturbation_seed": int(perturbation_seed),
    }


def build_audit_profile(
    *,
    architecture: str,
    sampled_audit: SampledAuditResult,
    has_auxiliary_route: bool,
) -> Dict[str, object]:
    if has_auxiliary_route:
        partial_score_route = AUDIT_LABELS["partial"]
        penalty_route = AUDIT_LABELS["partial"]
        score_penalty_alignment = AUDIT_LABELS["partial"]
        coordinate = AUDIT_LABELS["coordinate_ambiguous"]
    else:
        partial_score_route = AUDIT_LABELS["covered"]
        penalty_route = AUDIT_LABELS["covered"]
        score_penalty_alignment = AUDIT_LABELS["covered"]
        coordinate = AUDIT_LABELS["covered"]
    return {
        "architecture": architecture,
        "coverage_declaration": {
            "V_score_partial": "original raw input variables",
            "V_score_total_audit": "original raw input variables",
            "V_penalty": "original x-only legacy penalty",
            "P_pred": (
                ["raw-X route", "learned auxiliary-condition route"]
                if has_auxiliary_route
                else ["raw-X route"]
            ),
            "M_coord": (
                "auxiliary coordinates have no one-to-one raw-variable identity"
                if has_auxiliary_route
                else "identity raw-variable mapping"
            ),
            "H_attr": {
                "primary_nominal_lag": int(sampled_audit.nominal_lag),
                "bounded_total_raw_chain_horizon": int(sampled_audit.attribution_horizon),
            },
        },
        "audit_dimensions": {
            "partial_score_route_completeness": partial_score_route,
            "legacy_penalty_route_completeness": penalty_route,
            "legacy_score_penalty_alignment": score_penalty_alignment,
            "auxiliary_coordinate_identity": coordinate,
            "bounded_total_raw_chain_horizon": AUDIT_LABELS["horizon_truncated"],
            "total_raw_chain_score_coordinate_alignment": AUDIT_LABELS["covered"],
        },
        "diagnostic_not_guarantee": True,
        "formal_result": False,
        "development_only": True,
    }
