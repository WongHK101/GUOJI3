"""Reusable report layer for claim-specific Jacobian coverage audits.

This module does not infer causal identifiability. It records whether a declared
Granger-predictive graph object is supported by the predictor routes, Jacobian
penalty, source-coordinate map, and audited temporal horizon.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np


SCHEMA_VERSION = "jacobian-coverage-audit/1.0"

COVERED = "COVERED"
PARTIALLY_COVERED = "PARTIALLY COVERED"
COORDINATE_AMBIGUOUS = "COORDINATE-AMBIGUOUS"
HORIZON_TRUNCATED = "HORIZON-TRUNCATED"
UNASSESSED = "UNASSESSED"

AUDIT_LABELS = frozenset({
    COVERED,
    PARTIALLY_COVERED,
    COORDINATE_AMBIGUOUS,
    HORIZON_TRUNCATED,
    UNASSESSED,
})


@dataclass(frozen=True)
class PredictiveRoute:
    """One architecture-declared route class into the predictor."""

    route_id: str
    description: str
    enters_prediction: bool
    interpreted_as_graph_knowledge: bool
    score_covered: bool
    penalty_covered: bool
    penalty_exempt: bool = False
    exemption_reason: Optional[str] = None

    def validate(self) -> None:
        if not self.route_id.strip():
            raise ValueError("route_id must be non-empty")
        if self.score_covered and not self.enters_prediction:
            raise ValueError(f"{self.route_id}: a non-predictive route cannot be score-covered")
        if self.penalty_covered and not self.enters_prediction:
            raise ValueError(f"{self.route_id}: a non-predictive route cannot be penalty-covered")
        if self.penalty_exempt and not self.exemption_reason:
            raise ValueError(f"{self.route_id}: penalty exemption requires a reason")
        if self.penalty_covered and self.penalty_exempt:
            raise ValueError(f"{self.route_id}: penalty-covered and penalty-exempt conflict")


@dataclass(frozen=True)
class CoverageDeclaration:
    """Five-part coverage declaration plus explicit audit decisions."""

    architecture: str
    graph_claim: str
    score_variables: Tuple[str, ...]
    penalty_variables: Tuple[str, ...]
    predictive_routes: Tuple[PredictiveRoute, ...]
    coordinate_mapping: str
    coordinate_identity_valid: Optional[bool]
    primary_score_horizon: int
    attribution_horizon: int
    required_support_horizon: Optional[int]
    omitted_mass_beyond_horizon_assessed: bool
    score_penalty_coordinate_compatible: Optional[bool]
    score_penalty_horizon_relation: str

    def validate(self) -> None:
        if not self.architecture.strip() or not self.graph_claim.strip():
            raise ValueError("architecture and graph_claim must be non-empty")
        if not self.score_variables:
            raise ValueError("score_variables must be declared")
        if not self.penalty_variables:
            raise ValueError("penalty_variables must be declared")
        if self.primary_score_horizon <= 0 or self.attribution_horizon <= 0:
            raise ValueError("score and attribution horizons must be positive")
        if self.attribution_horizon < self.primary_score_horizon:
            raise ValueError("attribution_horizon cannot be shorter than primary_score_horizon")
        if self.required_support_horizon is not None and self.required_support_horizon <= 0:
            raise ValueError("required_support_horizon must be positive when supplied")
        route_ids = [route.route_id for route in self.predictive_routes]
        if len(route_ids) != len(set(route_ids)):
            raise ValueError("predictive route identifiers must be unique")
        if not self.predictive_routes:
            raise ValueError("at least one predictive route must be declared")
        for route in self.predictive_routes:
            route.validate()
        if not any(route.enters_prediction for route in self.predictive_routes):
            raise ValueError("at least one declared route must enter prediction")


@dataclass(frozen=True)
class AuditProfile:
    score_route_completeness: str
    penalty_route_completeness: str
    score_penalty_alignment: str
    coordinate_validity: str
    horizon_validity: str

    def validate(self) -> None:
        for field, value in asdict(self).items():
            if value not in AUDIT_LABELS:
                raise ValueError(f"{field} has unsupported audit label {value!r}")


@dataclass(frozen=True)
class AuditReport:
    schema_version: str
    declaration: CoverageDeclaration
    profile: AuditProfile
    diagnostics: Mapping[str, object]
    provenance: Mapping[str, object]
    score_object_files: Mapping[str, str]
    diagnostic_not_guarantee: bool = True

    def validate(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"unsupported schema_version {self.schema_version!r}")
        self.declaration.validate()
        self.profile.validate()
        _validate_finite_tree(self.diagnostics)
        if not self.provenance:
            raise ValueError("provenance must not be empty")
        if not self.score_object_files:
            raise ValueError("score_object_files must not be empty")
        if not self.diagnostic_not_guarantee:
            raise ValueError("coverage reports must retain diagnostic_not_guarantee=true")

    def to_dict(self) -> Dict[str, object]:
        self.validate()
        return {
            "schema_version": self.schema_version,
            "coverage_declaration": _dataclass_to_dict(self.declaration),
            "audit_profile": asdict(self.profile),
            "diagnostics": dict(self.diagnostics),
            "provenance": dict(self.provenance),
            "score_object_files": dict(self.score_object_files),
            "diagnostic_not_guarantee": self.diagnostic_not_guarantee,
        }

    def write_json(self, output_path: Path | str) -> Path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        return path

    def write_profile_csv(self, output_path: Path | str) -> Path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=["dimension", "status"],
                lineterminator="\n",
            )
            writer.writeheader()
            for dimension, status in asdict(self.profile).items():
                writer.writerow({"dimension": dimension, "status": status})
        return path


def build_audit_profile(declaration: CoverageDeclaration) -> AuditProfile:
    """Derive independent coverage labels from a validated declaration."""

    declaration.validate()
    graph_routes = [
        route
        for route in declaration.predictive_routes
        if route.enters_prediction and route.interpreted_as_graph_knowledge
    ]
    predictive_routes = [
        route for route in declaration.predictive_routes if route.enters_prediction
    ]
    if not graph_routes:
        score_status = UNASSESSED
    elif all(route.score_covered for route in graph_routes):
        score_status = COVERED
    else:
        score_status = PARTIALLY_COVERED

    if all(route.penalty_covered or route.penalty_exempt for route in predictive_routes):
        penalty_status = COVERED
    else:
        penalty_status = PARTIALLY_COVERED

    score_routes = {route.route_id for route in predictive_routes if route.score_covered}
    penalty_routes = {
        route.route_id
        for route in predictive_routes
        if route.penalty_covered or route.penalty_exempt
    }
    if declaration.score_penalty_coordinate_compatible is None:
        alignment_status = UNASSESSED
    elif (
        score_routes == penalty_routes
        and declaration.score_penalty_coordinate_compatible
        and declaration.score_penalty_horizon_relation.strip()
    ):
        alignment_status = COVERED
    else:
        alignment_status = PARTIALLY_COVERED

    if declaration.coordinate_identity_valid is None:
        coordinate_status = UNASSESSED
    elif declaration.coordinate_identity_valid:
        coordinate_status = COVERED
    else:
        coordinate_status = COORDINATE_AMBIGUOUS

    required = declaration.required_support_horizon
    if required is None:
        horizon_status = (
            COVERED
            if declaration.omitted_mass_beyond_horizon_assessed
            else HORIZON_TRUNCATED
        )
    elif declaration.attribution_horizon >= required:
        horizon_status = COVERED
    else:
        horizon_status = HORIZON_TRUNCATED

    profile = AuditProfile(
        score_route_completeness=score_status,
        penalty_route_completeness=penalty_status,
        score_penalty_alignment=alignment_status,
        coordinate_validity=coordinate_status,
        horizon_validity=horizon_status,
    )
    profile.validate()
    return profile


def route_chain_rule(
    partial_raw: np.ndarray,
    partial_auxiliary: np.ndarray,
    auxiliary_raw_jacobian: np.ndarray,
) -> np.ndarray:
    """Return ``partial_raw + partial_auxiliary @ auxiliary_raw_jacobian``."""

    partial_raw = np.asarray(partial_raw, dtype=np.float64)
    partial_auxiliary = np.asarray(partial_auxiliary, dtype=np.float64)
    auxiliary_raw_jacobian = np.asarray(auxiliary_raw_jacobian, dtype=np.float64)
    indirect = np.einsum(
        "...ij,...jk->...ik",
        partial_auxiliary,
        auxiliary_raw_jacobian,
        optimize=False,
    )
    if indirect.shape != partial_raw.shape:
        raise ValueError("chain-rule terms have incompatible shapes")
    return partial_raw + indirect


def finite_support_upper_bound(nominal_lag: int, transform_support: int) -> int:
    """Raw lag bound for K transformed lags with finite causal support R."""

    if nominal_lag <= 0 or transform_support <= 0:
        raise ValueError("nominal_lag and transform_support must be positive")
    return nominal_lag + transform_support - 1


def cross_source_leakage(jacobian: np.ndarray, *, eps: float = 1e-12) -> float:
    """L1 mass outside target-source diagonal for a source map."""

    values = np.abs(np.asarray(jacobian, dtype=np.float64))
    if values.ndim < 2 or values.shape[0] != values.shape[1]:
        raise ValueError("jacobian must begin with equal target and source dimensions")
    mask = ~np.eye(values.shape[0], dtype=bool)
    off_mass = float(np.sum(values[mask, ...]))
    total_mass = float(np.sum(values))
    return off_mass / (total_mass + eps)


def topk_boundary_margin(scores: Sequence[float], k: int) -> float:
    values = np.sort(np.asarray(scores, dtype=np.float64).reshape(-1))[::-1]
    if k <= 0 or k >= values.size:
        raise ValueError("k must leave at least one selected and one unselected item")
    return float(values[k - 1] - values[k])


def topk_stability_guaranteed(
    reference_scores: Sequence[float],
    candidate_scores: Sequence[float],
    k: int,
) -> bool:
    """Apply the sufficient ``margin > 2 * sup-norm difference`` test."""

    reference = np.asarray(reference_scores, dtype=np.float64).reshape(-1)
    candidate = np.asarray(candidate_scores, dtype=np.float64).reshape(-1)
    if reference.shape != candidate.shape:
        raise ValueError("score vectors must have the same shape")
    delta = float(np.max(np.abs(reference - candidate)))
    return topk_boundary_margin(reference, k) > 2.0 * delta


def _dataclass_to_dict(value) -> Dict[str, object]:
    result = asdict(value)
    result["predictive_routes"] = [asdict(route) for route in value.predictive_routes]
    return result


def _validate_finite_tree(value: object, path: str = "diagnostics") -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} contains a nonfinite value")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            _validate_finite_tree(item, f"{path}.{key}")
        return
    if isinstance(value, np.ndarray):
        if not np.all(np.isfinite(value)):
            raise ValueError(f"{path} contains a nonfinite array")
        return
    if isinstance(value, Iterable):
        for index, item in enumerate(value):
            _validate_finite_tree(item, f"{path}[{index}]")
        return
    raise TypeError(f"{path} contains unsupported diagnostic type {type(value).__name__}")
