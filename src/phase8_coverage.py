"""Phase 8 coverage-aligned raw-chain candidate and audit utilities.

The comparator adapters in this module wrap immutable legacy models. They add
read-only evaluation surfaces without changing comparator prediction, target,
penalty, objective, or checkpoint semantics. Only
``CoverageAlignedRawChainJRNGC`` is a new candidate method.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from knowledge_metrics import topk_edges_exact
from mamba_jrngc_pilot import BaselineJRNGC, MambaJRNGC, MambaPreprocessor, ResidualBlock


EPS = 1e-12
FD_ATOL = 1e-5
FD_RTOL = 1e-3


@dataclass(frozen=True)
class Phase8ModelConfig:
    d: int
    lag: int
    layers: int = 3
    hidden: int = 32
    dropout: float = 0.0
    d_cond: int = 4
    d_state: int = 4
    d_conv: int = 4
    expand: int = 2
    jacobian_lam: float = 0.01
    dtype: str = "float32"


@dataclass(frozen=True)
class AttributionResult:
    j_bar_total: np.ndarray
    j_bar_partial: np.ndarray
    j_bar_missing: np.ndarray
    eligible_window_count_by_lag: np.ndarray
    s_partial_nominal: np.ndarray
    s_gc_total: np.ndarray
    j_bar_total_lag1: np.ndarray
    s_reliable_history: np.ndarray
    s_prefix_all: np.ndarray
    prefix_maximizing_lag: np.ndarray
    prefix_maximizing_lag_window_count: np.ndarray
    prefix_max_outside_reliable: np.ndarray
    h_reliable: np.ndarray
    n_min: int
    temporal_tail_statistics: Mapping[str, object]
    m_missing: Optional[float]
    m_missing_undefined_reason: Optional[str]
    nominal_partial_total_pearson: Optional[float]
    nominal_pearson_undefined_reason: Optional[str]
    nominal_partial_total_topk_jaccard: Optional[float]

    def as_serializable(self) -> Dict[str, object]:
        out = asdict(self)
        for key, value in list(out.items()):
            if isinstance(value, np.ndarray):
                out[key] = value.tolist()
        return out


def torch_dtype(name: str) -> torch.dtype:
    if name == "float32":
        return torch.float32
    if name == "float64":
        return torch.float64
    raise ValueError(f"Unsupported dtype: {name}")


def as_raw_bdt(x, *, device: torch.device, dtype: torch.dtype, require_grad: bool = False) -> torch.Tensor:
    raw = torch.as_tensor(x, device=device, dtype=dtype)
    if raw.dim() == 2:
        raw = raw.unsqueeze(0)
    if raw.dim() != 3 or raw.shape[0] != 1:
        raise ValueError(f"Expected raw shape (d,T) or (1,d,T), got {tuple(raw.shape)}")
    if require_grad:
        raw = raw.detach().clone().requires_grad_(True)
    return raw


def target_indices(T: int, lag: int) -> np.ndarray:
    if lag < 1 or T <= lag:
        raise ValueError(f"Invalid T={T}, lag={lag}")
    return np.arange(lag, T, dtype=np.int64)


def eligible_targets_for_lag(T: int, lag: int, raw_lag: int) -> np.ndarray:
    if raw_lag < 1 or raw_lag >= T:
        raise ValueError(f"raw_lag must be in [1,{T - 1}], got {raw_lag}")
    return np.arange(max(lag, raw_lag), T, dtype=np.int64)


def _history_from_raw(raw: torch.Tensor, indices: Sequence[int], lag: int) -> torch.Tensor:
    rows = [raw[0, :, u - lag:u] for u in indices]
    return torch.stack(rows, dim=0)


def _targets_from_raw(raw: torch.Tensor, indices: Sequence[int]) -> torch.Tensor:
    return torch.stack([raw[0, :, u] for u in indices], dim=0)


class LegacyComparatorAdapter:
    """Composition-only adapter around an immutable legacy comparator."""

    comparator_name = "legacy"
    has_auxiliary_route = False

    def __init__(self, model: nn.Module):
        self.model = model
        self.d = int(model.d)
        self.lag = int(model.lag)

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.model.parameters()).dtype

    def raw_targets(self, x_full, indices: Optional[Sequence[int]] = None) -> torch.Tensor:
        raw = as_raw_bdt(x_full, device=self.device, dtype=torch.float32)
        idx = target_indices(raw.shape[2], self.lag) if indices is None else np.asarray(indices, dtype=np.int64)
        return _targets_from_raw(raw, idx)

    def predict_from_raw(
        self,
        raw_bdt: torch.Tensor,
        indices: Sequence[int],
        auxiliary_override: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if auxiliary_override is not None:
            raise ValueError("Baseline comparator has no auxiliary route")
        history = _history_from_raw(raw_bdt, indices, self.lag)
        return self.model(history)

    def predictions(self, x_full, indices: Optional[Sequence[int]] = None) -> torch.Tensor:
        raw = as_raw_bdt(x_full, device=self.device, dtype=torch.float32)
        idx = target_indices(raw.shape[2], self.lag) if indices is None else np.asarray(indices, dtype=np.int64)
        return self.predict_from_raw(raw, idx)

    def pure_mse(self, x_full, indices: Optional[Sequence[int]] = None) -> torch.Tensor:
        raw = as_raw_bdt(x_full, device=self.device, dtype=torch.float32)
        idx = target_indices(raw.shape[2], self.lag) if indices is None else np.asarray(indices, dtype=np.int64)
        pred = self.predict_from_raw(raw, idx)
        target = _targets_from_raw(raw, idx)
        return torch.mean((pred - target) ** 2)

    def direct_total_objective(self, x_full) -> torch.Tensor:
        return self.model.compute_loss(x_full)

    def loss_components(self, x_full) -> Dict[str, torch.Tensor]:
        mse = self.pure_mse(x_full)
        total = self.direct_total_objective(x_full)
        penalty = total - mse
        return {
            "fixed_target_prediction_mse": mse,
            "jacobian_penalty": penalty,
            "total_regularized_objective": total,
        }

    def partial_nominal_score(self, x_full) -> np.ndarray:
        gc = np.asarray(self.model.get_gc_matrix(x_full), dtype=np.float64)
        return gc if gc.ndim == 2 else np.max(gc, axis=2)

    def predict_partial_from_raw(self, raw_bdt: torch.Tensor, target_u: int) -> Tuple[torch.Tensor, torch.Tensor]:
        history = raw_bdt[0, :, target_u - self.lag:target_u].detach().clone().requires_grad_(True)
        pred = self.model(history.unsqueeze(0))[0]
        return pred, history


class LegacyBaselineAdapter(LegacyComparatorAdapter):
    comparator_name = "baseline_jrngc"

    def __init__(self, model: BaselineJRNGC):
        if not isinstance(model, BaselineJRNGC):
            raise TypeError("LegacyBaselineAdapter requires BaselineJRNGC")
        super().__init__(model)


class LegacyConcatXOnlyAdapter(LegacyComparatorAdapter):
    comparator_name = "concat_x_only"
    has_auxiliary_route = True

    def __init__(self, model: MambaJRNGC):
        mro_names = {base.__name__ for base in type(model).__mro__}
        if not isinstance(model, MambaJRNGC) and "MambaJRNGC" not in mro_names:
            raise TypeError("LegacyConcatXOnlyAdapter requires MambaJRNGC")
        super().__init__(model)
        self.d_cond = int(model.d_cond)

    def condition_sequence(self, raw_bdt: torch.Tensor) -> torch.Tensor:
        condition, _ = self.model.preprocessor(raw_bdt)
        return condition

    def predict_from_raw(
        self,
        raw_bdt: torch.Tensor,
        indices: Sequence[int],
        auxiliary_override: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        condition = self.condition_sequence(raw_bdt) if auxiliary_override is None else auxiliary_override
        rows = []
        for u in indices:
            x_hist = raw_bdt[0, :, u - self.lag:u]
            c_hist = condition[0, u - self.lag:u, :].transpose(0, 1)
            rows.append(torch.cat([x_hist, c_hist], dim=0))
        xz = torch.stack(rows, dim=0)
        return self.model(xz.flatten(start_dim=1))

    def predict_partial_from_raw(self, raw_bdt: torch.Tensor, target_u: int) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            condition = self.condition_sequence(raw_bdt.detach())
        x_hist = raw_bdt[0, :, target_u - self.lag:target_u].detach().clone().requires_grad_(True)
        c_hist = condition[0, target_u - self.lag:target_u, :].transpose(0, 1).detach()
        xz = torch.cat([x_hist, c_hist], dim=0).unsqueeze(0)
        return self.model(xz.flatten(start_dim=1))[0], x_hist


class LegacyFullAuxAdapter(LegacyConcatXOnlyAdapter):
    comparator_name = "full_auxiliary"

    def __init__(self, model: MambaJRNGC, *, lambda_mode: str):
        if lambda_mode not in {"equal", "lc10"}:
            raise ValueError(f"Unsupported full auxiliary lambda mode: {lambda_mode}")
        super().__init__(model)
        self.lambda_mode = lambda_mode


class Phase8NoAuxInputSpaceControl:
    """New matched input-space control; never a legacy replication claim."""

    control_status = "new_matched_input_space_control_not_legacy_replication"
    graph_recovery_evidence_allowed = False

    def __init__(self, model):
        self.model = model
        self.d = int(model.d)
        self.lag = int(model.lag)

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.model.parameters()).dtype

    def predict_from_raw(self, raw_bdt: torch.Tensor, indices: Sequence[int]) -> torch.Tensor:
        filtered = self.model.filter_sequence(raw_bdt)
        histories = [filtered[0, u - self.lag:u, :].transpose(0, 1) for u in indices]
        return self.model(torch.stack(histories, dim=0))

    def raw_targets(self, raw_bdt: torch.Tensor, indices: Sequence[int]) -> torch.Tensor:
        return _targets_from_raw(raw_bdt, indices)

    def fixed_target_prediction_mse(
        self,
        raw_route_bdt: torch.Tensor,
        clean_target_bdt: torch.Tensor,
        indices: Sequence[int],
    ) -> torch.Tensor:
        prediction = self.predict_from_raw(raw_route_bdt, indices)
        target = self.raw_targets(clean_target_bdt, indices)
        return torch.mean((prediction - target) ** 2)


class CoverageAlignedRawChainJRNGC(nn.Module):
    """New Phase 8 candidate with full-prefix raw-chain regularization."""

    method_name = "coverage_aligned_raw_chain"

    def __init__(self, cfg: Phase8ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.d = cfg.d
        self.lag = cfg.lag
        self.d_cond = cfg.d_cond
        self.jacobian_lam = cfg.jacobian_lam
        dtype = torch_dtype(cfg.dtype)
        self.preprocessor = MambaPreprocessor(
            cfg.d,
            d_state=cfg.d_state,
            d_conv=cfg.d_conv,
            d_cond=cfg.d_cond,
        )
        total_dim = (cfg.d + cfg.d_cond) * cfg.lag
        self.inputgate = nn.Linear(total_dim, cfg.hidden)
        self.outputgate = nn.Linear(cfg.hidden, cfg.d)
        self.encoders = nn.ModuleList([
            ResidualBlock(cfg.hidden, cfg.hidden, cfg.hidden, cfg.dropout)
            for _ in range(cfg.layers)
        ])
        self.to(dtype=dtype)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    def forward(self, xz_flat: torch.Tensor) -> torch.Tensor:
        h = self.inputgate(xz_flat.to(self.dtype))
        for encoder in self.encoders:
            h = encoder(h)
        return self.outputgate(h)

    def condition_sequence(self, raw_bdt: torch.Tensor) -> torch.Tensor:
        condition, _ = self.preprocessor(raw_bdt)
        return condition

    def predict_from_raw(
        self,
        raw_bdt: torch.Tensor,
        indices: Sequence[int],
        auxiliary_override: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        condition = self.condition_sequence(raw_bdt) if auxiliary_override is None else auxiliary_override
        rows = []
        for u in indices:
            x_hist = raw_bdt[0, :, u - self.lag:u]
            c_hist = condition[0, u - self.lag:u, :].transpose(0, 1)
            rows.append(torch.cat([x_hist, c_hist], dim=0))
        return self.forward(torch.stack(rows, dim=0).flatten(start_dim=1))

    def raw_targets(self, raw_bdt: torch.Tensor, indices: Sequence[int]) -> torch.Tensor:
        return _targets_from_raw(raw_bdt, indices)

    def pure_mse(self, raw_bdt: torch.Tensor, indices: Optional[Sequence[int]] = None) -> torch.Tensor:
        idx = target_indices(raw_bdt.shape[2], self.lag) if indices is None else np.asarray(indices, dtype=np.int64)
        pred = self.predict_from_raw(raw_bdt, idx)
        return torch.mean((pred - self.raw_targets(raw_bdt, idx)) ** 2)

    def predict_partial_from_raw(self, raw_bdt: torch.Tensor, target_u: int) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            condition = self.condition_sequence(raw_bdt.detach())
        x_hist = raw_bdt[0, :, target_u - self.lag:target_u].detach().clone().requires_grad_(True)
        c_hist = condition[0, target_u - self.lag:target_u, :].transpose(0, 1).detach()
        xz = torch.cat([x_hist, c_hist], dim=0).unsqueeze(0)
        return self.forward(xz.flatten(start_dim=1))[0], x_hist

    def loss_components(self, raw_bdt: torch.Tensor, schedule_entry: Mapping[str, object]) -> Dict[str, torch.Tensor]:
        if not raw_bdt.requires_grad:
            raw_bdt = raw_bdt.detach().clone().requires_grad_(True)
        mse = self.pure_mse(raw_bdt)
        penalty_unweighted = sampled_lag_balanced_penalty(self, raw_bdt, schedule_entry, create_graph=True)
        penalty = self.jacobian_lam * penalty_unweighted
        return {
            "fixed_target_prediction_mse": mse,
            "jacobian_penalty": penalty,
            "total_regularized_objective": mse + penalty,
        }


def make_legacy_baseline(cfg: Phase8ModelConfig) -> LegacyBaselineAdapter:
    model = BaselineJRNGC(
        d=cfg.d,
        lag=cfg.lag,
        layers=cfg.layers,
        hidden=cfg.hidden,
        dropout=cfg.dropout,
        jacobian_lam=cfg.jacobian_lam,
    )
    return LegacyBaselineAdapter(model)


def make_legacy_concat(cfg: Phase8ModelConfig) -> LegacyConcatXOnlyAdapter:
    model = MambaJRNGC(
        d=cfg.d,
        lag=cfg.lag,
        layers=cfg.layers,
        hidden=cfg.hidden,
        dropout=cfg.dropout,
        jacobian_lam=cfg.jacobian_lam,
        d_state=cfg.d_state,
        d_cond=cfg.d_cond,
        use_time_weight_loss=False,
    )
    return LegacyConcatXOnlyAdapter(model)


def make_legacy_full_aux(cfg: Phase8ModelConfig, lambda_mode: str) -> LegacyFullAuxAdapter:
    # Lazy import preserves the frozen implementation as the source of truth.
    from experiments.risk_mitigation_20260515.run_full_aux_penalty import (
        MambaConcatFullPenaltyJRNGC,
    )

    lam_c = cfg.jacobian_lam if lambda_mode == "equal" else 10.0 * cfg.jacobian_lam
    model = MambaConcatFullPenaltyJRNGC(
        d=cfg.d,
        lag=cfg.lag,
        layers=cfg.layers,
        hidden=cfg.hidden,
        dropout=cfg.dropout,
        jacobian_lam=cfg.jacobian_lam,
        d_state=cfg.d_state,
        d_cond=cfg.d_cond,
        use_time_weight_loss=False,
        lam_x=cfg.jacobian_lam,
        lam_c=lam_c,
    )
    return LegacyFullAuxAdapter(model, lambda_mode=lambda_mode)


def make_no_aux_input_space_control(cfg: Phase8ModelConfig) -> Phase8NoAuxInputSpaceControl:
    from repaired_istf import RawChainMambaISTFJRNGC, RepairedISTFConfig

    repaired_cfg = RepairedISTFConfig(
        d=cfg.d,
        lag=cfg.lag,
        attribution_horizon=max(32, cfg.lag),
        layers=cfg.layers,
        hidden=cfg.hidden,
        dropout=cfg.dropout,
        jacobian_lam=cfg.jacobian_lam,
        identity_lam=0.0,
        d_state=cfg.d_state,
        mamba_expand=cfg.expand,
        mamba_d_conv=cfg.d_conv,
        dtype=cfg.dtype,
    )
    return Phase8NoAuxInputSpaceControl(RawChainMambaISTFJRNGC(repaired_cfg))


METHOD_IMPLEMENTATION_REGISTRY = {
    "baseline_jrngc": "LegacyBaselineAdapter(BaselineJRNGC)",
    "concat_x_only": "LegacyConcatXOnlyAdapter(MambaJRNGC)",
    "full_aux_equal_lambda": "LegacyFullAuxAdapter(MambaConcatFullPenaltyJRNGC,equal)",
    "full_aux_lc10": "LegacyFullAuxAdapter(MambaConcatFullPenaltyJRNGC,lc10)",
    "coverage_aligned_raw_chain": "CoverageAlignedRawChainJRNGC",
    "concat_fixed_target_interventions": "LegacyConcatXOnlyAdapter(MambaJRNGC)",
    "no_aux_fixed_target_interventions": "RawChainMambaISTFJRNGC(new_matched_control)",
}
for _d_cond in (1, 2, 4, 8, 16):
    METHOD_IMPLEMENTATION_REGISTRY[f"concat_dcond_{_d_cond}"] = "LegacyConcatXOnlyAdapter(MambaJRNGC)"


def canonical_json_sha256(payload: object) -> str:
    data = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    return hashlib.sha256(data).hexdigest()


def build_balanced_lag_schedule(
    *,
    T: int,
    lag: int,
    d_out: int,
    max_iter: int,
    seed: int,
) -> List[Dict[str, object]]:
    """Build the frozen lag-first schedule.

    Lag and output streams are cyclic permutations. Conditional target-window
    streams are independently permuted for each lag and consumed cyclically.
    """
    if d_out < 2 or T <= lag + 1:
        raise ValueError("Schedule requires at least two outputs and a nontrivial prefix")
    h_max = T - 1
    rng = np.random.default_rng(seed)
    lag_order = np.arange(1, h_max + 1, dtype=np.int64)
    output_order = np.arange(d_out, dtype=np.int64)
    rng.shuffle(lag_order)
    rng.shuffle(output_order)
    window_orders: Dict[int, np.ndarray] = {}
    window_cursor: Dict[int, int] = {}
    for h in range(1, h_max + 1):
        values = eligible_targets_for_lag(T, lag, h)
        local_rng = np.random.default_rng(seed + 104729 * h)
        local_rng.shuffle(values)
        window_orders[h] = values
        window_cursor[h] = 0

    schedule: List[Dict[str, object]] = []
    lag_stream_pos = 0
    output_stream_pos = 0
    for iteration in range(max_iter):
        lags = [
            int(lag_order[(lag_stream_pos + offset) % h_max])
            for offset in range(2)
        ]
        lag_stream_pos += 2
        outputs = [
            int(output_order[(output_stream_pos + offset) % d_out])
            for offset in range(2)
        ]
        output_stream_pos += 2
        if len(set(lags)) != 2 or len(set(outputs)) != 2:
            raise AssertionError("Frozen schedule produced a duplicate lag or output within one step")
        windows = []
        for h in lags:
            order = window_orders[h]
            cursor = window_cursor[h]
            windows.append(int(order[cursor % len(order)]))
            window_cursor[h] = cursor + 1
        schedule.append({
            "iteration": iteration,
            "lags": lags,
            "eligible_windows": windows,
            "output_targets": outputs,
        })
    return schedule


def schedule_sha256(schedule: Sequence[Mapping[str, object]]) -> str:
    return canonical_json_sha256(list(schedule))


def sampled_lag_balanced_penalty(
    model: CoverageAlignedRawChainJRNGC,
    raw_bdt: torch.Tensor,
    schedule_entry: Mapping[str, object],
    *,
    create_graph: bool,
) -> torch.Tensor:
    lags = [int(v) for v in schedule_entry["lags"]]  # type: ignore[index]
    windows = [int(v) for v in schedule_entry["eligible_windows"]]  # type: ignore[index]
    outputs = [int(v) for v in schedule_entry["output_targets"]]  # type: ignore[index]
    if len(lags) != 2 or len(windows) != 2 or len(outputs) != 2:
        raise ValueError("Estimator requires 2 lags, 1 window per lag, and 2 outputs")
    if len(set(lags)) != 2 or len(set(outputs)) != 2:
        raise ValueError("Estimator lags and outputs must be distinct within a step")
    T = int(raw_bdt.shape[2])
    h_max = T - 1
    sampled_sum = torch.zeros((), device=raw_bdt.device, dtype=raw_bdt.dtype)
    for h, u in zip(lags, windows):
        if u not in eligible_targets_for_lag(T, model.lag, h):
            raise ValueError(f"Window {u} is not eligible for lag {h}")
        pred = model.predict_from_raw(raw_bdt, [u])[0]
        for output in outputs:
            grad = torch.autograd.grad(
                pred[output],
                raw_bdt,
                create_graph=create_graph,
                retain_graph=True,
                allow_unused=False,
            )[0]
            sampled_sum = sampled_sum + torch.sum(torch.abs(grad[0, :, u - h]))
    denominator = len(lags) * 1 * len(outputs) * model.d * model.lag
    return (h_max / denominator) * sampled_sum


def exact_lag_balanced_objective(
    model: CoverageAlignedRawChainJRNGC,
    raw_bdt: torch.Tensor,
    *,
    create_graph: bool,
) -> torch.Tensor:
    T = int(raw_bdt.shape[2])
    total = torch.zeros((), device=raw_bdt.device, dtype=raw_bdt.dtype)
    for h in range(1, T):
        eligible = eligible_targets_for_lag(T, model.lag, h)
        lag_sum = torch.zeros_like(total)
        for u in eligible.tolist():
            pred = model.predict_from_raw(raw_bdt, [u])[0]
            for output in range(model.d):
                grad = torch.autograd.grad(
                    pred[output],
                    raw_bdt,
                    create_graph=create_graph,
                    retain_graph=True,
                    allow_unused=False,
                )[0]
                lag_sum = lag_sum + torch.sum(torch.abs(grad[0, :, u - h]))
        total = total + lag_sum / len(eligible)
    return total / (model.d * model.d * model.lag)


def _model_device_dtype(model) -> Tuple[torch.device, torch.dtype]:
    if isinstance(model, LegacyComparatorAdapter):
        return model.device, torch.float32
    return next(model.parameters()).device, next(model.parameters()).dtype


def _predict_total(model, raw_bdt: torch.Tensor, target_u: int) -> torch.Tensor:
    return model.predict_from_raw(raw_bdt, [target_u])[0]


def total_raw_chain_at_target(
    model,
    x_full,
    target_u: int,
    *,
    h_max: Optional[int] = None,
    create_graph: bool = False,
    detach_raw: bool = False,
) -> torch.Tensor:
    """Return signed total raw-chain Jacobian [target,source,raw_lag]."""
    device, dtype = _model_device_dtype(model)
    raw = as_raw_bdt(x_full, device=device, dtype=dtype, require_grad=not detach_raw)
    if detach_raw:
        raw = raw.detach()
    max_h = min(target_u, h_max if h_max is not None else target_u)
    pred = _predict_total(model, raw, target_u)
    rows = []
    for output in range(model.d):
        grad = torch.autograd.grad(
            pred[output],
            raw,
            create_graph=create_graph,
            retain_graph=True,
            allow_unused=True,
        )[0]
        if grad is None:
            raise RuntimeError("Raw-chain input is disconnected from prediction")
        rows.append(torch.stack([grad[0, :, target_u - h] for h in range(1, max_h + 1)], dim=1))
    return torch.stack(rows, dim=0)


def partial_raw_chain_at_target(model, x_full, target_u: int) -> torch.Tensor:
    """Return signed direct/partial Jacobian [target,source,K], h=1 first."""
    device, dtype = _model_device_dtype(model)
    raw = as_raw_bdt(x_full, device=device, dtype=dtype, require_grad=False)
    pred, history = model.predict_partial_from_raw(raw, target_u)
    rows = []
    for output in range(model.d):
        grad = torch.autograd.grad(
            pred[output],
            history,
            create_graph=False,
            retain_graph=True,
            allow_unused=False,
        )[0]
        rows.append(torch.stack([grad[:, model.lag - h] for h in range(1, model.lag + 1)], dim=1))
    return torch.stack(rows, dim=0)


def offdiag_vector(score: np.ndarray) -> np.ndarray:
    score = np.asarray(score, dtype=np.float64)
    if score.ndim != 2 or score.shape[0] != score.shape[1]:
        raise ValueError(f"Expected square score, got {score.shape}")
    return score[~np.eye(score.shape[0], dtype=bool)]


def pearson_with_reason(a: np.ndarray, b: np.ndarray) -> Tuple[Optional[float], Optional[str]]:
    av = offdiag_vector(a)
    bv = offdiag_vector(b)
    if av.size != bv.size or av.size < 2:
        return None, "insufficient_or_mismatched_entries"
    if np.std(av) <= EPS or np.std(bv) <= EPS:
        return None, "constant_offdiagonal_vector"
    av_centered = av - np.mean(av)
    bv_centered = bv - np.mean(bv)
    denominator = math.sqrt(float(np.dot(av_centered, av_centered) * np.dot(bv_centered, bv_centered)))
    if denominator <= EPS:
        return None, "zero_centered_norm"
    return float(np.dot(av_centered, bv_centered) / denominator), None


def exact_topk_jaccard(a: np.ndarray, b: np.ndarray, k: int) -> Optional[float]:
    if k < 1:
        return None
    ea = topk_edges_exact(a, k=k, exclude_diag=True)
    eb = topk_edges_exact(b, k=k, exclude_diag=True)
    union = ea | eb
    return None if not union else float(len(ea & eb) / len(union))


def extract_attribution_objects(
    model,
    x_full,
    *,
    true_edge_count: Optional[int] = None,
    n_min: Optional[int] = None,
) -> AttributionResult:
    """Compute the locked full-prefix and nominal-lag attribution objects.

    Absolute values are accumulated per window in float64. Raw lag h is stored
    at index h-1. Diagonal entries remain in all saved tensors and are removed
    only from diagnostics/metrics that explicitly say off-diagonal.
    """
    x_np = np.asarray(x_full)
    T = int(x_np.shape[1])
    H = T - 1
    idx = target_indices(T, model.lag)
    sum_total = np.zeros((model.d, model.d, H), dtype=np.float64)
    sum_partial = np.zeros_like(sum_total)
    sum_missing = np.zeros_like(sum_total)
    counts = np.zeros(H, dtype=np.int64)
    tail_values: List[float] = []
    undefined_tail = 0
    offdiag = ~np.eye(model.d, dtype=bool)

    for u in idx.tolist():
        total = total_raw_chain_at_target(model, x_np, u, h_max=H, create_graph=False)
        partial = partial_raw_chain_at_target(model, x_np, u)
        total_np = total.detach().cpu().numpy().astype(np.float64, copy=False)
        partial_np = partial.detach().cpu().numpy().astype(np.float64, copy=False)
        sum_total[:, :, :u] += np.abs(total_np)
        direct_full = np.zeros_like(total_np)
        direct_full[:, :, :model.lag] = partial_np
        sum_partial[:, :, :u] += np.abs(direct_full)
        sum_missing[:, :, :u] += np.abs(total_np - direct_full)
        counts[:u] += 1

        all_mass = float(np.sum(np.abs(total_np)[offdiag, :]))
        tail_mass = float(np.sum(np.abs(total_np[:, :, model.lag:])[offdiag, :]))
        if all_mass <= EPS:
            undefined_tail += 1
        else:
            tail_values.append(tail_mass / all_mass)

    if np.any(counts <= 0):
        raise AssertionError("Every full-prefix lag must have at least one eligible target")
    j_total = sum_total / counts.reshape(1, 1, H)
    j_partial = sum_partial / counts.reshape(1, 1, H)
    j_missing = sum_missing / counts.reshape(1, 1, H)
    s_partial = np.max(j_partial[:, :, :model.lag], axis=2)
    s_gc_total = np.max(j_total[:, :, :model.lag], axis=2)

    frozen_n_min = max(20, int(math.ceil(0.10 * len(idx)))) if n_min is None else int(n_min)
    reliable_positions = np.flatnonzero(counts >= frozen_n_min)
    if reliable_positions.size == 0:
        raise ValueError("Reliable-support set is empty")
    s_reliable = np.max(j_total[:, :, reliable_positions], axis=2)
    s_prefix = np.max(j_total, axis=2)
    max_pos = np.argmax(j_total, axis=2)
    max_lag = max_pos + 1
    max_counts = counts[max_pos]
    reliable_set = set((reliable_positions + 1).tolist())
    max_outside = np.vectorize(lambda h: int(h) not in reliable_set)(max_lag)

    total_offdiag_mass = float(np.sum(j_total[offdiag, :]))
    missing_offdiag_mass = float(np.sum(j_missing[offdiag, :]))
    if total_offdiag_mass <= EPS:
        m_missing = None
        m_reason = "zero_total_offdiagonal_mass"
    else:
        m_missing = missing_offdiag_mass / (total_offdiag_mass + EPS)
        m_reason = None

    pearson, pearson_reason = pearson_with_reason(s_partial, s_gc_total)
    jaccard = None if true_edge_count is None else exact_topk_jaccard(s_partial, s_gc_total, true_edge_count)
    tail_arr = np.asarray(tail_values, dtype=np.float64)
    tail_stats: Dict[str, object] = {
        "defined_window_count": int(tail_arr.size),
        "undefined_window_count": int(undefined_tail),
        "mean": None,
        "median": None,
        "p95": None,
        "maximum": None,
    }
    if tail_arr.size:
        tail_stats.update({
            "mean": float(np.mean(tail_arr)),
            "median": float(np.median(tail_arr)),
            "p95": float(np.quantile(tail_arr, 0.95)),
            "maximum": float(np.max(tail_arr)),
        })

    return AttributionResult(
        j_bar_total=j_total,
        j_bar_partial=j_partial,
        j_bar_missing=j_missing,
        eligible_window_count_by_lag=counts,
        s_partial_nominal=s_partial,
        s_gc_total=s_gc_total,
        j_bar_total_lag1=j_total[:, :, 0],
        s_reliable_history=s_reliable,
        s_prefix_all=s_prefix,
        prefix_maximizing_lag=max_lag,
        prefix_maximizing_lag_window_count=max_counts,
        prefix_max_outside_reliable=max_outside,
        h_reliable=reliable_positions + 1,
        n_min=frozen_n_min,
        temporal_tail_statistics=tail_stats,
        m_missing=None if m_missing is None else float(m_missing),
        m_missing_undefined_reason=m_reason,
        nominal_partial_total_pearson=pearson,
        nominal_pearson_undefined_reason=pearson_reason,
        nominal_partial_total_topk_jaccard=jaccard,
    )


def coefficient_r_total_lag1(j_bar_total_lag1: np.ndarray, A: np.ndarray) -> Optional[float]:
    recovered = offdiag_vector(np.asarray(j_bar_total_lag1, dtype=np.float64))
    truth = offdiag_vector(np.abs(np.asarray(A, dtype=np.float64)))
    if np.std(recovered) <= EPS or np.std(truth) <= EPS:
        return None
    recovered = recovered - np.mean(recovered)
    truth = truth - np.mean(truth)
    denominator = math.sqrt(float(np.dot(recovered, recovered) * np.dot(truth, truth)))
    return None if denominator <= EPS else float(np.dot(recovered, truth) / denominator)


def _concat_legacy_objective_with_routes(
    adapter: LegacyConcatXOnlyAdapter,
    raw_for_history: torch.Tensor,
    condition: torch.Tensor,
    raw_for_target: torch.Tensor,
) -> torch.Tensor:
    indices = target_indices(raw_for_history.shape[2], adapter.lag)
    rows = []
    targets = []
    for u in indices.tolist():
        x_with_target = raw_for_history[0, :, u - adapter.lag:u + 1]
        c_with_target = condition[0, u - adapter.lag:u + 1, :].transpose(0, 1)
        rows.append(torch.cat([x_with_target, c_with_target], dim=0))
        targets.append(raw_for_target[0, :, u])
    windows = torch.stack(rows, dim=0)
    target = torch.stack(targets, dim=0)
    pred = adapter.model(windows[:, :, :adapter.lag].flatten(start_dim=1))
    mse = torch.mean((pred - target) ** 2)
    penalty = adapter.model.compute_jacobian_loss(windows[:min(len(windows), 100)])
    return mse + penalty


def _shuffle_time_per_variable(raw: torch.Tensor, seed: int) -> torch.Tensor:
    out = raw.detach().clone()
    rng = np.random.default_rng(seed)
    T = raw.shape[2]
    for source in range(raw.shape[1]):
        perm = torch.as_tensor(rng.permutation(T), device=raw.device, dtype=torch.long)
        out[0, source] = raw[0, source, perm]
    return out


def _shuffle_condition_time(condition: torch.Tensor, seed: int) -> torch.Tensor:
    out = condition.detach().clone()
    rng = np.random.default_rng(seed)
    T = condition.shape[1]
    for coordinate in range(condition.shape[2]):
        perm = torch.as_tensor(rng.permutation(T), device=condition.device, dtype=torch.long)
        out[0, :, coordinate] = condition[0, perm, coordinate]
    return out


def fixed_target_concat_interventions(
    adapter: LegacyConcatXOnlyAdapter,
    x_full,
    *,
    perturbation_seed: int,
) -> Dict[str, object]:
    """Evaluate locked concat interventions with clean raw targets fixed."""
    raw = as_raw_bdt(x_full, device=adapter.device, dtype=torch.float32, require_grad=False)
    idx = target_indices(raw.shape[2], adapter.lag)
    clean_target = _targets_from_raw(raw, idx).detach()
    clean_c = adapter.condition_sequence(raw).detach()
    zero_x = torch.zeros_like(raw)
    zero_c = torch.zeros_like(clean_c)
    shuffled_x = _shuffle_time_per_variable(raw, perturbation_seed)
    shuffled_c_only = _shuffle_condition_time(clean_c, perturbation_seed + 1)
    shuffled_both_c = adapter.condition_sequence(shuffled_x).detach()

    route_inputs = {
        "clean": (raw, clean_c),
        "mask_x": (zero_x, clean_c),
        "mask_c": (raw, zero_c),
        "mask_both": (zero_x, zero_c),
        "shuffle_x_only": (shuffled_x, clean_c),
        "shuffle_c_only": (raw, shuffled_c_only),
        "shuffle_both_routes": (shuffled_x, shuffled_both_c),
    }
    fixed_mse: Dict[str, float] = {}
    for name, (x_route, c_route) in route_inputs.items():
        pred = adapter.predict_from_raw(x_route, idx, auxiliary_override=c_route)
        fixed_mse[name] = float(torch.mean((pred - clean_target) ** 2).detach().cpu())
    fixed_delta = {name: value - fixed_mse["clean"] for name, value in fixed_mse.items()}

    legacy_targets = {
        "clean": raw,
        "mask_x": zero_x,
        "mask_c": raw,
        "mask_both": zero_x,
        "shuffle_x_only": shuffled_x,
        "shuffle_c_only": raw,
        "shuffle_both_routes": shuffled_x,
    }
    legacy_objective: Dict[str, float] = {}
    for name, (x_route, c_route) in route_inputs.items():
        value = _concat_legacy_objective_with_routes(
            adapter,
            x_route,
            c_route,
            legacy_targets[name],
        )
        legacy_objective[name] = float(value.detach().cpu())
    legacy_delta = {name: value - legacy_objective["clean"] for name, value in legacy_objective.items()}

    return {
        "fixed_target_prediction_mse": fixed_mse,
        "fixed_target_prediction_mse_delta": fixed_delta,
        "legacy_total_regularized_objective": legacy_objective,
        "legacy_objective_delta": legacy_delta,
        "target_policy": "clean_raw_target_fixed",
        "mask_value": 0.0,
        "shuffle_axis": "time_within_each_coordinate",
        "perturbation_seed": int(perturbation_seed),
    }


def finite_difference_total_raw_chain_audit() -> Dict[str, object]:
    torch.manual_seed(8101)
    rng = np.random.default_rng(8102)
    cfg = Phase8ModelConfig(
        d=3,
        lag=2,
        layers=1,
        hidden=8,
        d_cond=2,
        d_state=2,
        d_conv=2,
        dtype="float64",
    )
    model = CoverageAlignedRawChainJRNGC(cfg).cpu().eval()
    x = rng.normal(scale=0.2, size=(3, 12)).astype(np.float64)
    target_u, output, source, raw_lag = 8, 1, 2, 1
    jac = total_raw_chain_at_target(model, x, target_u, h_max=target_u)
    autograd_value = float(jac[output, source, raw_lag - 1].detach())
    epsilon = 1e-6
    x_plus = x.copy()
    x_minus = x.copy()
    x_plus[source, target_u - raw_lag] += epsilon
    x_minus[source, target_u - raw_lag] -= epsilon
    with torch.no_grad():
        raw_plus = as_raw_bdt(x_plus, device=torch.device("cpu"), dtype=torch.float64)
        raw_minus = as_raw_bdt(x_minus, device=torch.device("cpu"), dtype=torch.float64)
        y_plus = float(model.predict_from_raw(raw_plus, [target_u])[0, output])
        y_minus = float(model.predict_from_raw(raw_minus, [target_u])[0, output])
    fd_value = (y_plus - y_minus) / (2.0 * epsilon)
    absolute_error = abs(fd_value - autograd_value)
    tolerance = FD_ATOL + FD_RTOL * max(abs(fd_value), abs(autograd_value))
    return {
        "passed": bool(absolute_error <= tolerance),
        "dtype": "float64",
        "target_u": target_u,
        "output_target": output,
        "source_variable": source,
        "raw_lag": raw_lag,
        "epsilon": epsilon,
        "finite_difference": fd_value,
        "autograd": autograd_value,
        "absolute_error": absolute_error,
        "tolerance": tolerance,
    }


def direct_indirect_chain_decomposition_audit() -> Dict[str, object]:
    torch.manual_seed(8111)
    rng = np.random.default_rng(8112)
    cfg = Phase8ModelConfig(
        d=3,
        lag=2,
        layers=1,
        hidden=8,
        d_cond=2,
        d_state=2,
        d_conv=2,
        dtype="float64",
    )
    model = CoverageAlignedRawChainJRNGC(cfg).cpu().eval()
    x = rng.normal(scale=0.25, size=(3, 12)).astype(np.float64)
    u, output = 9, 0

    raw_total = as_raw_bdt(x, device=torch.device("cpu"), dtype=torch.float64, require_grad=True)
    y_total = model.predict_from_raw(raw_total, [u])[0, output]
    g_total = torch.autograd.grad(y_total, raw_total, retain_graph=False)[0]

    raw_direct = as_raw_bdt(x, device=torch.device("cpu"), dtype=torch.float64, require_grad=True)
    with torch.no_grad():
        c_fixed = model.condition_sequence(raw_direct.detach())
    y_direct = model.predict_from_raw(raw_direct, [u], auxiliary_override=c_fixed)[0, output]
    g_direct = torch.autograd.grad(y_direct, raw_direct, retain_graph=False)[0]

    raw_indirect = as_raw_bdt(x, device=torch.device("cpu"), dtype=torch.float64, require_grad=True)
    c_indirect = model.condition_sequence(raw_indirect)
    x_detached = raw_indirect.detach()
    y_indirect = model.predict_from_raw(x_detached, [u], auxiliary_override=c_indirect)[0, output]
    g_indirect = torch.autograd.grad(y_indirect, raw_indirect, retain_graph=False)[0]

    difference = g_total - (g_direct + g_indirect)
    max_abs = float(torch.max(torch.abs(difference)))
    denom = float(torch.linalg.norm(g_total)) + EPS
    relative_l2 = float(torch.linalg.norm(difference)) / denom
    passed = max_abs <= 1e-10 or relative_l2 <= 1e-8
    return {
        "passed": bool(passed),
        "dtype": "float64",
        "target_u": u,
        "output_target": output,
        "max_absolute_difference": max_abs,
        "relative_l2_difference": relative_l2,
        "absolute_tolerance": 1e-10,
        "relative_tolerance": 1e-8,
        "total_gradient_norm": float(torch.linalg.norm(g_total)),
        "direct_gradient_norm": float(torch.linalg.norm(g_direct)),
        "indirect_gradient_norm": float(torch.linalg.norm(g_indirect)),
    }


def _flatten_parameter_gradients(
    model: nn.Module,
    gradients: Sequence[Optional[torch.Tensor]],
) -> Tuple[torch.Tensor, Dict[str, float]]:
    pieces = []
    group_sq = {"predictor": 0.0, "preprocessor": 0.0}
    for (name, parameter), gradient in zip(model.named_parameters(), gradients):
        value = torch.zeros_like(parameter) if gradient is None else gradient
        pieces.append(value.reshape(-1))
        group = "preprocessor" if name.startswith("preprocessor.") else "predictor"
        if "weight_head" not in name:
            group_sq[group] += float(torch.sum(value.detach() ** 2))
    return torch.cat(pieces), {key: math.sqrt(value) for key, value in group_sq.items()}


def estimator_exact_reference_audit(draw_count: int = 512) -> Dict[str, object]:
    torch.manual_seed(8121)
    rng = np.random.default_rng(8122)
    cfg = Phase8ModelConfig(
        d=3,
        lag=1,
        layers=1,
        hidden=8,
        d_cond=2,
        d_state=2,
        d_conv=2,
        dtype="float64",
    )
    model = CoverageAlignedRawChainJRNGC(cfg).cpu().train()
    x = rng.normal(scale=0.2, size=(3, 12)).astype(np.float64)
    raw = as_raw_bdt(x, device=torch.device("cpu"), dtype=torch.float64, require_grad=True)
    parameters = tuple(model.parameters())

    exact = exact_lag_balanced_objective(model, raw, create_graph=True)
    exact_gradients = torch.autograd.grad(exact, parameters, allow_unused=True)
    exact_vector, exact_groups = _flatten_parameter_gradients(model, exact_gradients)

    schedule = build_balanced_lag_schedule(T=12, lag=1, d_out=3, max_iter=draw_count, seed=8123)
    estimated_values: List[float] = []
    estimated_vector = torch.zeros_like(exact_vector)
    estimated_groups_accum = {"predictor": 0.0, "preprocessor": 0.0}
    for entry in schedule:
        raw_draw = as_raw_bdt(x, device=torch.device("cpu"), dtype=torch.float64, require_grad=True)
        estimate = sampled_lag_balanced_penalty(model, raw_draw, entry, create_graph=True)
        gradients = torch.autograd.grad(estimate, parameters, allow_unused=True)
        vector, groups = _flatten_parameter_gradients(model, gradients)
        estimated_values.append(float(estimate.detach()))
        estimated_vector += vector.detach()
        for key in estimated_groups_accum:
            estimated_groups_accum[key] += groups[key]
    estimated_vector /= draw_count
    estimated_value = float(np.mean(estimated_values))
    exact_value = float(exact.detach())
    relative_error = abs(estimated_value - exact_value) / max(abs(exact_value), EPS)
    cosine = float(torch.nn.functional.cosine_similarity(
        exact_vector.detach().reshape(1, -1),
        estimated_vector.reshape(1, -1),
    )[0])
    finite = bool(
        np.isfinite(exact_value)
        and np.isfinite(estimated_value)
        and torch.isfinite(exact_vector).all()
        and torch.isfinite(estimated_vector).all()
    )
    nonzero = bool(
        exact_groups["predictor"] > 0
        and exact_groups["preprocessor"] > 0
        and torch.linalg.norm(estimated_vector) > 0
    )
    return {
        "passed": bool(finite and nonzero and relative_error <= 0.05 and cosine >= 0.95),
        "fixture": {
            "d": 3,
            "T": 12,
            "K": 1,
            "H_max": 11,
            "d_cond": 2,
            "layers": 1,
            "hidden": 8,
            "model_seed": 8121,
            "data_seed": 8122,
            "schedule_seed": 8123,
            "draw_count": draw_count,
        },
        "exact_objective": exact_value,
        "estimated_objective": estimated_value,
        "relative_objective_error": relative_error,
        "relative_objective_error_max": 0.05,
        "parameter_gradient_cosine": cosine,
        "parameter_gradient_cosine_min": 0.95,
        "exact_group_gradient_norms": exact_groups,
        "mean_draw_group_gradient_norms": {
            key: value / draw_count for key, value in estimated_groups_accum.items()
        },
        "finite": finite,
        "nonzero_predictor_and_preprocessor": nonzero,
        "schedule_sha256": schedule_sha256(schedule),
    }


def deterministic_schedule_audit() -> Dict[str, object]:
    a = build_balanced_lag_schedule(T=500, lag=1, d_out=8, max_iter=2000, seed=32001)
    b = build_balanced_lag_schedule(T=500, lag=1, d_out=8, max_iter=2000, seed=32001)
    c = build_balanced_lag_schedule(T=500, lag=1, d_out=8, max_iter=2000, seed=32002)
    lag_counts = {h: 0 for h in range(1, 500)}
    output_counts = {j: 0 for j in range(8)}
    eligible = True
    distinct = True
    for entry in a:
        lags = entry["lags"]
        windows = entry["eligible_windows"]
        outputs = entry["output_targets"]
        distinct = distinct and len(set(lags)) == 2 and len(set(outputs)) == 2
        for h, u in zip(lags, windows):
            lag_counts[int(h)] += 1
            eligible = eligible and int(u) >= int(h) and int(u) >= 1 and int(u) < 500
        for output in outputs:
            output_counts[int(output)] += 1
    factor = 499 / (2 * 1 * 2 * 8 * 1)
    passed = (
        a == b
        and schedule_sha256(a) != schedule_sha256(c)
        and set(lag_counts.values()).issubset({8, 9})
        and set(output_counts.values()) == {500}
        and eligible
        and distinct
        and abs(factor - 499 / 32) < 1e-15
    )
    return {
        "passed": bool(passed),
        "schedule_sha256": schedule_sha256(a),
        "same_seed_reproduced": a == b,
        "different_seed_changes_hash": schedule_sha256(a) != schedule_sha256(c),
        "lag_count_min": min(lag_counts.values()),
        "lag_count_max": max(lag_counts.values()),
        "output_counts": output_counts,
        "within_step_distinct": distinct,
        "all_windows_eligible": eligible,
        "sampled_sum_factor": factor,
        "expected_sampled_sum_factor": 499 / 32,
    }


def _legacy_direct_prediction_target(adapter: LegacyComparatorAdapter, x_full) -> Tuple[torch.Tensor, torch.Tensor]:
    if isinstance(adapter, LegacyConcatXOnlyAdapter):
        windows, _ = adapter.model.preprocess_and_windowing(x_full)
        prediction = adapter.model(windows[:, :, :adapter.lag].flatten(start_dim=1))
        target = windows[:, :adapter.d, -1]
    else:
        windows = adapter.model.make_windows(x_full)
        prediction = adapter.model(windows[:, :, :adapter.lag])
        target = windows[:, :, -1]
    return prediction, target


def _max_parameter_gradient_difference(
    direct_model: nn.Module,
    adapted_model: nn.Module,
    x_full,
) -> Tuple[float, float, bool]:
    direct_model.zero_grad(set_to_none=True)
    adapted_model.zero_grad(set_to_none=True)
    direct_loss = direct_model.compute_loss(x_full)
    adapted_loss = adapted_model.compute_loss(x_full)
    direct_loss.backward()
    adapted_loss.backward()
    direct_named = dict(direct_model.named_parameters())
    adapted_named = dict(adapted_model.named_parameters())
    names_equal = list(direct_named) == list(adapted_named)
    max_abs = 0.0
    diff_sq = 0.0
    base_sq = 0.0
    for name in direct_named:
        left = direct_named[name].grad
        right = adapted_named[name].grad
        if left is None or right is None:
            names_equal = names_equal and left is None and right is None
            continue
        delta = left - right
        max_abs = max(max_abs, float(torch.max(torch.abs(delta))))
        diff_sq += float(torch.sum(delta ** 2))
        base_sq += float(torch.sum(left ** 2))
    relative_l2 = math.sqrt(diff_sq) / (math.sqrt(base_sq) + EPS)
    return max_abs, relative_l2, names_equal


def comparator_parity_audit() -> Dict[str, object]:
    rng = np.random.default_rng(8132)
    x = rng.normal(scale=0.2, size=(3, 10)).astype(np.float32)
    cfg = Phase8ModelConfig(d=3, lag=1, layers=1, hidden=5, d_cond=2, d_state=2)
    builders = {
        "baseline": lambda: make_legacy_baseline(cfg),
        "concat_x_only": lambda: make_legacy_concat(cfg),
        "full_aux_equal": lambda: make_legacy_full_aux(cfg, "equal"),
        "full_aux_lc10": lambda: make_legacy_full_aux(cfg, "lc10"),
    }
    reports: Dict[str, object] = {}
    all_passed = True
    for offset, (name, builder) in enumerate(builders.items()):
        torch.manual_seed(8131 + offset)
        adapter = builder()
        adapter.model.cpu().eval()
        direct_pred, direct_target = _legacy_direct_prediction_target(adapter, x)
        adapted_pred = adapter.predictions(x)
        adapted_target = adapter.raw_targets(x)
        prediction_diff = float(torch.max(torch.abs(direct_pred - adapted_pred)))
        targets_exact = bool(torch.equal(direct_target, adapted_target))
        direct_mse = torch.mean((direct_pred - direct_target) ** 2)
        components = adapter.loss_components(x)
        mse_diff = float(torch.abs(direct_mse - components["fixed_target_prediction_mse"]))
        direct_total = adapter.model.compute_loss(x)
        objective_diff = float(torch.abs(direct_total - components["total_regularized_objective"]))
        direct_penalty = direct_total - direct_mse
        penalty_diff = float(torch.abs(direct_penalty - components["jacobian_penalty"]))
        direct_score = np.asarray(adapter.model.get_gc_matrix(x), dtype=np.float64)
        direct_score = direct_score if direct_score.ndim == 2 else np.max(direct_score, axis=2)
        adapted_score = adapter.partial_nominal_score(x)
        score_diff = float(np.max(np.abs(direct_score - adapted_score)))

        torch.manual_seed(9000 + offset)
        direct_copy = builder().model.cpu().train()
        torch.manual_seed(9000 + offset)
        adapted_copy = builder().model.cpu().train()
        adapted_copy.load_state_dict(direct_copy.state_dict())
        grad_max, grad_rel, gradient_structure_equal = _max_parameter_gradient_difference(
            direct_copy,
            adapted_copy,
            x,
        )
        passed = (
            prediction_diff <= 1e-7 + 1e-6 * float(torch.max(torch.abs(direct_pred)))
            and targets_exact
            and mse_diff <= 1e-7 + 1e-6 * abs(float(direct_mse))
            and penalty_diff <= 1e-7 + 1e-6 * abs(float(direct_penalty))
            and objective_diff <= 1e-7 + 1e-6 * abs(float(direct_total))
            and score_diff <= 1e-7
            and grad_max <= 1e-6
            and grad_rel <= 1e-5
            and gradient_structure_equal
        )
        all_passed = all_passed and passed
        reports[name] = {
            "passed": bool(passed),
            "forward_prediction_max_abs_diff": prediction_diff,
            "raw_targets_elementwise_exact": targets_exact,
            "pure_mse_abs_diff": mse_diff,
            "jacobian_penalty_abs_diff": penalty_diff,
            "total_objective_abs_diff": objective_diff,
            "legacy_partial_score_max_abs_diff": score_diff,
            "objective_gradient_max_abs_diff": grad_max,
            "objective_gradient_relative_l2_diff": grad_rel,
            "parameter_names_shapes_and_gradient_presence_equal": gradient_structure_equal,
            "dtype": "float32_native_legacy",
        }
    return {
        "passed": bool(all_passed),
        "float32_tolerances": {
            "value_atol": 1e-7,
            "value_rtol": 1e-6,
            "score_max_abs": 1e-7,
            "gradient_max_abs": 1e-6,
            "gradient_relative_l2": 1e-5,
        },
        "comparators": reports,
    }
