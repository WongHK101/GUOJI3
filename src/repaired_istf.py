"""P0.1 repaired ISTF models and raw-chain attribution helpers.

This module intentionally leaves legacy classes unchanged. The repaired
classes train on filtered history but raw targets, and use extended truncated
raw-chain Jacobians for both training penalties and evaluation scores.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from minimal_mamba import MambaBlock
from knowledge_metrics import adjacency_to_edge_set as canonical_adjacency_to_edge_set
from knowledge_metrics import topk_edges_exact as canonical_topk_edges_exact


EPS = 1e-12


class ResidualBlock(nn.Module):
    """Local copy of the JRNGC residual MLP block used by legacy pilot code."""

    def __init__(self, input_dim: int, hidden: int, output_dim: int, dropout: float):
        super().__init__()
        self.linear_1 = nn.Linear(input_dim, hidden)
        self.linear_2 = nn.Linear(hidden, output_dim)
        self.linear_res = nn.Linear(input_dim, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.layernorm = nn.LayerNorm(output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.linear_1(x)
        h = self.relu(h)
        h = self.linear_2(h)
        h = self.dropout(h)
        res = self.linear_res(x)
        return self.layernorm(h + res)


@dataclass(frozen=True)
class RepairedISTFConfig:
    d: int
    lag: int
    attribution_horizon: int = 32
    layers: int = 1
    hidden: int = 16
    dropout: float = 0.0
    jacobian_lam: float = 0.01
    identity_lam: float = 0.05
    residual_gain: float = 0.1
    depthwise_kernel_size: int = 3
    d_state: int = 4
    mamba_expand: int = 2
    mamba_d_conv: int = 4
    ema_alpha: float = 0.9
    dtype: str = "float32"


@dataclass(frozen=True)
class JacobianEstimatorConfig:
    attribution_horizon: int = 32
    sampled_windows_per_step: int = 2
    sampled_targets_per_step: int = 2
    jacobian_seed: int = 7101
    normalization: str = "sum_abs_over_sampled_windows_targets_sources_horizon_div_by_W_targets_sources_nominal_lag"


@dataclass(frozen=True)
class EvaluationWindowConfig:
    score_window_seed: int = 9103
    score_max_windows: int = 32
    full_window_audit: bool = True


def _torch_dtype(name: str) -> torch.dtype:
    if name == "float64":
        return torch.float64
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def _as_tensor_x(x_full, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    x = torch.as_tensor(x_full, device=device, dtype=dtype)
    if x.dim() == 2:
        x = x.unsqueeze(0)
    if x.dim() != 3:
        raise ValueError(f"x_full must have shape (d,T) or (B,d,T), got {tuple(x.shape)}")
    return x


def eligible_target_indices(T: int, lag: int, attribution_horizon: int) -> np.ndarray:
    start = max(lag, attribution_horizon)
    if T <= start:
        raise ValueError(f"T={T} leaves no eligible targets for lag={lag}, H={attribution_horizon}")
    return np.arange(start, T, dtype=np.int64)


def deterministic_sample_indices(
    eligible: Sequence[int],
    n: int,
    seed: int,
    require_min_target: Optional[int] = None,
) -> np.ndarray:
    arr = np.asarray(list(eligible), dtype=np.int64)
    if require_min_target is not None:
        arr = arr[arr >= require_min_target]
    if arr.size == 0:
        raise ValueError("No eligible target indices available")
    if n <= 0 or n >= arr.size:
        return arr.copy()
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(arr, size=n, replace=False).astype(np.int64))


def make_cyclic_schedule(
    target_indices: Sequence[int],
    d: int,
    max_iter: int,
    windows_per_step: int = 2,
    targets_per_step: int = 2,
    seed: int = 7101,
) -> List[Dict[str, List[int]]]:
    windows = np.asarray(list(target_indices), dtype=np.int64)
    if windows.size == 0:
        raise ValueError("target_indices cannot be empty")
    rng = np.random.default_rng(seed)
    window_order = windows.copy()
    target_order = np.arange(d, dtype=np.int64)
    rng.shuffle(window_order)
    rng.shuffle(target_order)
    schedule: List[Dict[str, List[int]]] = []
    for it in range(max_iter):
        w = [int(window_order[(it * windows_per_step + k) % len(window_order)])
             for k in range(windows_per_step)]
        t = [int(target_order[(it * targets_per_step + k) % len(target_order)])
             for k in range(targets_per_step)]
        schedule.append({"iter": int(it), "target_indices": w, "output_targets": t})
    return schedule


def schedule_hash(schedule: Sequence[Dict[str, List[int]]]) -> str:
    payload = json.dumps(schedule, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class CPDepthwiseCausalFilter(nn.Module):
    """Strict coordinate-preserving causal depthwise filter."""

    def __init__(self, d: int, kernel_size: int = 3, residual_gain: float = 0.1,
                 dtype: torch.dtype = torch.float32):
        super().__init__()
        self.d = d
        self.kernel_size = kernel_size
        self.residual_gain = float(residual_gain)
        self.conv = nn.Conv1d(
            d,
            d,
            kernel_size=kernel_size,
            dilation=1,
            groups=d,
            bias=False,
            dtype=dtype,
        )
        nn.init.zeros_(self.conv.weight)

    @property
    def receptive_field(self) -> int:
        return self.kernel_size

    def forward(self, x_t: torch.Tensor) -> torch.Tensor:
        # x_t: (B,T,d). Strict left-causal padding, no cross-channel projection.
        residual = x_t
        x_c = x_t.transpose(1, 2)
        x_c = F.pad(x_c, (self.kernel_size - 1, 0))
        y = self.conv(x_c).transpose(1, 2)
        return residual + self.residual_gain * y


class RepairedBaseJRNGC(nn.Module):
    method_status = "repaired"
    training_target_domain = "raw"
    training_score_coordinate = "raw-chain"
    evaluation_loss_domain = "raw"
    filter_receptive_field = None

    def __init__(self, cfg: RepairedISTFConfig):
        super().__init__()
        if cfg.attribution_horizon < cfg.lag:
            raise ValueError("attribution_horizon must be >= lag")
        self.cfg = cfg
        self.d = cfg.d
        self.lag = cfg.lag
        self.attribution_horizon = cfg.attribution_horizon
        self.jacobian_lam = cfg.jacobian_lam
        self.identity_lam = cfg.identity_lam
        dtype = _torch_dtype(cfg.dtype)
        self.inputgate = nn.Linear(cfg.d * cfg.lag, cfg.hidden, dtype=dtype)
        self.outputgate = nn.Linear(cfg.hidden, cfg.d, dtype=dtype)
        self.encoders = nn.ModuleList([
            ResidualBlock(cfg.hidden, cfg.hidden, cfg.hidden, cfg.dropout)
            for _ in range(cfg.layers)
        ])
        # ResidualBlock is constructed in default dtype; align with requested dtype.
        self.to(dtype=dtype)

    def forward(self, x_history: torch.Tensor) -> torch.Tensor:
        h = x_history.flatten(start_dim=1).to(next(self.parameters()).dtype)
        h = self.inputgate(h)
        for net in self.encoders:
            h = net(h)
        return self.outputgate(h)

    def _identity_filter(self, raw_t: torch.Tensor) -> torch.Tensor:
        return raw_t

    def filter_sequence(self, raw_bdt: torch.Tensor) -> torch.Tensor:
        raw_t = raw_bdt.transpose(1, 2)
        return self._identity_filter(raw_t)

    def make_histories(
        self,
        x_full,
        target_indices: Optional[Sequence[int]] = None,
        require_grad: bool = False,
    ) -> Dict[str, torch.Tensor]:
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        raw = _as_tensor_x(x_full, device, dtype)
        if require_grad:
            raw = raw.detach().clone().requires_grad_(True)
        T = raw.shape[2]
        if target_indices is None:
            target_indices = eligible_target_indices(T, self.lag, self.attribution_horizon)
        idx = torch.as_tensor(list(target_indices), device=device, dtype=torch.long)
        filt_t = self.filter_sequence(raw)
        raw_t = raw.transpose(1, 2)
        filt_hist = []
        raw_hist = []
        raw_target = []
        filt_target = []
        for u in idx.tolist():
            filt_hist.append(filt_t[0, u - self.lag:u, :].transpose(0, 1))
            raw_hist.append(raw_t[0, u - self.lag:u, :].transpose(0, 1))
            raw_target.append(raw_t[0, u, :])
            filt_target.append(filt_t[0, u, :])
        return {
            "raw_bdt": raw,
            "raw_t": raw_t,
            "filtered_t": filt_t,
            "target_indices": idx,
            "filtered_history": torch.stack(filt_hist, dim=0),
            "raw_history": torch.stack(raw_hist, dim=0),
            "raw_target": torch.stack(raw_target, dim=0),
            "filtered_target": torch.stack(filt_target, dim=0),
        }

    def prediction_losses(self, x_full, target_indices: Optional[Sequence[int]] = None) -> Dict[str, torch.Tensor]:
        batch = self.make_histories(x_full, target_indices=target_indices, require_grad=False)
        pred = self(batch["filtered_history"])
        raw_loss = torch.mean((pred - batch["raw_target"]) ** 2)
        filt_loss = torch.mean((pred - batch["filtered_target"]) ** 2)
        return {"raw": raw_loss, "filtered": filt_loss}

    def identity_penalty(self, x_full, target_indices: Optional[Sequence[int]] = None) -> torch.Tensor:
        if self.identity_lam == 0:
            return torch.zeros((), device=next(self.parameters()).device, dtype=next(self.parameters()).dtype)
        batch = self.make_histories(x_full, target_indices=target_indices, require_grad=False)
        diff = torch.mean((batch["filtered_history"] - batch["raw_history"]) ** 2)
        denom = torch.mean(batch["raw_history"] ** 2) + EPS
        return self.identity_lam * diff / denom

    def compute_loss_components(
        self,
        x_full,
        schedule_entry: Optional[Dict[str, List[int]]] = None,
        target_indices: Optional[Sequence[int]] = None,
    ) -> Dict[str, torch.Tensor]:
        pred_losses = self.prediction_losses(x_full, target_indices=target_indices)
        if schedule_entry is None:
            T = np.asarray(x_full).shape[1]
            eligible = eligible_target_indices(T, self.lag, self.attribution_horizon)
            schedule_entry = {"target_indices": [int(eligible[0])], "output_targets": [0]}
        jac = raw_chain_jacobian_penalty(
            self,
            x_full,
            schedule_entry["target_indices"],
            schedule_entry["output_targets"],
            create_graph=True,
        )
        ident = self.identity_penalty(x_full, target_indices=target_indices)
        total = pred_losses["raw"] + self.jacobian_lam * jac + ident
        return {
            "train_prediction_loss": pred_losses["raw"],
            "eval_filtered_prediction_loss": pred_losses["filtered"].detach(),
            "raw_chain_jacobian_penalty_unweighted": jac,
            "raw_chain_jacobian_penalty": self.jacobian_lam * jac,
            "identity_penalty": ident,
            "total_training_objective": total,
        }

    def compute_loss(self, x_full, return_components: bool = False, schedule_entry=None):
        comp = self.compute_loss_components(x_full, schedule_entry=schedule_entry)
        return comp if return_components else comp["total_training_objective"]

    def filter_diagnostics(self, x_full, target_indices: Optional[Sequence[int]] = None) -> Dict[str, float]:
        batch = self.make_histories(x_full, target_indices=target_indices, require_grad=False)
        raw_h = batch["raw_history"].detach()
        filt_h = batch["filtered_history"].detach()
        diff = torch.mean((filt_h - raw_h) ** 2)
        denom = torch.mean(raw_h ** 2) + EPS
        out = {
            "identity_deviation": float((diff / denom).cpu()),
            "filtered_raw_variance_ratio": float((torch.var(filt_h) / (torch.var(raw_h) + EPS)).cpu()),
            "residual_gain": float(getattr(self, "residual_gain", 0.0)),
            "kernel_frobenius_norm": 0.0,
        }
        filt = getattr(self, "filter", None)
        if filt is not None and hasattr(filt, "conv"):
            out["kernel_frobenius_norm"] = float(torch.linalg.norm(filt.conv.weight.detach()).cpu())
        return out


class RawTargetBaselineJRNGC(RepairedBaseJRNGC):
    method_name = "raw_target_baseline"
    identity_lam = 0.0

    def __init__(self, cfg: RepairedISTFConfig):
        cfg = RepairedISTFConfig(**{**asdict(cfg), "identity_lam": 0.0})
        super().__init__(cfg)
        self.identity_lam = 0.0
        self.filter_receptive_field = 1


class CPDepthwiseISTFJRNGC(RepairedBaseJRNGC):
    method_name = "cp_depthwise_istf"

    def __init__(self, cfg: RepairedISTFConfig):
        super().__init__(cfg)
        dtype = _torch_dtype(cfg.dtype)
        self.residual_gain = cfg.residual_gain
        self.filter = CPDepthwiseCausalFilter(
            cfg.d,
            kernel_size=cfg.depthwise_kernel_size,
            residual_gain=cfg.residual_gain,
            dtype=dtype,
        )
        self.filter_receptive_field = cfg.depthwise_kernel_size

    def _identity_filter(self, raw_t: torch.Tensor) -> torch.Tensor:
        return self.filter(raw_t)


class RawChainMambaISTFJRNGC(RepairedBaseJRNGC):
    method_name = "raw_chain_mamba_istf"

    def __init__(self, cfg: RepairedISTFConfig):
        super().__init__(cfg)
        self.residual_gain = cfg.residual_gain
        self.filter = MambaBlock(
            d_model=cfg.d,
            d_state=cfg.d_state,
            d_conv=cfg.mamba_d_conv,
            expand=cfg.mamba_expand,
            residual_scale=cfg.residual_gain,
        )
        self.filter.to(dtype=_torch_dtype(cfg.dtype))
        self.filter_receptive_field = "prefix_scan_unbounded_truncated_for_attribution"

    def _identity_filter(self, raw_t: torch.Tensor) -> torch.Tensor:
        return self.filter(raw_t)


class FixedEMAJRNGC(RepairedBaseJRNGC):
    method_name = "fixed_ema_jrngc"
    method_status = "reference"

    def __init__(self, cfg: RepairedISTFConfig):
        cfg = RepairedISTFConfig(**{**asdict(cfg), "identity_lam": 0.0})
        super().__init__(cfg)
        self.alpha = float(cfg.ema_alpha)
        self.identity_lam = 0.0
        self.filter_receptive_field = "prefix_recursion_unbounded_truncated_for_attribution"

    def _identity_filter(self, raw_t: torch.Tensor) -> torch.Tensor:
        z = torch.zeros_like(raw_t)
        z[:, 0, :] = raw_t[:, 0, :]
        for t in range(1, raw_t.shape[1]):
            z[:, t, :] = self.alpha * z[:, t - 1, :] + (1.0 - self.alpha) * raw_t[:, t, :]
        return z


def instantiate_repaired_method(name: str, cfg: RepairedISTFConfig) -> RepairedBaseJRNGC:
    if name == "baseline":
        return RawTargetBaselineJRNGC(cfg)
    if name == "cp_depthwise":
        return CPDepthwiseISTFJRNGC(cfg)
    if name == "raw_chain_mamba":
        return RawChainMambaISTFJRNGC(cfg)
    if name == "fixed_ema":
        return FixedEMAJRNGC(cfg)
    raise ValueError(f"Unknown repaired method: {name}")


def _predict_from_raw_for_target(model: RepairedBaseJRNGC, raw_bdt: torch.Tensor, target_u: int) -> torch.Tensor:
    filt_t = model.filter_sequence(raw_bdt)
    hist = filt_t[0, target_u - model.lag:target_u, :].transpose(0, 1).unsqueeze(0)
    return model(hist)[0]


def raw_chain_jacobian_for_windows(
    model: RepairedBaseJRNGC,
    x_full,
    target_indices: Sequence[int],
    attribution_horizon: Optional[int] = None,
    create_graph: bool = False,
    output_targets: Optional[Sequence[int]] = None,
    full_prefix: bool = False,
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """Return per-window raw-chain Jacobians.

    For fixed H, returns tensor (W,target,source,H), ordered oldest->latest.
    For full_prefix=True, the tensor is padded to max prefix length and the
    unpadded per-window tensors are also returned.
    """
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    raw = _as_tensor_x(x_full, device, dtype).detach().clone().requires_grad_(True)
    H = attribution_horizon or model.attribution_horizon
    idx_list = [int(i) for i in target_indices]
    targets = list(range(model.d)) if output_targets is None else [int(t) for t in output_targets]
    per_window: List[torch.Tensor] = []
    max_len = max(idx_list) if full_prefix else H
    out = torch.zeros((len(idx_list), model.d, model.d, max_len), device=device, dtype=dtype)
    for w_pos, u in enumerate(idx_list):
        pred = _predict_from_raw_for_target(model, raw, u)
        local_len = u if full_prefix else H
        jac_u = torch.zeros((model.d, model.d, local_len), device=device, dtype=dtype)
        for target in targets:
            grad = torch.autograd.grad(
                pred[target],
                raw,
                retain_graph=True,
                create_graph=create_graph,
                allow_unused=False,
            )[0][0]
            if full_prefix:
                local = grad[:, :u]
            else:
                local = grad[:, u - H:u]
            jac_u[target] = local
        per_window.append(jac_u)
        out[w_pos, :, :, -local_len:] = jac_u
    return out, per_window


def raw_chain_jacobian_penalty(
    model: RepairedBaseJRNGC,
    x_full,
    target_indices: Sequence[int],
    output_targets: Sequence[int],
    create_graph: bool = True,
) -> torch.Tensor:
    jac, _ = raw_chain_jacobian_for_windows(
        model,
        x_full,
        target_indices=target_indices,
        attribution_horizon=model.attribution_horizon,
        create_graph=create_graph,
        output_targets=output_targets,
        full_prefix=False,
    )
    numer = torch.sum(torch.abs(jac[:, output_targets, :, :]))
    denom = max(1, len(target_indices)) * max(1, len(output_targets)) * model.d * model.lag
    return numer / denom


def aggregate_window_jacobians(jac_windows: torch.Tensor, lag: int) -> Dict[str, np.ndarray]:
    """absolute value -> mean over windows -> max over lag/horizon."""
    jbar = torch.mean(torch.abs(jac_windows), dim=0)
    score_nominal = torch.max(jbar[:, :, -lag:], dim=2).values
    score_full_h = torch.max(jbar, dim=2).values
    return {
        "j_bar": jbar.detach().cpu().numpy(),
        "score_nominal": score_nominal.detach().cpu().numpy(),
        "score_full_H": score_full_h.detach().cpu().numpy(),
    }


def filtered_coordinate_jacobian_for_windows(
    model: RepairedBaseJRNGC,
    x_full,
    target_indices: Sequence[int],
    create_graph: bool = False,
) -> torch.Tensor:
    batch = model.make_histories(x_full, target_indices=target_indices, require_grad=False)
    xh = batch["filtered_history"].detach().clone().requires_grad_(True)
    pred = model(xh)
    jac = torch.zeros((len(target_indices), model.d, model.d, model.lag), device=xh.device, dtype=xh.dtype)
    for target in range(model.d):
        grad = torch.autograd.grad(
            pred[:, target].sum(),
            xh,
            retain_graph=True,
            create_graph=create_graph,
        )[0]
        jac[:, target] = grad
    return jac


def adjacency_to_edge_set_local(adj_2d: np.ndarray, exclude_diag: bool = True) -> set[Tuple[int, int]]:
    """Return edges as (source, target), matching Phase 3/4 orientation."""
    arr = np.asarray(adj_2d)
    edges: set[Tuple[int, int]] = set()
    for target in range(arr.shape[0]):
        for source in range(arr.shape[1]):
            if exclude_diag and target == source:
                continue
            if arr[target, source] != 0:
                edges.add((source, target))
    return edges


def topk_edges_exact_local(scores_2d: np.ndarray, k: int, exclude_diag: bool = True) -> set[Tuple[int, int]]:
    """Deterministic exact top-k edges as (source, target).

    The score matrix convention is score[target, source]. Ties are broken by
    target then source so same-seed runs are byte-stable.
    """
    scores = np.asarray(scores_2d, dtype=np.float64)
    if scores.ndim != 2 or scores.shape[0] != scores.shape[1]:
        raise ValueError(f"scores_2d must be square, got {scores.shape}")
    d = scores.shape[0]
    candidates: List[Tuple[float, int, int]] = []
    for target in range(d):
        for source in range(d):
            if exclude_diag and target == source:
                continue
            candidates.append((float(scores[target, source]), target, source))
    if k <= 0 or not candidates:
        return set()
    k_eff = min(int(k), len(candidates))
    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
    return {(source, target) for _, target, source in candidates[:k_eff]}


def exact_topk_metrics(scores_2d: np.ndarray, gc_true) -> Dict[str, float]:
    gt = np.asarray(gc_true)
    if gt.ndim == 3:
        gt_2d = (gt.sum(axis=2) > 0).astype(np.int32)
    else:
        gt_2d = gt.astype(np.int32)
    true_edges = canonical_adjacency_to_edge_set(gt_2d, exclude_diag=True)
    pred_edges = canonical_topk_edges_exact(scores_2d, k=len(true_edges), exclude_diag=True)
    tp = len(true_edges & pred_edges)
    fp = len(pred_edges - true_edges)
    fn = len(true_edges - pred_edges)
    d = gt_2d.shape[0]
    tn = d * (d - 1) - tp - fp - fn
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, EPS)
    shd = fp + fn
    mcc_denom = math.sqrt(max((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn), 0))
    mcc = (tp * tn - fp * fn) / max(mcc_denom, EPS)
    return {
        "f1_exact_topk": float(f1),
        "precision_exact_topk": float(precision),
        "recall_exact_topk": float(recall),
        "shd_exact_topk": int(shd),
        "nshd_exact_topk": float(shd / max(len(true_edges), 1)),
        "mcc_exact_topk": float(mcc),
        "n_true_edges": int(len(true_edges)),
    }


def _offdiag_labels_scores(gt_2d: np.ndarray, scores_2d: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    gt = np.asarray(gt_2d).astype(np.int32)
    scores = np.asarray(scores_2d, dtype=np.float64)
    mask = ~np.eye(gt.shape[0], dtype=bool)
    return gt[mask].astype(np.int32), scores[mask].astype(np.float64)


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    sorted_vals = values[order]
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_vals[end] == sorted_vals[start]:
            end += 1
        avg_rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = avg_rank
        start = end
    return ranks


def _binary_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    n_pos = int(np.sum(y_true == 1))
    n_neg = int(np.sum(y_true == 0))
    if n_pos == 0 or n_neg == 0:
        return 0.5
    ranks = _average_ranks(y_score)
    rank_sum_pos = float(np.sum(ranks[y_true == 1]))
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _binary_auprc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    n_pos = int(np.sum(y_true == 1))
    if n_pos == 0:
        return 0.0
    order = np.argsort(-y_score, kind="mergesort")
    y_sorted = y_true[order]
    tp = np.cumsum(y_sorted == 1)
    fp = np.cumsum(y_sorted == 0)
    precision = tp / np.maximum(tp + fp, 1)
    return float(np.sum(precision[y_sorted == 1]) / n_pos)


def _best_threshold_f1(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if y_true.size == 0:
        return 0.0
    best = 0.0
    for thr in np.unique(y_score):
        pred = (y_score >= thr).astype(np.int32)
        tp = int(np.sum((pred == 1) & (y_true == 1)))
        fp = int(np.sum((pred == 1) & (y_true == 0)))
        fn = int(np.sum((pred == 0) & (y_true == 1)))
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, EPS)
        best = max(best, f1)
    return float(best)


def graph_recovery_metrics(scores_2d: np.ndarray, gc_true) -> Dict[str, float]:
    gt = np.asarray(gc_true)
    if gt.ndim == 3:
        gt_2d = (gt.sum(axis=2) > 0).astype(np.int32)
    else:
        gt_2d = gt.astype(np.int32)
    y_true, y_score = _offdiag_labels_scores(gt_2d, scores_2d)
    exact = exact_topk_metrics(scores_2d, gt_2d)
    out = {
        "auroc": _binary_auroc(y_true, y_score),
        "auprc": _binary_auprc(y_true, y_score),
        "f1_threshold": _best_threshold_f1(y_true, y_score),
        "shd_topk_legacy": float(exact["shd_exact_topk"]),
        "nshd_topk_legacy": float(exact["nshd_exact_topk"]),
        "mcc_topk_legacy": float(exact["mcc_exact_topk"]),
    }
    out.update(exact)
    return out


def offdiag_vector(scores: np.ndarray) -> np.ndarray:
    arr = np.asarray(scores, dtype=np.float64)
    mask = ~np.eye(arr.shape[0], dtype=bool)
    return arr[mask]


def pearson_or_reason(a: np.ndarray, b: np.ndarray) -> Dict[str, object]:
    va = offdiag_vector(a)
    vb = offdiag_vector(b)
    if np.std(va) < EPS or np.std(vb) < EPS:
        return {"value": None, "undefined_reason": "constant_offdiag_vector"}
    da = va - float(np.mean(va))
    db = vb - float(np.mean(vb))
    denom = math.sqrt(float(np.sum(da * da) * np.sum(db * db)))
    if denom < EPS:
        return {"value": None, "undefined_reason": "constant_offdiag_vector"}
    return {"value": float(np.sum(da * db) / denom), "undefined_reason": None}


def spearman_corr(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    va = offdiag_vector(a)
    vb = offdiag_vector(b)
    if va.size < 2:
        return None
    ra = np.argsort(np.argsort(va)).astype(np.float64)
    rb = np.argsort(np.argsort(vb)).astype(np.float64)
    if np.std(ra) < EPS or np.std(rb) < EPS:
        return None
    da = ra - float(np.mean(ra))
    db = rb - float(np.mean(rb))
    denom = math.sqrt(float(np.sum(da * da) * np.sum(db * db)))
    if denom < EPS:
        return None
    return float(np.sum(da * db) / denom)


def topk_jaccard(a: np.ndarray, b: np.ndarray, k: int) -> float:
    ea = canonical_topk_edges_exact(a, k=k, exclude_diag=True)
    eb = canonical_topk_edges_exact(b, k=k, exclude_diag=True)
    return len(ea & eb) / max(len(ea | eb), 1)


def temporal_horizon_mass_from_jac(jac_windows: torch.Tensor, lag: int) -> Dict[str, object]:
    mass = torch.sum(torch.abs(jac_windows), dim=(1, 2, 3))
    outside = torch.sum(torch.abs(jac_windows[:, :, :, :-lag]), dim=(1, 2, 3))
    vals = (outside / (mass + EPS)).detach().cpu().numpy()
    return {
        "mean": float(np.mean(vals)),
        "median": float(np.median(vals)),
        "p95": float(np.percentile(vals, 95)),
        "max": float(np.max(vals)),
        "per_window": [float(v) for v in vals],
    }


def filter_cross_variable_leakage(
    model: RepairedBaseJRNGC,
    x_full,
    target_indices: Sequence[int],
    attribution_horizon: Optional[int] = None,
) -> Dict[str, float]:
    if isinstance(model, RawTargetBaselineJRNGC):
        return {"cross_variable_leakage": 0.0, "total_l1_mass": 0.0, "cross_l1_mass": 0.0}
    H = attribution_horizon or model.attribution_horizon
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    raw = _as_tensor_x(x_full, device, dtype).detach().clone().requires_grad_(True)
    filt_t = model.filter_sequence(raw)
    total_mass = torch.zeros((), device=device, dtype=dtype)
    cross_mass = torch.zeros((), device=device, dtype=dtype)
    for u in [int(i) for i in target_indices]:
        for out_var in range(model.d):
            y = filt_t[0, u - model.lag:u, out_var].sum()
            grad = torch.autograd.grad(y, raw, retain_graph=True, create_graph=False)[0][0]
            local = grad[:, u - H:u]
            total_mass = total_mass + torch.sum(torch.abs(local))
            mask = torch.ones(model.d, dtype=torch.bool, device=device)
            mask[out_var] = False
            cross_mass = cross_mass + torch.sum(torch.abs(local[mask]))
    leakage = cross_mass / (total_mass + EPS)
    return {
        "cross_variable_leakage": float(leakage.detach().cpu()),
        "total_l1_mass": float(total_mass.detach().cpu()),
        "cross_l1_mass": float(cross_mass.detach().cpu()),
    }


def evaluate_repaired_model(
    model: RepairedBaseJRNGC,
    x_full,
    gc_true,
    target_indices: Sequence[int],
    attribution_horizon: Optional[int] = None,
    include_filtered_coordinate: bool = True,
) -> Dict[str, object]:
    H = attribution_horizon or model.attribution_horizon
    with torch.enable_grad():
        raw_jac, _ = raw_chain_jacobian_for_windows(
            model,
            x_full,
            target_indices=target_indices,
            attribution_horizon=H,
            create_graph=False,
            output_targets=None,
            full_prefix=False,
        )
    agg = aggregate_window_jacobians(raw_jac, model.lag)
    nominal_metrics = graph_recovery_metrics(agg["score_nominal"], gc_true)
    full_h_metrics = graph_recovery_metrics(agg["score_full_H"], gc_true)
    pred_losses = model.prediction_losses(x_full, target_indices=eligible_target_indices(
        np.asarray(x_full).shape[1], model.lag, model.attribution_horizon
    ))
    out: Dict[str, object] = {
        "raw_chain_j_bar": agg["j_bar"],
        "score_nominal": agg["score_nominal"],
        "score_full_H": agg["score_full_H"],
        "metrics_nominal": nominal_metrics,
        "metrics_full_H": full_h_metrics,
        "eval_raw_prediction_loss": float(pred_losses["raw"].detach().cpu()),
        "eval_filtered_prediction_loss": float(pred_losses["filtered"].detach().cpu()),
        "temporal_horizon_mass": temporal_horizon_mass_from_jac(raw_jac, model.lag),
        "cross_variable_leakage": filter_cross_variable_leakage(model, x_full, target_indices, H),
        "filter_diagnostics": model.filter_diagnostics(x_full, target_indices=target_indices),
    }
    if include_filtered_coordinate:
        filt = filtered_coordinate_jacobian_for_windows(model, x_full, target_indices, create_graph=False)
        fagg = aggregate_window_jacobians(filt, model.lag)
        out["filtered_coordinate_j_bar"] = fagg["j_bar"]
        out["filtered_coordinate_score_nominal"] = fagg["score_nominal"]
    return out


def horizon_sensitivity_audit(
    model: RepairedBaseJRNGC,
    x_full,
    gc_true,
    target_indices: Sequence[int],
    h_small: int = 32,
    h_large: int = 64,
) -> Dict[str, object]:
    idx = np.asarray(list(target_indices), dtype=np.int64)
    idx = idx[idx >= h_large]
    if idx.size == 0:
        raise ValueError("Horizon sensitivity requires target indices >= h_large")
    small = aggregate_window_jacobians(
        raw_chain_jacobian_for_windows(model, x_full, idx, h_small, create_graph=False)[0],
        model.lag,
    )
    large = aggregate_window_jacobians(
        raw_chain_jacobian_for_windows(model, x_full, idx, h_large, create_graph=False)[0],
        model.lag,
    )
    full_jac, per_full = raw_chain_jacobian_for_windows(
        model, x_full, idx, h_large, create_graph=False, full_prefix=True
    )
    full_agg = aggregate_window_jacobians(full_jac, model.lag)
    omitted_vals = []
    for u, jac_u in zip(idx.tolist(), per_full):
        early = jac_u[:, :, :max(0, u - h_small)]
        omitted = float(torch.sum(torch.abs(early)).detach().cpu())
        total = float(torch.sum(torch.abs(jac_u)).detach().cpu())
        omitted_vals.append(omitted / (total + EPS))
    k = int(np.sum(np.asarray(gc_true).sum(axis=2) > 0)) if np.asarray(gc_true).ndim == 3 else int(np.sum(gc_true))
    return {
        "target_indices": [int(i) for i in idx],
        "H32_vs_H64": _score_comparison(small["score_nominal"], large["score_nominal"], k),
        "H32_vs_full_prefix": _score_comparison(small["score_nominal"], full_agg["score_nominal"], k),
        "omitted_gradient_mass": {
            "mean": float(np.mean(omitted_vals)),
            "median": float(np.median(omitted_vals)),
            "p95": float(np.percentile(omitted_vals, 95)),
            "max": float(np.max(omitted_vals)),
            "per_window": [float(v) for v in omitted_vals],
        },
    }


def _score_comparison(a: np.ndarray, b: np.ndarray, k: int) -> Dict[str, object]:
    p = pearson_or_reason(a, b)
    return {
        "pearson": p,
        "spearman": spearman_corr(a, b),
        "topk_jaccard": topk_jaccard(a, b, k=k),
        "max_abs_diff": float(np.max(np.abs(a - b))),
        "mean_abs_diff": float(np.mean(np.abs(a - b))),
    }


def compare_sampled_vs_full(
    sampled_score: np.ndarray,
    full_score: np.ndarray,
    gc_true,
) -> Dict[str, object]:
    gt = np.asarray(gc_true)
    k = int(np.sum(gt.sum(axis=2) > 0)) if gt.ndim == 3 else int(np.sum(gt))
    sampled_metrics = graph_recovery_metrics(sampled_score, gc_true)
    full_metrics = graph_recovery_metrics(full_score, gc_true)
    return {
        "score_comparison": _score_comparison(sampled_score, full_score, k),
        "sampled_auroc": sampled_metrics["auroc"],
        "full_auroc": full_metrics["auroc"],
        "auroc_abs_diff": abs(sampled_metrics["auroc"] - full_metrics["auroc"]),
    }


def model_metadata(model: RepairedBaseJRNGC) -> Dict[str, object]:
    return {
        "method_name": getattr(model, "method_name", type(model).__name__),
        "method_status": getattr(model, "method_status", "unknown"),
        "training_target_domain": getattr(model, "training_target_domain", "raw"),
        "training_score_coordinate": getattr(model, "training_score_coordinate", "raw-chain"),
        "evaluation_loss_domain": getattr(model, "evaluation_loss_domain", "raw"),
        "filter_receptive_field": getattr(model, "filter_receptive_field", None),
        "config": asdict(model.cfg),
        "parameter_count": int(sum(p.numel() for p in model.parameters())),
    }


def canonical_baseline_penalty(
    model: RawTargetBaselineJRNGC,
    x_full,
    target_indices: Sequence[int],
    output_targets: Sequence[int],
) -> torch.Tensor:
    batch = model.make_histories(x_full, target_indices=target_indices, require_grad=False)
    xh = batch["raw_history"].detach().clone().requires_grad_(True)
    pred = model(xh)
    total = torch.zeros((), device=xh.device, dtype=xh.dtype)
    for t in output_targets:
        grad = torch.autograd.grad(pred[:, int(t)].sum(), xh, retain_graph=True, create_graph=False)[0]
        total = total + torch.sum(torch.abs(grad))
    denom = max(1, len(target_indices)) * max(1, len(output_targets)) * model.d * model.lag
    return total / denom


def _canonical_baseline_jacobian_for_targets(
    canonical_model: nn.Module,
    x_full,
    lag: int,
    target_indices: Sequence[int],
    output_targets: Optional[Sequence[int]] = None,
) -> torch.Tensor:
    windows = canonical_model.make_windows(x_full)
    win_idx = torch.as_tensor([int(u) - lag for u in target_indices], dtype=torch.long,
                              device=next(canonical_model.parameters()).device)
    x_input = windows[win_idx, :, :lag].detach().clone().requires_grad_(True)
    pred = canonical_model(x_input)
    d = x_input.shape[1]
    targets = range(d) if output_targets is None else [int(t) for t in output_targets]
    jac = torch.zeros((len(target_indices), d, d, lag), device=x_input.device, dtype=x_input.dtype)
    for target in targets:
        grad = torch.autograd.grad(
            pred[:, target].sum(),
            x_input,
            retain_graph=True,
            create_graph=False,
        )[0]
        jac[:, target] = grad
    return jac


def canonical_baseline_equivalence_audit(
    repaired_model: RawTargetBaselineJRNGC,
    x_full,
    target_indices: Sequence[int],
    output_targets: Sequence[int],
    gc_true=None,
    tolerance: float = 1e-6,
) -> Dict[str, object]:
    """Compare repaired baseline with legacy canonical BaselineJRNGC."""
    from mamba_jrngc_pilot import BaselineJRNGC as CanonicalBaselineJRNGC

    if not isinstance(repaired_model, RawTargetBaselineJRNGC):
        raise TypeError("canonical_baseline_equivalence_audit requires RawTargetBaselineJRNGC")
    canonical = CanonicalBaselineJRNGC(
        repaired_model.d,
        repaired_model.lag,
        layers=repaired_model.cfg.layers,
        hidden=repaired_model.cfg.hidden,
        dropout=repaired_model.cfg.dropout,
        jacobian_lam=repaired_model.cfg.jacobian_lam,
    )
    canonical.to(device=next(repaired_model.parameters()).device, dtype=next(repaired_model.parameters()).dtype)
    missing, unexpected = canonical.load_state_dict(repaired_model.state_dict(), strict=False)

    batch = repaired_model.make_histories(x_full, target_indices=target_indices, require_grad=False)
    repaired_pred = repaired_model(batch["raw_history"]).detach()
    windows = canonical.make_windows(x_full)
    win_idx = torch.as_tensor([int(u) - repaired_model.lag for u in target_indices],
                              dtype=torch.long, device=next(repaired_model.parameters()).device)
    canonical_hist = windows[win_idx, :, :repaired_model.lag]
    canonical_target = windows[win_idx, :, -1]
    canonical_pred = canonical(canonical_hist).detach()

    repaired_loss = torch.mean((repaired_pred - batch["raw_target"]) ** 2)
    canonical_loss = torch.mean((canonical_pred - canonical_target) ** 2)
    repaired_jac, _ = raw_chain_jacobian_for_windows(
        repaired_model,
        x_full,
        target_indices=target_indices,
        attribution_horizon=repaired_model.lag,
        create_graph=False,
        output_targets=output_targets,
    )
    canonical_jac = _canonical_baseline_jacobian_for_targets(
        canonical,
        x_full,
        repaired_model.lag,
        target_indices=target_indices,
        output_targets=output_targets,
    )
    repaired_penalty = torch.sum(torch.abs(repaired_jac[:, output_targets])) / (
        max(1, len(target_indices)) * max(1, len(output_targets)) * repaired_model.d * repaired_model.lag
    )
    canonical_penalty = torch.sum(torch.abs(canonical_jac[:, output_targets])) / (
        max(1, len(target_indices)) * max(1, len(output_targets)) * repaired_model.d * repaired_model.lag
    )

    all_targets = np.arange(repaired_model.lag, np.asarray(x_full).shape[1], dtype=np.int64)
    repaired_all_jac, _ = raw_chain_jacobian_for_windows(
        repaired_model,
        x_full,
        target_indices=all_targets,
        attribution_horizon=repaired_model.lag,
        create_graph=False,
        output_targets=None,
    )
    repaired_gc = torch.mean(torch.abs(repaired_all_jac), dim=0).detach().cpu().numpy()
    canonical_gc = canonical.get_gc_matrix(x_full)
    repaired_summary = np.max(np.abs(repaired_gc), axis=2)
    canonical_summary = np.max(np.abs(canonical_gc), axis=2)
    if gc_true is not None:
        gt = np.asarray(gc_true)
        gt_2d = (gt.sum(axis=2) > 0).astype(np.int32) if gt.ndim == 3 else gt.astype(np.int32)
        k = len(canonical_adjacency_to_edge_set(gt_2d, exclude_diag=True))
    else:
        k = max(1, repaired_model.d)
    repaired_edges = canonical_topk_edges_exact(repaired_summary, k=k, exclude_diag=True)
    canonical_edges = canonical_topk_edges_exact(canonical_summary, k=k, exclude_diag=True)

    diffs = {
        "raw_history_max_abs_diff": float(torch.max(torch.abs(batch["raw_history"] - canonical_hist)).detach().cpu()),
        "raw_target_max_abs_diff": float(torch.max(torch.abs(batch["raw_target"] - canonical_target)).detach().cpu()),
        "prediction_max_abs_diff": float(torch.max(torch.abs(repaired_pred - canonical_pred)).detach().cpu()),
        "prediction_loss_abs_diff": float(torch.abs(repaired_loss - canonical_loss).detach().cpu()),
        "jacobian_tensor_max_abs_diff": float(torch.max(torch.abs(repaired_jac - canonical_jac)).detach().cpu()),
        "jacobian_penalty_abs_diff": float(torch.abs(repaired_penalty - canonical_penalty).detach().cpu()),
        "summary_gc_max_abs_diff": float(np.max(np.abs(repaired_summary - canonical_summary))),
    }
    passed = all(v < tolerance for v in diffs.values()) and repaired_edges == canonical_edges
    return {
        "passed": bool(passed),
        "tolerance": float(tolerance),
        "state_dict_missing_keys": list(missing),
        "state_dict_unexpected_keys": list(unexpected),
        "target_indices": [int(i) for i in target_indices],
        "output_targets": [int(i) for i in output_targets],
        "diffs": diffs,
        "exact_topk_edges_equal": bool(repaired_edges == canonical_edges),
        "exact_topk_edges_repaired": sorted(list(repaired_edges)),
        "exact_topk_edges_canonical": sorted(list(canonical_edges)),
    }


def save_json(path: str, payload: object) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True, default=str)


def finite_values_ok(payload: object) -> bool:
    if isinstance(payload, dict):
        return all(finite_values_ok(v) for v in payload.values())
    if isinstance(payload, (list, tuple)):
        return all(finite_values_ok(v) for v in payload)
    if isinstance(payload, (float, int, np.floating, np.integer)):
        return np.isfinite(float(payload))
    return True


def legacy_file_hashes(project_root: str) -> Dict[str, str]:
    files = [
        os.path.join(project_root, "src", "mamba_jrngc_pilot.py"),
        os.path.join(project_root, "src", "minimal_mamba.py"),
    ]
    out = {}
    for path in files:
        with open(path, "rb") as f:
            out[os.path.relpath(path, project_root)] = hashlib.sha256(f.read()).hexdigest()
    return out


def now_timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")
