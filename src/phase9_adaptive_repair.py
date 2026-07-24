"""Development-only Phase 9 repair candidates.

The classes in this module do not replace any frozen Phase 7/8 implementation.
They target two bounded postmortem hypotheses:

* a coordinate-wise adaptive FIR should start from the useful fixed FIR3
  reference instead of the identity/zero-kernel point;
* the full-prefix coverage gradient should not be allowed to overwhelm or
  directly oppose the prediction gradient.

Neither mechanism is a formal result until it passes an independently frozen
confirmation protocol.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from phase8_coverage import CoverageAlignedRawChainJRNGC, as_raw_bdt
from repaired_istf import RepairedBaseJRNGC, RepairedISTFConfig


EPS = 1e-12


class ConstrainedAdaptiveFIRFilter(nn.Module):
    """Coordinate-wise convex FIR with a bounded learnable residual gate.

    For interior time points,

        z_i(t) = (1-g_i) x_i(t) + g_i sum_r w_{i,r} x_i(t-r),

    where ``w_i`` lies on the probability simplex and ``0 < g_i < gate_max``.
    Initial weights are uniform and the gate is initialized to ``init_gate``,
    reproducing the interior FixedFIR3 impulse response when kernel_size=3 and
    init_gate=0.1.
    """

    def __init__(
        self,
        d: int,
        *,
        kernel_size: int = 3,
        gate_max: float = 0.25,
        init_gate: float = 0.1,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        if kernel_size < 2:
            raise ValueError("kernel_size must be >= 2")
        if not 0.0 < init_gate < gate_max < 1.0:
            raise ValueError("Require 0 < init_gate < gate_max < 1")
        self.d = int(d)
        self.kernel_size = int(kernel_size)
        self.gate_max = float(gate_max)
        self.kernel_logits = nn.Parameter(torch.zeros(d, kernel_size, dtype=dtype))
        normalized_gate = init_gate / gate_max
        initial_logit = float(np.log(normalized_gate / (1.0 - normalized_gate)))
        self.gate_logits = nn.Parameter(torch.full((d,), initial_logit, dtype=dtype))

    @property
    def receptive_field(self) -> int:
        return self.kernel_size

    def simplex_weights(self) -> torch.Tensor:
        return torch.softmax(self.kernel_logits, dim=1)

    def gates(self) -> torch.Tensor:
        return self.gate_max * torch.sigmoid(self.gate_logits)

    def effective_impulse_response(self) -> torch.Tensor:
        weights = self.simplex_weights()
        gates = self.gates()
        impulse = gates[:, None] * weights
        impulse = impulse.clone()
        impulse[:, 0] = impulse[:, 0] + (1.0 - gates)
        return impulse

    def residual_kernel(self) -> torch.Tensor:
        impulse = self.effective_impulse_response()
        delta = torch.zeros_like(impulse)
        delta[:, 0] = 1.0
        return impulse - delta

    def forward(self, raw_t: torch.Tensor) -> torch.Tensor:
        if raw_t.ndim != 3 or raw_t.shape[2] != self.d:
            raise ValueError(f"Expected (batch,time,{self.d}), got {tuple(raw_t.shape)}")
        batch, time, _ = raw_t.shape
        shifted = []
        valid = []
        for raw_lag in range(self.kernel_size):
            if raw_lag == 0:
                shifted.append(raw_t)
                valid.append(torch.ones(time, device=raw_t.device, dtype=raw_t.dtype))
            else:
                pad = torch.zeros(
                    batch,
                    raw_lag,
                    self.d,
                    device=raw_t.device,
                    dtype=raw_t.dtype,
                )
                shifted.append(torch.cat([pad, raw_t[:, :-raw_lag, :]], dim=1))
                mask = torch.cat([
                    torch.zeros(raw_lag, device=raw_t.device, dtype=raw_t.dtype),
                    torch.ones(time - raw_lag, device=raw_t.device, dtype=raw_t.dtype),
                ])
                valid.append(mask)
        lagged = torch.stack(shifted, dim=3)  # (B,T,d,R)
        valid_mask = torch.stack(valid, dim=1).view(1, time, 1, self.kernel_size)
        weights = self.simplex_weights().view(1, 1, self.d, self.kernel_size)
        effective_weights = weights * valid_mask
        effective_weights = effective_weights / effective_weights.sum(dim=3, keepdim=True).clamp_min(EPS)
        smoothed = torch.sum(lagged * effective_weights, dim=3)
        gates = self.gates().view(1, 1, self.d)
        return (1.0 - gates) * raw_t + gates * smoothed


class ConstrainedAdaptiveFIRJRNGC(RepairedBaseJRNGC):
    """Raw-target, raw-chain JRNGC with a coordinate-preserving adaptive FIR."""

    method_name = "constrained_adaptive_fir"
    method_status = "phase9_development_candidate"

    def __init__(
        self,
        cfg: RepairedISTFConfig,
        *,
        kernel_size: int = 3,
        gate_max: float = 0.25,
        init_gate: float = 0.1,
    ):
        super().__init__(cfg)
        dtype = next(self.parameters()).dtype
        self.filter = ConstrainedAdaptiveFIRFilter(
            cfg.d,
            kernel_size=kernel_size,
            gate_max=gate_max,
            init_gate=init_gate,
            dtype=dtype,
        )
        self.filter_receptive_field = int(kernel_size)
        self.residual_gain = float(init_gate)

    def _identity_filter(self, raw_t: torch.Tensor) -> torch.Tensor:
        return self.filter(raw_t)

    def filter_diagnostics(
        self,
        x_full,
        target_indices: Optional[Sequence[int]] = None,
    ) -> Dict[str, object]:
        out: Dict[str, object] = super().filter_diagnostics(
            x_full,
            target_indices=target_indices,
        )
        with torch.no_grad():
            weights = self.filter.simplex_weights().detach().cpu().numpy()
            gates = self.filter.gates().detach().cpu().numpy()
            impulse = self.filter.effective_impulse_response().detach().cpu().numpy()
            residual = self.filter.residual_kernel().detach().cpu().numpy()
        out.update({
            "simplex_weights": weights.tolist(),
            "gates": gates.tolist(),
            "gate_mean": float(np.mean(gates)),
            "gate_min": float(np.min(gates)),
            "gate_max_observed": float(np.max(gates)),
            "effective_impulse_response": impulse.tolist(),
            "residual_kernel_frobenius_norm": float(np.linalg.norm(residual)),
            "kernel_frobenius_norm": float(np.linalg.norm(weights)),
        })
        return out


@dataclass(frozen=True)
class ProjectionDiagnostics:
    prediction_gradient_norm: float
    coverage_gradient_norm_before: float
    coverage_gradient_norm_after: float
    gradient_cosine_before: Optional[float]
    gradient_cosine_after: Optional[float]
    conflict_projected: bool
    norm_capped: bool
    max_coverage_to_prediction_ratio: float

    def as_dict(self) -> Dict[str, object]:
        return {
            "prediction_gradient_norm": self.prediction_gradient_norm,
            "coverage_gradient_norm_before": self.coverage_gradient_norm_before,
            "coverage_gradient_norm_after": self.coverage_gradient_norm_after,
            "gradient_cosine_before": self.gradient_cosine_before,
            "gradient_cosine_after": self.gradient_cosine_after,
            "conflict_projected": self.conflict_projected,
            "norm_capped": self.norm_capped,
            "max_coverage_to_prediction_ratio": self.max_coverage_to_prediction_ratio,
        }


def _replace_none(
    gradients: Sequence[Optional[torch.Tensor]],
    parameters: Sequence[nn.Parameter],
) -> List[torch.Tensor]:
    return [
        torch.zeros_like(parameter) if gradient is None else gradient
        for gradient, parameter in zip(gradients, parameters)
    ]


def _global_inner(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> torch.Tensor:
    return sum((torch.sum(x * y) for x, y in zip(a, b)), start=torch.zeros((), device=a[0].device))


def _global_norm(a: Sequence[torch.Tensor]) -> torch.Tensor:
    return torch.sqrt(_global_inner(a, a).clamp_min(0.0))


def project_coverage_gradient(
    prediction_gradients: Sequence[Optional[torch.Tensor]],
    coverage_gradients: Sequence[Optional[torch.Tensor]],
    parameters: Sequence[nn.Parameter],
    *,
    max_coverage_to_prediction_ratio: float = 1.0,
) -> Tuple[List[torch.Tensor], ProjectionDiagnostics]:
    """Project a conflicting coverage gradient and cap its global norm.

    The returned update gradient is ``g_prediction + g_coverage_adjusted``.
    Projection is performed only when the raw inner product is negative. The
    diagnostic is about the supplied gradient direction; Adam preconditioning
    means it is not advertised as an exact finite-step loss guarantee.
    """

    if max_coverage_to_prediction_ratio <= 0:
        raise ValueError("max_coverage_to_prediction_ratio must be positive")
    params = list(parameters)
    if not params:
        raise ValueError("No trainable parameters")
    g_pred = _replace_none(prediction_gradients, params)
    g_cov = _replace_none(coverage_gradients, params)
    pred_norm = _global_norm(g_pred)
    cov_norm_before = _global_norm(g_cov)
    inner_before = _global_inner(g_pred, g_cov)
    denom = pred_norm * cov_norm_before
    cosine_before = None if float(denom.detach()) <= EPS else float((inner_before / denom).detach())

    conflict = bool(float(inner_before.detach()) < 0.0 and float(pred_norm.detach()) > EPS)
    if conflict:
        coefficient = inner_before / (_global_inner(g_pred, g_pred) + EPS)
        g_cov = [cov - coefficient * pred for cov, pred in zip(g_cov, g_pred)]

    cov_norm_projected = _global_norm(g_cov)
    limit = max_coverage_to_prediction_ratio * pred_norm
    cap = bool(float(cov_norm_projected.detach()) > float(limit.detach()) and float(limit.detach()) > EPS)
    if cap:
        scale = limit / (cov_norm_projected + EPS)
        g_cov = [gradient * scale for gradient in g_cov]

    cov_norm_after = _global_norm(g_cov)
    inner_after = _global_inner(g_pred, g_cov)
    denom_after = pred_norm * cov_norm_after
    cosine_after = None if float(denom_after.detach()) <= EPS else float((inner_after / denom_after).detach())
    combined = [pred + cov for pred, cov in zip(g_pred, g_cov)]
    diagnostics = ProjectionDiagnostics(
        prediction_gradient_norm=float(pred_norm.detach()),
        coverage_gradient_norm_before=float(cov_norm_before.detach()),
        coverage_gradient_norm_after=float(cov_norm_after.detach()),
        gradient_cosine_before=cosine_before,
        gradient_cosine_after=cosine_after,
        conflict_projected=conflict,
        norm_capped=cap,
        max_coverage_to_prediction_ratio=float(max_coverage_to_prediction_ratio),
    )
    return combined, diagnostics


def train_prediction_guarded_coverage(
    model: CoverageAlignedRawChainJRNGC,
    x_full,
    *,
    schedule: Sequence[Mapping[str, object]],
    max_iter: int,
    learning_rate: float = 1e-3,
    weight_decay: float = 0.0,
    gradient_clip_norm: float = 1.0,
    max_coverage_to_prediction_ratio: float = 1.0,
) -> Dict[str, object]:
    """Development trainer for full-prefix coverage with gradient projection."""

    if len(schedule) != max_iter:
        raise ValueError("schedule length must equal max_iter")
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.Adam(parameters, lr=learning_rate, weight_decay=weight_decay)
    trace: Dict[str, List[object]] = {
        "fixed_target_prediction_mse": [],
        "jacobian_penalty": [],
        "total_regularized_objective": [],
        "projection": [],
    }
    for iteration, entry in enumerate(schedule):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        raw = as_raw_bdt(
            x_full,
            device=model.device,
            dtype=model.dtype,
            require_grad=True,
        )
        components = model.loss_components(raw, entry)
        prediction_loss = components["fixed_target_prediction_mse"]
        coverage_loss = components["jacobian_penalty"]
        prediction_gradients = torch.autograd.grad(
            prediction_loss,
            parameters,
            retain_graph=True,
            allow_unused=True,
        )
        coverage_gradients = torch.autograd.grad(
            coverage_loss,
            parameters,
            retain_graph=False,
            allow_unused=True,
        )
        combined, diagnostics = project_coverage_gradient(
            prediction_gradients,
            coverage_gradients,
            parameters,
            max_coverage_to_prediction_ratio=max_coverage_to_prediction_ratio,
        )
        for parameter, gradient in zip(parameters, combined):
            if not torch.isfinite(gradient).all():
                raise FloatingPointError(f"Nonfinite projected gradient at iteration {iteration}")
            parameter.grad = gradient.detach()
        torch.nn.utils.clip_grad_norm_(parameters, gradient_clip_norm)
        optimizer.step()
        total = prediction_loss + coverage_loss
        trace["fixed_target_prediction_mse"].append(float(prediction_loss.detach()))
        trace["jacobian_penalty"].append(float(coverage_loss.detach()))
        trace["total_regularized_objective"].append(float(total.detach()))
        trace["projection"].append(diagnostics.as_dict())
    return {
        "training_policy": "phase9_prediction_guarded_coverage_development_only",
        "optimizer": "Adam",
        "iterations_completed": int(max_iter),
        "learning_rate": float(learning_rate),
        "weight_decay": float(weight_decay),
        "gradient_clip_norm": float(gradient_clip_norm),
        "max_coverage_to_prediction_ratio": float(max_coverage_to_prediction_ratio),
        "trace": trace,
    }

