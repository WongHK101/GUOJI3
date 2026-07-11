"""Checkpoint-locked training helpers for future authorized Phase 8 runs.

No function in this module is invoked by the CPU-preflight command. The module
exists so checkpoint selection cannot be improvised when GPU execution is
later reviewed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Mapping, MutableMapping, Optional, Sequence

import numpy as np
import torch

from phase8_coverage import (
    CoverageAlignedRawChainJRNGC,
    LegacyComparatorAdapter,
    Phase8NoAuxInputSpaceControl,
    as_raw_bdt,
)


LEGACY_RESTORE_POLICY = "legacy_min_total_objective_restore"
FIXED_FINAL_POLICY = "fixed_full_budget_final"
FIXED_FINAL_NON_EVIDENTIARY_POLICY = "fixed_full_budget_final_non_evidentiary"


@dataclass(frozen=True)
class TrainingMetadata:
    checkpoint_policy: str
    gating_checkpoint: str
    selected_iteration: int
    best_total_objective_iteration: int
    iterations_completed: int
    early_stopped: bool
    check_every: int
    lookback: int
    optimizer: str
    learning_rate: float
    weight_decay: float
    gradient_clip_norm: float

    def as_dict(self) -> Dict[str, object]:
        return asdict(self)


def _state_to_cpu(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def separated_loss_components(
    handle,
    x_full,
    *,
    schedule_entry: Optional[Mapping[str, object]],
) -> Dict[str, torch.Tensor]:
    if isinstance(handle, LegacyComparatorAdapter):
        return handle.loss_components(x_full)
    if isinstance(handle, CoverageAlignedRawChainJRNGC):
        if schedule_entry is None:
            raise ValueError("Coverage-aligned repair requires a frozen schedule entry")
        raw = as_raw_bdt(
            x_full,
            device=handle.device,
            dtype=handle.dtype,
            require_grad=True,
        )
        return handle.loss_components(raw, schedule_entry)
    if isinstance(handle, Phase8NoAuxInputSpaceControl):
        if schedule_entry is None:
            raise ValueError("No-auxiliary matched control requires its frozen cyclic schedule")
        return handle.loss_components(x_full, schedule_entry=schedule_entry)
    raise TypeError(f"Unsupported Phase 8 training handle: {type(handle).__name__}")


def _objective(
    handle,
    x_full,
    *,
    schedule_entry: Optional[Mapping[str, object]],
) -> torch.Tensor:
    if isinstance(handle, LegacyComparatorAdapter):
        return handle.direct_total_objective(x_full)
    return separated_loss_components(
        handle,
        x_full,
        schedule_entry=schedule_entry,
    )["total_regularized_objective"]


def train_with_frozen_checkpoint_policy(
    handle,
    x_full,
    *,
    max_iter: int,
    checkpoint_policy: str,
    schedule: Optional[Sequence[Mapping[str, object]]] = None,
    learning_rate: float = 1e-3,
    weight_decay: float = 0.0,
    gradient_clip_norm: float = 1.0,
    check_every: int = 50,
    lookback: int = 10,
    objective_trace: Optional[List[float]] = None,
    capture_completed_iterations: Sequence[int] = (),
    captured_states: Optional[MutableMapping[int, Dict[str, torch.Tensor]]] = None,
    component_trace: Optional[MutableMapping[str, List[float]]] = None,
) -> TrainingMetadata:
    """Train under either exact legacy restore or fixed-final semantics."""
    if checkpoint_policy not in {
        LEGACY_RESTORE_POLICY,
        FIXED_FINAL_POLICY,
        FIXED_FINAL_NON_EVIDENTIARY_POLICY,
    }:
        raise ValueError(f"Unsupported checkpoint policy: {checkpoint_policy}")
    if schedule is not None and len(schedule) != max_iter:
        raise ValueError("Frozen estimator schedule length must equal max_iter")
    model = handle.model if isinstance(handle, (LegacyComparatorAdapter, Phase8NoAuxInputSpaceControl)) else handle
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    best_total = float("inf")
    best_iteration = -1
    best_state: Optional[Dict[str, torch.Tensor]] = None
    early_stopped = False
    iterations_completed = 0
    capture_set = set(int(value) for value in capture_completed_iterations)

    for iteration in range(max_iter):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        entry = None if schedule is None else schedule[iteration]
        if component_trace is None:
            objective = _objective(handle, x_full, schedule_entry=entry)
        else:
            components = separated_loss_components(handle, x_full, schedule_entry=entry)
            objective = components["total_regularized_objective"]
            for name, value in components.items():
                component_trace.setdefault(name, []).append(float(value.detach()))
        if not torch.isfinite(objective):
            raise FloatingPointError(f"Nonfinite total objective at iteration {iteration}")
        objective.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
        optimizer.step()
        iterations_completed = iteration + 1
        if objective_trace is not None:
            objective_trace.append(float(objective.detach()))
        if iterations_completed in capture_set:
            if captured_states is None:
                raise ValueError("captured_states mapping is required when checkpoint capture is requested")
            captured_states[iterations_completed] = _state_to_cpu(model)

        if iteration % check_every == 0:
            value = float(objective.detach())
            if value < best_total:
                best_total = value
                best_iteration = iteration
                best_state = _state_to_cpu(model)
            elif (
                checkpoint_policy == LEGACY_RESTORE_POLICY
                and iteration > 1000
                and (iteration - best_iteration) >= lookback * check_every
            ):
                early_stopped = True
                break

    if best_state is None:
        raise RuntimeError("No checkpoint was recorded")
    if checkpoint_policy == LEGACY_RESTORE_POLICY:
        model.load_state_dict(best_state)
        selected_iteration = best_iteration
        gating_checkpoint = "restored_legacy_best"
    else:
        selected_iteration = iterations_completed - 1
        gating_checkpoint = "final"

    return TrainingMetadata(
        checkpoint_policy=checkpoint_policy,
        gating_checkpoint=gating_checkpoint,
        selected_iteration=selected_iteration,
        best_total_objective_iteration=best_iteration,
        iterations_completed=iterations_completed,
        early_stopped=early_stopped,
        check_every=check_every,
        lookback=lookback,
        optimizer="Adam",
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        gradient_clip_norm=gradient_clip_norm,
    )


def recompute_selected_state_metrics(handle, x_full) -> Dict[str, float]:
    """Recompute separated fields after the policy-selected state is loaded."""
    if not isinstance(handle, LegacyComparatorAdapter):
        raise TypeError("Selected-state recomputation currently applies to legacy adapters")
    model = handle.model
    model.eval()
    components = handle.loss_components(x_full)
    return {
        key: float(value.detach().cpu())
        for key, value in components.items()
    }
