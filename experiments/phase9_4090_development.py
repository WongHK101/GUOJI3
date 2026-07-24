"""Development-only RTX 4090 experiments for bounded Phase 9 hypotheses.

This runner never writes into frozen Phase 7/8 result roots. It evaluates:

1. a constrained adaptive FIR initialized from the FixedFIR3 reference on D2;
2. prediction-guarded optimization of the Phase 8 full-prefix repair.

All outputs are retrospective/development evidence. They cannot become formal
paper evidence without an independently frozen confirmation protocol.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
for _path in (PROJECT_ROOT, SRC_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from factorial_data import FACTORIAL_SETTINGS, generate_factorial_cell  # noqa: E402
from phase8_coverage import (  # noqa: E402
    CoverageAlignedRawChainJRNGC,
    Phase8ModelConfig,
    as_raw_bdt,
    build_stratified_lag_schedule,
    coefficient_r_total_lag1,
    extract_attribution_objects,
    schedule_sha256 as phase8_schedule_sha256,
)
from phase9_adaptive_repair import (  # noqa: E402
    ConstrainedAdaptiveFIRJRNGC,
    train_history_guarded_coverage,
    train_prediction_guarded_coverage,
)
from repaired_istf import (  # noqa: E402
    RepairedISTFConfig,
    canonical_metric_adapter,
    eligible_target_indices,
    evaluate_repaired_model_chunked,
    instantiate_repaired_method,
    make_cyclic_schedule,
    schedule_hash,
)


CELL_FLAGS = {
    "Stat+Linear": (True, True),
    "Stat+Nonlinear": (True, False),
    "NS+Linear": (False, True),
    "NS+Nonlinear": (False, False),
}
D2_METHODS = ("baseline", "cp_depthwise", "fixed_fir3", "adaptive_fir")
PHASE8_METHODS = (
    "coverage_standard",
    "coverage_prediction_guarded",
    "coverage_history_guarded",
)


def parse_csv_ints(value: str) -> List[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_csv_strings(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--track",
        choices=("d2_adaptive_fir", "phase8_gradient_guard", "all"),
        default="all",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-iter", type=int, default=500)
    parser.add_argument("--d2-data-seeds", default="0,101,102")
    parser.add_argument("--phase8-data-seeds", default="13001,13002,13003")
    parser.add_argument("--train-seeds", default="0")
    parser.add_argument("--d2-cells", default=",".join(CELL_FLAGS))
    parser.add_argument("--d2-methods", default=",".join(D2_METHODS))
    parser.add_argument("--phase8-methods", default=",".join(PHASE8_METHODS))
    parser.add_argument("--phase8-gradient-ratio", type=float, default=1.0)
    parser.add_argument("--skip-full-attribution", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=json_default) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_torch_save(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def atomic_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def git_commit() -> str:
    import subprocess

    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def configure_determinism(seed: int, device: torch.device) -> Dict[str, object]:
    torch.manual_seed(seed)
    np.random.seed(seed % (2**32 - 1))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats(device)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return {
        "seed": int(seed),
        "torch_deterministic_algorithms": True,
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
    }


def environment_payload(device: torch.device) -> Dict[str, object]:
    return {
        "development_only": True,
        "formal_result": False,
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": sys.version,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "device": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else None
        ),
        "git_commit": git_commit(),
    }


def predictor_state(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    prefixes = ("inputgate.", "outputgate.", "encoders.")
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
        if name.startswith(prefixes)
    }


def tensor_state_sha256(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def model_seed(data_seed: int, train_seed: int) -> int:
    return 900_000 + 1000 * int(data_seed) + 10 * int(train_seed)


def jacobian_seed(data_seed: int, train_seed: int) -> int:
    return 910_000 + 1000 * int(data_seed) + 10 * int(train_seed)


def generate_d2(
    *,
    cell: str,
    data_seed: int,
    d: int = 6,
    T: int = 180,
    lag: int = 3,
):
    stationary, linear = CELL_FLAGS[cell]
    params = FACTORIAL_SETTINGS["D2"]
    return generate_factorial_cell(
        d=d,
        T=T,
        lag=lag,
        seed=data_seed,
        stationary=stationary,
        linear=linear,
        coeff_scale=params["coeff_scale"],
        noise_scale=params["noise_scale"],
        regime_shift_strength=0.0 if stationary else params["regime_shift_strength"],
        nonlinear_strength=0.0 if linear else params["nonlinear_strength"],
        nonlinear_scale=params["nonlinear_scale"],
        sparsity=0.2,
        return_metadata=True,
    )


def generate_phase8_var1(
    *,
    d: int,
    T: int,
    seed: int,
    sparsity: float = 0.3,
    noise_scale: float = 0.1,
):
    """Development namespace equivalent of the legacy full-aux VAR(1) generator."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    A = torch.randn((d, d), generator=generator) * (
        torch.rand((d, d), generator=generator) < sparsity
    ).float()
    spectral_radius = float(torch.linalg.eigvals(A).abs().max())
    A = A * (0.8 / max(spectral_radius, 0.01))
    A_np = A.numpy()
    noise_seed = seed + 1
    rng = np.random.RandomState(noise_seed)
    x = np.zeros((d, T), dtype=np.float64)
    x[:, 0] = rng.randn(d) * 0.1
    for t in range(1, T):
        x[:, t] = A_np @ x[:, t - 1] + rng.randn(d) * noise_scale
    graph = (np.abs(A_np) > 0.01).astype(np.float64)
    metadata = {
        "generator": "phase9_development_var1_matches_legacy_full_aux_equations",
        "development_only": True,
        "graph_seed": int(seed),
        "noise_seed": int(noise_seed),
        "d": int(d),
        "T": int(T),
        "sparsity": float(sparsity),
        "noise_scale": float(noise_scale),
        "spectral_radius": float(torch.linalg.eigvals(torch.as_tensor(A_np)).abs().max()),
    }
    return x, graph, A_np, metadata


def make_d2_model(
    method: str,
    *,
    data_seed: int,
    train_seed: int,
) -> tuple[torch.nn.Module, RepairedISTFConfig, str]:
    identity_lam = 0.05 if method == "cp_depthwise" else 0.0
    cfg = RepairedISTFConfig(
        d=6,
        lag=3,
        attribution_horizon=32,
        layers=1,
        hidden=16,
        dropout=0.0,
        jacobian_lam=0.01,
        identity_lam=identity_lam,
        residual_gain=0.1,
        depthwise_kernel_size=3,
        fir3_gamma=0.1,
        dtype="float32",
    )
    seed = model_seed(data_seed, train_seed)
    torch.manual_seed(seed)
    if method == "adaptive_fir":
        model = ConstrainedAdaptiveFIRJRNGC(
            cfg,
            kernel_size=3,
            gate_max=0.25,
            init_gate=0.1,
        )
    else:
        model = instantiate_repaired_method(method, cfg)
    return model, cfg, tensor_state_sha256(predictor_state(model))


def finite_payload(value: object) -> bool:
    if isinstance(value, Mapping):
        return all(finite_payload(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite_payload(item) for item in value)
    if isinstance(value, (float, np.floating)):
        return bool(np.isfinite(value))
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all())
    if isinstance(value, np.ndarray):
        return bool(np.isfinite(value).all())
    return True


def run_d2_record(
    *,
    output_root: Path,
    device: torch.device,
    cell: str,
    method: str,
    data_seed: int,
    train_seed: int,
    max_iter: int,
    resume: bool,
) -> Dict[str, object]:
    run_id = (
        f"d2__{cell.replace('+', '_').replace(' ', '')}"
        f"__{method}__ds{data_seed}__ts{train_seed}__it{max_iter}"
    )
    run_dir = output_root / "d2_adaptive_fir" / run_id
    status_path = run_dir / "status.json"
    if resume and status_path.exists():
        existing = json.loads(status_path.read_text(encoding="utf-8"))
        if existing.get("status") == "complete":
            return existing
    run_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    atomic_json(status_path, {
        "status": "running",
        "run_id": run_id,
        "development_only": True,
        "formal_result": False,
    })

    x, graph, metadata = generate_d2(cell=cell, data_seed=data_seed)
    deterministic = configure_determinism(model_seed(data_seed, train_seed), device)
    model, cfg, predictor_hash = make_d2_model(
        method,
        data_seed=data_seed,
        train_seed=train_seed,
    )
    model = model.to(device)
    indices = eligible_target_indices(x.shape[1], cfg.lag, cfg.attribution_horizon)
    schedule = make_cyclic_schedule(
        indices,
        d=cfg.d,
        max_iter=max_iter,
        windows_per_step=2,
        targets_per_step=2,
        seed=jacobian_seed(data_seed, train_seed),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.0)
    trace: List[Dict[str, float]] = []
    train_started = time.perf_counter()
    for iteration, entry in enumerate(schedule, start=1):
        optimizer.zero_grad(set_to_none=True)
        components = model.compute_loss_components(
            x,
            schedule_entry=entry,
            target_indices=indices,
        )
        components["total_training_objective"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        trace.append({
            "iteration": iteration,
            **{
                key: float(value.detach().cpu())
                for key, value in components.items()
            },
        })
    training_seconds = time.perf_counter() - train_started

    eval_started = time.perf_counter()
    evaluation = evaluate_repaired_model_chunked(
        model,
        x,
        graph,
        target_indices=indices,
        attribution_horizon=cfg.attribution_horizon,
        chunk_size=64,
        include_filtered_coordinate=True,
        prediction_target_indices=indices,
        leakage_target_indices=indices[: min(16, len(indices))],
    )
    evaluation_seconds = time.perf_counter() - eval_started
    score_nominal = np.asarray(evaluation["score_nominal"], dtype=np.float64)
    filtered_score = np.asarray(
        evaluation["filtered_coordinate_score_nominal"],
        dtype=np.float64,
    )
    score_delta = score_nominal - filtered_score
    diagnostics = {
        "filter": evaluation["filter_diagnostics"],
        "cross_variable_leakage": evaluation["cross_variable_leakage"],
        "temporal_horizon_mass": evaluation["temporal_horizon_mass"],
        "raw_vs_filtered_score_max_abs_difference": float(np.max(np.abs(score_delta))),
        "raw_vs_filtered_score_mean_abs_difference": float(np.mean(np.abs(score_delta))),
    }
    metrics = {
        "nominal": evaluation["metrics_nominal"],
        "full_H": evaluation["metrics_full_H"],
        "eval_raw_prediction_loss": evaluation["eval_raw_prediction_loss"],
        "eval_filtered_prediction_loss": evaluation["eval_filtered_prediction_loss"],
    }
    no_nan_inf = finite_payload(trace) and finite_payload(metrics) and finite_payload(diagnostics)
    atomic_json(run_dir / "config.json", {
        "track": "d2_adaptive_fir",
        "run_id": run_id,
        "method": method,
        "cell": cell,
        "data_seed": data_seed,
        "train_seed": train_seed,
        "max_iter": max_iter,
        "model": asdict(cfg),
        "predictor_initialization_sha256": predictor_hash,
        "jacobian_schedule_sha256": schedule_hash(schedule),
        "development_only": True,
        "formal_result": False,
    })
    atomic_json(run_dir / "generator_metadata.json", metadata)
    atomic_json(run_dir / "determinism.json", deterministic)
    atomic_json(run_dir / "loss_trace.json", trace)
    atomic_json(run_dir / "metrics.json", metrics)
    atomic_json(run_dir / "diagnostics.json", diagnostics)
    atomic_npz(
        run_dir / "scores.npz",
        raw_chain_j_bar=np.asarray(evaluation["raw_chain_j_bar"]),
        score_nominal=score_nominal,
        score_full_H=np.asarray(evaluation["score_full_H"]),
        filtered_coordinate_score_nominal=filtered_score,
    )
    atomic_torch_save(run_dir / "checkpoint.pt", {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "config": asdict(cfg),
        "run_id": run_id,
        "iteration": max_iter,
        "development_only": True,
    })
    status = {
        "status": "complete" if no_nan_inf else "failed",
        "run_id": run_id,
        "track": "d2_adaptive_fir",
        "method": method,
        "cell": cell,
        "data_seed": data_seed,
        "train_seed": train_seed,
        "max_iter": max_iter,
        "training_seconds": training_seconds,
        "evaluation_seconds": evaluation_seconds,
        "wall_seconds": time.perf_counter() - started,
        "cuda_max_memory_allocated_mb": (
            float(torch.cuda.max_memory_allocated(device) / 1024**2)
            if device.type == "cuda"
            else None
        ),
        "no_nan_inf": bool(no_nan_inf),
        "development_only": True,
        "formal_result": False,
    }
    atomic_json(status_path, status)
    return status


def make_phase8_model(
    *,
    data_seed: int,
    train_seed: int,
    device: torch.device,
) -> tuple[CoverageAlignedRawChainJRNGC, Phase8ModelConfig, Dict[str, torch.Tensor]]:
    cfg = Phase8ModelConfig(
        d=8,
        lag=1,
        layers=3,
        hidden=32,
        dropout=0.0,
        d_cond=4,
        d_state=4,
        d_conv=4,
        expand=2,
        jacobian_lam=0.01,
        attribution_horizon=32,
        dtype="float32",
    )
    torch.manual_seed(model_seed(data_seed, train_seed))
    model = CoverageAlignedRawChainJRNGC(cfg).to(device)
    initial = {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }
    return model, cfg, initial


def train_phase8_standard(
    model: CoverageAlignedRawChainJRNGC,
    x_full,
    *,
    schedule: Sequence[Mapping[str, object]],
    max_iter: int,
) -> Dict[str, object]:
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.0)
    trace = {
        "fixed_target_prediction_mse": [],
        "jacobian_penalty": [],
        "total_regularized_objective": [],
    }
    for entry in schedule:
        optimizer.zero_grad(set_to_none=True)
        raw = as_raw_bdt(
            x_full,
            device=model.device,
            dtype=model.dtype,
            require_grad=True,
        )
        components = model.loss_components(raw, entry)
        components["total_regularized_objective"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        for name in trace:
            trace[name].append(float(components[name].detach().cpu()))
    return {
        "training_policy": "phase8_standard_total_objective_development_replay",
        "iterations_completed": int(max_iter),
        "trace": trace,
    }


def vector_correlation(a: np.ndarray, b: np.ndarray) -> float | None:
    mask = ~np.eye(a.shape[0], dtype=bool)
    av = np.asarray(a, dtype=np.float64)[mask]
    bv = np.asarray(b, dtype=np.float64)[mask]
    if np.std(av) <= 1e-12 or np.std(bv) <= 1e-12:
        return None
    return float(np.corrcoef(av, bv)[0, 1])


def phase8_evaluation(
    model: CoverageAlignedRawChainJRNGC,
    x: np.ndarray,
    graph: np.ndarray,
    A: np.ndarray,
    *,
    skip_full_attribution: bool,
) -> tuple[Dict[str, object], Dict[str, np.ndarray]]:
    raw = as_raw_bdt(x, device=model.device, dtype=model.dtype)
    with torch.no_grad():
        pure_mse = float(model.pure_mse(raw).detach().cpu())
    if skip_full_attribution:
        return {
            "fixed_target_prediction_mse": pure_mse,
            "full_attribution_skipped": True,
        }, {}
    attribution = extract_attribution_objects(
        model,
        x,
        true_edge_count=int(np.sum(graph) - np.trace(graph)),
        n_min=50,
    )
    metrics = canonical_metric_adapter(graph, attribution.s_gc_total)
    coefficient_r = coefficient_r_total_lag1(
        attribution.j_bar_total_lag1,
        A,
    )
    payload = {
        "fixed_target_prediction_mse": pure_mse,
        "metrics_total_nominal": metrics,
        "coefficient_r_total_lag1": coefficient_r,
        "m_missing": attribution.m_missing,
        "partial_total_nominal_pearson": attribution.nominal_partial_total_pearson,
        "partial_total_nominal_topk_jaccard": attribution.nominal_partial_total_topk_jaccard,
        "temporal_tail_statistics": attribution.temporal_tail_statistics,
        "reliable_history_vs_nominal_pearson": vector_correlation(
            attribution.s_reliable_history,
            attribution.s_gc_total,
        ),
        "n_min": attribution.n_min,
        "reliable_horizon_max": int(np.max(attribution.h_reliable)),
        "full_attribution_skipped": False,
    }
    arrays = {
        "j_bar_total": attribution.j_bar_total,
        "j_bar_partial": attribution.j_bar_partial,
        "j_bar_missing": attribution.j_bar_missing,
        "s_gc_total": attribution.s_gc_total,
        "s_partial_nominal": attribution.s_partial_nominal,
        "s_reliable_history": attribution.s_reliable_history,
        "s_prefix_all": attribution.s_prefix_all,
        "j_bar_total_lag1": attribution.j_bar_total_lag1,
        "eligible_window_count_by_lag": attribution.eligible_window_count_by_lag,
        "h_reliable": attribution.h_reliable,
    }
    return payload, arrays


def projection_summary(training: Mapping[str, object]) -> Dict[str, object] | None:
    trace = training.get("trace")
    if not isinstance(trace, Mapping) or "projection" not in trace:
        return None
    rows = trace["projection"]
    if not isinstance(rows, Sequence) or not rows:
        return None
    conflicts = np.asarray([bool(row["conflict_projected"]) for row in rows])
    caps = np.asarray([bool(row["norm_capped"]) for row in rows])
    before = np.asarray([
        np.nan if row["gradient_cosine_before"] is None else row["gradient_cosine_before"]
        for row in rows
    ], dtype=np.float64)
    after = np.asarray([
        np.nan if row["gradient_cosine_after"] is None else row["gradient_cosine_after"]
        for row in rows
    ], dtype=np.float64)
    ratios = np.asarray([
        row["coverage_gradient_norm_before"] / max(row["prediction_gradient_norm"], 1e-12)
        for row in rows
    ], dtype=np.float64)
    return {
        "conflict_fraction": float(np.mean(conflicts)),
        "norm_cap_fraction": float(np.mean(caps)),
        "median_cosine_before": (
            None if np.all(np.isnan(before)) else float(np.nanmedian(before))
        ),
        "median_cosine_after": (
            None if np.all(np.isnan(after)) else float(np.nanmedian(after))
        ),
        "median_raw_coverage_to_prediction_gradient_ratio": float(np.median(ratios)),
        "p95_raw_coverage_to_prediction_gradient_ratio": float(np.quantile(ratios, 0.95)),
    }


def parameter_displacement(
    model: torch.nn.Module,
    initial_state: Mapping[str, torch.Tensor],
) -> Dict[str, float]:
    totals = {"all": 0.0, "preprocessor": 0.0, "predictor": 0.0}
    for name, value in model.state_dict().items():
        if name not in initial_state or not value.is_floating_point():
            continue
        delta2 = float(torch.sum(
            (value.detach().cpu() - initial_state[name].to(value.dtype)) ** 2
        ))
        totals["all"] += delta2
        group = "preprocessor" if name.startswith("preprocessor.") else "predictor"
        totals[group] += delta2
    return {f"{name}_l2": math.sqrt(value) for name, value in totals.items()}


def run_phase8_record(
    *,
    output_root: Path,
    device: torch.device,
    method: str,
    data_seed: int,
    train_seed: int,
    max_iter: int,
    gradient_ratio: float,
    skip_full_attribution: bool,
    resume: bool,
) -> Dict[str, object]:
    run_id = f"phase8__{method}__ds{data_seed}__ts{train_seed}__it{max_iter}"
    run_dir = output_root / "phase8_gradient_guard" / run_id
    status_path = run_dir / "status.json"
    if resume and status_path.exists():
        existing = json.loads(status_path.read_text(encoding="utf-8"))
        if existing.get("status") == "complete":
            return existing
    run_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    atomic_json(status_path, {
        "status": "running",
        "run_id": run_id,
        "development_only": True,
        "formal_result": False,
    })
    x, graph, A, metadata = generate_phase8_var1(d=8, T=500, seed=data_seed)
    deterministic = configure_determinism(model_seed(data_seed, train_seed), device)
    model, cfg, initial_state = make_phase8_model(
        data_seed=data_seed,
        train_seed=train_seed,
        device=device,
    )
    schedule = build_stratified_lag_schedule(
        T=500,
        lag=1,
        d_out=8,
        max_iter=max_iter,
        seed=jacobian_seed(data_seed, train_seed),
    )
    train_started = time.perf_counter()
    if method == "coverage_standard":
        training = train_phase8_standard(
            model,
            x,
            schedule=schedule,
            max_iter=max_iter,
        )
    elif method == "coverage_prediction_guarded":
        training = train_prediction_guarded_coverage(
            model,
            x,
            schedule=schedule,
            max_iter=max_iter,
            learning_rate=1e-3,
            weight_decay=0.0,
            gradient_clip_norm=1.0,
            max_coverage_to_prediction_ratio=gradient_ratio,
        )
    elif method == "coverage_history_guarded":
        training = train_history_guarded_coverage(
            model,
            x,
            schedule=schedule,
            max_iter=max_iter,
            learning_rate=1e-3,
            weight_decay=0.0,
            gradient_clip_norm=1.0,
            max_history_to_core_gradient_ratio=gradient_ratio,
        )
    else:
        raise ValueError(f"Unknown Phase 8 development method: {method}")
    training_seconds = time.perf_counter() - train_started
    eval_started = time.perf_counter()
    evaluation, arrays = phase8_evaluation(
        model,
        x,
        graph,
        A,
        skip_full_attribution=skip_full_attribution,
    )
    evaluation_seconds = time.perf_counter() - eval_started
    displacement = parameter_displacement(model, initial_state)
    diagnostics = {
        "parameter_displacement": displacement,
        "projection": projection_summary(training),
    }
    no_nan_inf = (
        finite_payload(training)
        and finite_payload(evaluation)
        and finite_payload(diagnostics)
    )
    atomic_json(run_dir / "config.json", {
        "track": "phase8_gradient_guard",
        "run_id": run_id,
        "method": method,
        "data_seed": data_seed,
        "train_seed": train_seed,
        "max_iter": max_iter,
        "model": asdict(cfg),
        "gradient_ratio": (
            gradient_ratio
            if method in {"coverage_prediction_guarded", "coverage_history_guarded"}
            else None
        ),
        "schedule_sha256": phase8_schedule_sha256(schedule),
        "initial_state_sha256": tensor_state_sha256(initial_state),
        "score_object": "total_raw_chain_nominal_lag",
        "development_only": True,
        "formal_result": False,
    })
    atomic_json(run_dir / "generator_metadata.json", metadata)
    atomic_json(run_dir / "determinism.json", deterministic)
    atomic_json(run_dir / "training.json", training)
    atomic_json(run_dir / "evaluation.json", evaluation)
    atomic_json(run_dir / "diagnostics.json", diagnostics)
    if arrays:
        atomic_npz(run_dir / "attribution_objects.npz", **arrays)
    atomic_torch_save(run_dir / "checkpoint.pt", {
        "model_state": model.state_dict(),
        "config": asdict(cfg),
        "run_id": run_id,
        "iteration": max_iter,
        "development_only": True,
    })
    status = {
        "status": "complete" if no_nan_inf else "failed",
        "run_id": run_id,
        "track": "phase8_gradient_guard",
        "method": method,
        "data_seed": data_seed,
        "train_seed": train_seed,
        "max_iter": max_iter,
        "training_seconds": training_seconds,
        "evaluation_seconds": evaluation_seconds,
        "wall_seconds": time.perf_counter() - started,
        "cuda_max_memory_allocated_mb": (
            float(torch.cuda.max_memory_allocated(device) / 1024**2)
            if device.type == "cuda"
            else None
        ),
        "no_nan_inf": bool(no_nan_inf),
        "development_only": True,
        "formal_result": False,
    }
    atomic_json(status_path, status)
    return status


def collect_statuses(output_root: Path) -> List[Dict[str, object]]:
    statuses = []
    for path in sorted(output_root.rglob("status.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["status_path"] = str(path)
            statuses.append(payload)
        except json.JSONDecodeError:
            continue
    return statuses


def run_selected(args: argparse.Namespace) -> int:
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    args.output_root.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output_root / "environment.json", environment_payload(device))
    atomic_json(args.output_root / "protocol.json", {
        "development_only": True,
        "formal_result": False,
        "phase7_seeds_4_to_8_accessed": False,
        "stage1b_started": False,
        "manuscript_modified": False,
        "track": args.track,
        "max_iter": args.max_iter,
        "d2_data_seeds": parse_csv_ints(args.d2_data_seeds),
        "phase8_data_seeds": parse_csv_ints(args.phase8_data_seeds),
        "train_seeds": parse_csv_ints(args.train_seeds),
    })
    failures = []
    if args.track in {"d2_adaptive_fir", "all"}:
        cells = parse_csv_strings(args.d2_cells)
        methods = parse_csv_strings(args.d2_methods)
        unknown_cells = set(cells) - set(CELL_FLAGS)
        unknown_methods = set(methods) - set(D2_METHODS)
        if unknown_cells or unknown_methods:
            raise ValueError(
                f"Unknown D2 cells={sorted(unknown_cells)}, methods={sorted(unknown_methods)}"
            )
        for data_seed in parse_csv_ints(args.d2_data_seeds):
            for train_seed in parse_csv_ints(args.train_seeds):
                expected_hash = None
                for cell in cells:
                    for method in methods:
                        status = run_d2_record(
                            output_root=args.output_root,
                            device=device,
                            cell=cell,
                            method=method,
                            data_seed=data_seed,
                            train_seed=train_seed,
                            max_iter=args.max_iter,
                            resume=args.resume,
                        )
                        config_path = (
                            args.output_root
                            / "d2_adaptive_fir"
                            / str(status["run_id"])
                            / "config.json"
                        )
                        config = json.loads(config_path.read_text(encoding="utf-8"))
                        current_hash = config["predictor_initialization_sha256"]
                        if expected_hash is None:
                            expected_hash = current_hash
                        elif current_hash != expected_hash:
                            raise AssertionError(
                                "D2 paired predictor initialization mismatch "
                                f"for data_seed={data_seed}, train_seed={train_seed}"
                            )
                        if status["status"] != "complete":
                            failures.append(status["run_id"])
    if args.track in {"phase8_gradient_guard", "all"}:
        methods = parse_csv_strings(args.phase8_methods)
        unknown = set(methods) - set(PHASE8_METHODS)
        if unknown:
            raise ValueError(f"Unknown Phase 8 methods: {sorted(unknown)}")
        for data_seed in parse_csv_ints(args.phase8_data_seeds):
            for train_seed in parse_csv_ints(args.train_seeds):
                for method in methods:
                    status = run_phase8_record(
                        output_root=args.output_root,
                        device=device,
                        method=method,
                        data_seed=data_seed,
                        train_seed=train_seed,
                        max_iter=args.max_iter,
                        gradient_ratio=args.phase8_gradient_ratio,
                        skip_full_attribution=args.skip_full_attribution,
                        resume=args.resume,
                    )
                    if status["status"] != "complete":
                        failures.append(status["run_id"])
    statuses = collect_statuses(args.output_root)
    atomic_json(args.output_root / "run_summary.json", {
        "development_only": True,
        "formal_result": False,
        "status_count": len(statuses),
        "complete_count": sum(row.get("status") == "complete" for row in statuses),
        "failed_count": sum(row.get("status") == "failed" for row in statuses),
        "failures": failures,
        "statuses": statuses,
    })
    return 1 if failures else 0


def main() -> int:
    return run_selected(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
