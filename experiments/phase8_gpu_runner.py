"""Release-locked Phase 8 GPU record runner.

This runner executes one authorized matrix record at a time. Confirmation
records are always rejected; a separate confirmation token is intentionally
unsupported in this release.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
for path in [PROJECT_ROOT, SRC_DIR]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from phase8_coverage import (  # noqa: E402
    CoverageAlignedRawChainJRNGC,
    LegacyBaselineAdapter,
    LegacyComparatorAdapter,
    Phase8ModelConfig,
    Phase8NoAuxInputSpaceControl,
    as_raw_bdt,
    build_balanced_lag_schedule,
    build_no_aux_control_schedule,
    coefficient_r_total_lag1,
    extract_attribution_objects,
    extract_baseline_attribution_objects,
    fixed_target_concat_interventions,
    fixed_target_no_aux_interventions,
    make_legacy_baseline,
    make_legacy_concat,
    make_legacy_full_aux,
    make_no_aux_input_space_control,
    schedule_sha256,
    target_indices,
)
from phase8_protocol import (  # noqa: E402
    file_sha256,
    load_json,
    load_run_matrix,
    resolve_run_record,
    validate_run_matrix,
    verify_release_lock,
)
from phase8_training import (  # noqa: E402
    separated_loss_components,
    train_with_frozen_checkpoint_policy,
)
from repaired_istf import canonical_metric_adapter  # noqa: E402
from nonstationary_var import _generate_one as generate_nsvar_one  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-matrix", type=Path, required=True)
    parser.add_argument("--release-lock-dir", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--record-id", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    temporary.replace(path)


def atomic_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def atomic_torch_save(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def configure_determinism(seed: int, device: torch.device) -> Dict[str, object]:
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("CUBLAS_WORKSPACE_CONFIG must be :4096:8")
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return {
        "seed": seed,
        "torch_deterministic_algorithms": True,
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
        "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }


def validate_authorization(
    path: Path,
    *,
    release_commit: str,
    config_sha256: str,
    matrix_sha256: str,
    phase: str,
    block: str,
) -> Dict[str, object]:
    payload = load_json(path)
    expected = {
        "authorization": "GPT_APPROVED_PHASE8_GPU_PREFLIGHT_REPLICATION_PILOT",
        "release_commit": release_commit,
        "config_sha256": config_sha256,
        "run_matrix_sha256": matrix_sha256,
    }
    failures = {key: {"actual": payload.get(key), "expected": value} for key, value in expected.items() if payload.get(key) != value}
    if failures:
        raise PermissionError(f"Execution authorization mismatch: {failures}")
    if phase == "gated_confirmation" or block == "repair_confirmation":
        raise PermissionError("Phase 8 confirmation is sealed and cannot execute in this release")
    allowed_blocks = set(payload.get("allowed_blocks", []))
    if block not in allowed_blocks:
        raise PermissionError(f"Block {block} is not authorized")
    return payload


def generate_var1_data(
    d: int,
    T: int,
    seed: int,
    *,
    numpy_seed_offset: int,
    generator_name: str,
    sparsity: float = 0.3,
    noise_scale: float = 0.1,
):
    """Reproduce the frozen legacy VAR(1) generators without touching model RNG."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    A = torch.randn((d, d), generator=generator) * (
        torch.rand((d, d), generator=generator) < sparsity
    ).float()
    spectral_radius_tensor = torch.linalg.eigvals(A).abs().max()
    spectral_radius = float(spectral_radius_tensor)
    if numpy_seed_offset == 0:
        # test_shortcut_diagnostics.py used tensor-valued scaling without a
        # zero guard; preserve that exact legacy numerical path.
        A = A * (0.8 / spectral_radius_tensor)
        spectral_radius_guard = "none_legacy_shortcut"
    else:
        A = A * (0.8 / max(spectral_radius, 0.01))
        spectral_radius_guard = "max_float_sr_0.01_legacy_full_aux"
    A_np = A.numpy()
    numpy_seed = seed + int(numpy_seed_offset)
    rng = np.random.RandomState(numpy_seed)
    # The legacy generators accumulated the series in NumPy float64 before
    # each comparator converted the input to its native float32 path.
    x = np.zeros((d, T), dtype=np.float64)
    x[:, 0] = rng.randn(d) * 0.1
    for t in range(1, T):
        x[:, t] = A_np @ x[:, t - 1] + rng.randn(d) * noise_scale
    gc = (np.abs(A_np) > 0.01).astype(np.float64)
    return x, gc, A_np, {
        "generator": generator_name,
        "graph_seed": seed,
        "noise_seed": numpy_seed,
        "numpy_seed_offset": int(numpy_seed_offset),
        "sparsity": sparsity,
        "noise_scale": noise_scale,
        "spectral_radius": float(torch.linalg.eigvals(torch.as_tensor(A_np)).abs().max()),
        "spectral_radius_guard": spectral_radius_guard,
    }


def generate_record_data(record: Mapping[str, object]):
    d = int(record["d"])
    T = int(record["T"])
    seed = int(record["data_seed"])
    if record["block"] == "fixed_target_interventions":
        rng = np.random.RandomState(seed)
        x, gc = generate_nsvar_one(
            d,
            int(record["K"]),
            T,
            rng,
            0.3,
            0.1,
            0.2,
            1.0,
        )
        return x, gc, None, {
            "generator": "nonstationary_var._generate_one",
            "rng_seed": seed,
            "trend_strength": 0.3,
            "coefficient_drift": 0.1,
            "sparsity": 0.2,
            "noise_scale": 1.0,
        }
    if record["block"] in {"capacity_replication", "coefficient_replication"}:
        return generate_var1_data(
            d,
            T,
            seed,
            numpy_seed_offset=0,
            generator_name="legacy_shortcut_diagnostics_var1",
        )
    return generate_var1_data(
        d,
        T,
        seed,
        numpy_seed_offset=1,
        generator_name="legacy_full_aux_penalty_var1",
    )


def model_config(record: Mapping[str, object], execution_config: Mapping[str, object]) -> Phase8ModelConfig:
    block_cfg = execution_config["blocks"][record["block"]]  # type: ignore[index]
    return Phase8ModelConfig(
        d=int(record["d"]),
        lag=int(record["K"]),
        layers=int(block_cfg.get("layers", 3)),
        hidden=int(block_cfg.get("hidden", 32)),
        d_cond=int(record["d_cond"]),
        d_state=int(block_cfg.get("d_state", 4)),
        d_conv=4,
        expand=2,
        jacobian_lam=0.01,
        attribution_horizon=32,
        dtype="float32",
    )


def instantiate_handle(record: Mapping[str, object], execution_config: Mapping[str, object]):
    cfg = model_config(record, execution_config)
    method = str(record["method"])
    if method == "baseline_jrngc":
        return make_legacy_baseline(cfg)
    if method in {"concat_x_only", "concat_fixed_target_interventions"} or method.startswith("concat_dcond_"):
        return make_legacy_concat(cfg)
    if method == "full_aux_equal_lambda":
        return make_legacy_full_aux(cfg, "equal")
    if method == "full_aux_lc10":
        return make_legacy_full_aux(cfg, "lc10")
    if method == "coverage_aligned_raw_chain":
        return CoverageAlignedRawChainJRNGC(cfg)
    if method == "no_aux_fixed_target_interventions":
        return make_no_aux_input_space_control(cfg)
    raise ValueError(f"Unsupported method: {method}")


def underlying_model(handle) -> torch.nn.Module:
    return handle.model if isinstance(handle, (LegacyComparatorAdapter, Phase8NoAuxInputSpaceControl)) else handle


def move_handle(handle, device: torch.device):
    underlying_model(handle).to(device)
    return handle


def prediction_and_target(handle, x_full) -> Tuple[np.ndarray, np.ndarray]:
    model = underlying_model(handle)
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    raw = as_raw_bdt(x_full, device=device, dtype=dtype)
    idx = target_indices(raw.shape[2], int(handle.lag))
    with torch.no_grad():
        if isinstance(handle, LegacyComparatorAdapter):
            prediction = handle.predict_from_raw(raw.to(torch.float32), idx)
            target = handle.raw_targets(raw.to(torch.float32), idx)
        elif isinstance(handle, Phase8NoAuxInputSpaceControl):
            prediction = handle.predict_from_raw(raw, idx)
            target = handle.raw_targets(raw, idx)
        else:
            prediction = handle.predict_from_raw(raw, idx)
            target = handle.raw_targets(raw, idx)
    return prediction.detach().cpu().numpy(), target.detach().cpu().numpy()


def gradient_norms(loss: torch.Tensor, parameters: Sequence[torch.nn.Parameter]) -> Tuple[float, bool]:
    gradients = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
    total = 0.0
    finite = True
    for gradient in gradients:
        if gradient is None:
            continue
        finite = finite and bool(torch.isfinite(gradient).all())
        total += float(torch.sum(gradient.detach() ** 2))
    return math.sqrt(total), finite


def repair_scale_snapshot(handle, x_full, schedule_entry: Mapping[str, object]) -> Dict[str, object]:
    components = separated_loss_components(handle, x_full, schedule_entry=schedule_entry)
    model = underlying_model(handle)
    parameters = tuple(parameter for parameter in model.parameters() if parameter.requires_grad)
    preprocessor_parameters = tuple(
        parameter for name, parameter in model.named_parameters()
        if parameter.requires_grad and name.startswith("preprocessor.")
    )
    predictor_parameters = tuple(
        parameter for name, parameter in model.named_parameters()
        if parameter.requires_grad and not name.startswith("preprocessor.")
    )
    mse_norm, mse_finite = gradient_norms(components["fixed_target_prediction_mse"], parameters)
    penalty_norm, penalty_finite = gradient_norms(components["jacobian_penalty"], parameters)
    mse_predictor_norm, mse_predictor_finite = gradient_norms(
        components["fixed_target_prediction_mse"], predictor_parameters
    )
    mse_preprocessor_norm, mse_preprocessor_finite = gradient_norms(
        components["fixed_target_prediction_mse"], preprocessor_parameters
    )
    penalty_predictor_norm, penalty_predictor_finite = gradient_norms(
        components["jacobian_penalty"], predictor_parameters
    )
    penalty_preprocessor_norm, penalty_preprocessor_finite = gradient_norms(
        components["jacobian_penalty"], preprocessor_parameters
    )
    mse = float(components["fixed_target_prediction_mse"].detach())
    penalty = float(components["jacobian_penalty"].detach())
    pred, target = prediction_and_target(handle, x_full)
    output_variance = float(np.var(pred))
    target_variance = float(np.var(target))
    return {
        "fixed_target_prediction_mse": mse,
        "jacobian_penalty": penalty,
        "total_regularized_objective": float(components["total_regularized_objective"].detach()),
        "lambda_R_lag_over_mse": penalty / max(mse, 1e-12),
        "regularizer_gradient_norm": penalty_norm,
        "prediction_gradient_norm": mse_norm,
        "regularizer_gradient_over_prediction_gradient": penalty_norm / max(mse_norm, 1e-12),
        "prediction_predictor_gradient_norm": mse_predictor_norm,
        "prediction_preprocessor_gradient_norm": mse_preprocessor_norm,
        "regularizer_predictor_gradient_norm": penalty_predictor_norm,
        "regularizer_preprocessor_gradient_norm": penalty_preprocessor_norm,
        "gradients_finite": bool(
            mse_finite
            and penalty_finite
            and mse_predictor_finite
            and mse_preprocessor_finite
            and penalty_predictor_finite
            and penalty_preprocessor_finite
        ),
        "prediction_gradient_nonzero": bool(mse_norm > 0),
        "regularizer_gradient_nonzero": bool(penalty_norm > 0),
        "predictor_gradient_reachable": bool(mse_predictor_norm > 0 and penalty_predictor_norm > 0),
        "preprocessor_gradient_reachable": bool(mse_preprocessor_norm > 0 and penalty_preprocessor_norm > 0),
        "output_variance": output_variance,
        "target_variance": target_variance,
        "output_target_variance_ratio": output_variance / max(target_variance, 1e-12),
    }


def true_edge_count(gc_true: np.ndarray) -> int:
    adjacency = np.asarray(gc_true)
    if adjacency.ndim == 3:
        adjacency = np.any(adjacency != 0, axis=2)
    adjacency = adjacency.copy()
    np.fill_diagonal(adjacency, 0)
    return int(np.sum(adjacency != 0))


def final_loss_components(handle, x_full, schedule_entry: Optional[Mapping[str, object]]) -> Dict[str, float]:
    components = separated_loss_components(handle, x_full, schedule_entry=schedule_entry)
    return {key: float(value.detach().cpu()) for key, value in components.items()}


def run_record(args: argparse.Namespace) -> int:
    config = load_json(args.config)
    matrix_report = validate_run_matrix(args.config, args.run_matrix)
    if not matrix_report["passed"]:
        raise RuntimeError(f"Run matrix validation failed: {matrix_report['failures']}")
    rows = {row["record_id"]: row for row in load_run_matrix(args.run_matrix)}
    if args.record_id not in rows:
        raise KeyError(f"Unknown record id: {args.record_id}")
    resolved = resolve_run_record(rows[args.record_id], config, config_sha256=file_sha256(args.config))
    release = verify_release_lock(PROJECT_ROOT, args.release_lock_dir, require_clean=True)
    authorization = validate_authorization(
        args.authorization,
        release_commit=str(release["actual_commit"] or release["approved_commit"]),
        config_sha256=file_sha256(args.config),
        matrix_sha256=file_sha256(args.run_matrix),
        phase=str(resolved["phase"]),
        block=str(resolved["block"]),
    )

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Authorized Phase 8 record execution requires CUDA")
    run_dir = args.output_root / "runs" / args.record_id
    complete_path = run_dir / "status.json"
    if args.resume and complete_path.is_file():
        existing = load_json(complete_path)
        if existing.get("status") == "complete":
            print(f"SKIP complete {args.record_id}")
            return 0
    run_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    atomic_json(complete_path, {"status": "running", "record_id": args.record_id, "started_at_unix": started})

    try:
        deterministic = configure_determinism(int(resolved["model_seed"]), device)
        torch.cuda.reset_peak_memory_stats(device)
        x, gc_true, A_true, data_metadata = generate_record_data(resolved)
        handle = move_handle(instantiate_handle(resolved, config), device)
        model = underlying_model(handle)
        schedule = None
        schedule_seed = None
        schedule_hash_value = None
        method = str(resolved["method"])
        if isinstance(handle, CoverageAlignedRawChainJRNGC):
            schedule_seed = int(resolved["jacobian_seed"])
            schedule = build_balanced_lag_schedule(
                T=int(resolved["T"]),
                lag=int(resolved["K"]),
                d_out=int(resolved["d"]),
                max_iter=int(resolved["max_iter"]),
                seed=schedule_seed,
            )
            schedule_hash_value = schedule_sha256(schedule)
        elif isinstance(handle, Phase8NoAuxInputSpaceControl):
            schedule, schedule_seed, schedule_hash_value = build_no_aux_control_schedule(
                handle,
                T=int(resolved["T"]),
                max_iter=int(resolved["max_iter"]),
                model_seed=int(resolved["model_seed"]),
            )

        initial_scale = None
        if resolved["block"] == "repair_scale_benchmark":
            initial_scale = repair_scale_snapshot(handle, x, schedule[0])
        objective_trace = []
        component_trace: Dict[str, list] = {}
        captured_states: Dict[int, Dict[str, torch.Tensor]] = {}
        capture_points = [20] if int(resolved["max_iter"]) >= 20 else []
        if resolved["block"] == "repair_scale_benchmark":
            capture_points = [
                int(value) for value in config["gpu_preflight_reporting_completed_iterations"]
                if 0 < int(value) <= int(resolved["max_iter"])
            ]
        training_started = time.time()
        metadata = train_with_frozen_checkpoint_policy(
            handle,
            x,
            max_iter=int(resolved["max_iter"]),
            checkpoint_policy=str(resolved["checkpoint_policy"]),
            schedule=schedule,
            learning_rate=0.001,
            weight_decay=0.0,
            gradient_clip_norm=1.0,
            check_every=50,
            lookback=10,
            objective_trace=objective_trace,
            capture_completed_iterations=capture_points,
            captured_states=captured_states,
            component_trace=component_trace if resolved["block"] == "repair_scale_benchmark" else None,
        )
        torch.cuda.synchronize(device)
        training_seconds = time.time() - training_started
        model.eval()
        selected_entry = None if schedule is None else schedule[min(metadata.selected_iteration, len(schedule) - 1)]
        separated = final_loss_components(handle, x, selected_entry)
        prediction, raw_target = prediction_and_target(handle, x)
        output_variance = float(np.var(prediction))
        target_variance = float(np.var(raw_target))

        intervention = None
        attribution = None
        graph_metrics = None
        evaluation_seconds = 0.0
        scores_to_save: Dict[str, np.ndarray] = {}
        metrics_payload: Dict[str, object] = {
            **separated,
            "output_variance": output_variance,
            "target_variance": target_variance,
            "output_target_variance_ratio": output_variance / max(target_variance, 1e-12),
        }
        if resolved["block"] == "fixed_target_interventions":
            if isinstance(handle, Phase8NoAuxInputSpaceControl):
                intervention = fixed_target_no_aux_interventions(
                    handle,
                    x,
                    perturbation_seed=int(resolved["perturbation_seed"]),
                )
            else:
                intervention = fixed_target_concat_interventions(
                    handle,
                    x,
                    perturbation_seed=int(resolved["perturbation_seed"]),
                )
            metrics_payload["interventions"] = intervention
        else:
            evaluation_started = time.time()
            edge_count = true_edge_count(gc_true)
            if isinstance(handle, LegacyBaselineAdapter):
                attribution = extract_baseline_attribution_objects(
                    handle,
                    x,
                    true_edge_count=edge_count,
                    n_min=int(rows[args.record_id]["n_min"]),
                )
            else:
                attribution = extract_attribution_objects(
                    handle,
                    x,
                    true_edge_count=edge_count,
                    n_min=int(rows[args.record_id]["n_min"]),
                )
            torch.cuda.synchronize(device)
            evaluation_seconds = time.time() - evaluation_started
            partial_metrics = canonical_metric_adapter(gc_true, attribution.s_partial_nominal)
            total_metrics = canonical_metric_adapter(gc_true, attribution.s_gc_total)
            graph_metrics = {"partial_nominal": partial_metrics, "total_nominal": total_metrics}
            metrics_payload.update({
                "graph_metrics": graph_metrics,
                "coefficient_r_partial_lag1": None if A_true is None else coefficient_r_total_lag1(
                    attribution.j_bar_partial[:, :, 0], A_true
                ),
                "coefficient_r_total_lag1": None if A_true is None else coefficient_r_total_lag1(
                    attribution.j_bar_total_lag1, A_true
                ),
                "temporal_tail_statistics": attribution.temporal_tail_statistics,
                "M_missing": attribution.m_missing,
                "M_missing_undefined_reason": attribution.m_missing_undefined_reason,
                "nominal_partial_total_pearson": attribution.nominal_partial_total_pearson,
                "nominal_pearson_undefined_reason": attribution.nominal_pearson_undefined_reason,
                "nominal_partial_total_topk_jaccard": attribution.nominal_partial_total_topk_jaccard,
            })
            scores_to_save = {
                "j_bar_total": attribution.j_bar_total,
                "j_bar_partial": attribution.j_bar_partial,
                "j_bar_missing": attribution.j_bar_missing,
                "S_partial_nominal": attribution.s_partial_nominal,
                "S_GC_total_nominal": attribution.s_gc_total,
                "J_bar_total_lag1": attribution.j_bar_total_lag1,
                "S_reliable_history": attribution.s_reliable_history,
                "S_prefix_all": attribution.s_prefix_all,
                "prefix_maximizing_lag": attribution.prefix_maximizing_lag,
                "prefix_maximizing_lag_window_count": attribution.prefix_maximizing_lag_window_count,
                "prefix_max_outside_reliable": attribution.prefix_max_outside_reliable,
                "eligible_window_count_by_lag": attribution.eligible_window_count_by_lag,
            }

        final_scale = None
        if resolved["block"] == "repair_scale_benchmark":
            final_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
            scale_by_iteration = {"0": initial_scale}
            for completed_iteration in capture_points:
                model.load_state_dict(captured_states[completed_iteration])
                scale_by_iteration[str(completed_iteration)] = repair_scale_snapshot(
                    handle,
                    x,
                    schedule[completed_iteration - 1],
                )
            model.load_state_dict(final_state)
            final_scale = scale_by_iteration[str(int(resolved["max_iter"]))]
            metrics_payload["regularizer_scale_preflight"] = {
                "initial": initial_scale,
                "final": final_scale,
                "by_completed_iteration": scale_by_iteration,
                "pure_mse_relative_change": (
                    final_scale["fixed_target_prediction_mse"] - initial_scale["fixed_target_prediction_mse"]
                ) / max(initial_scale["fixed_target_prediction_mse"], 1e-12),
            }

        checkpoint = {
            "record_id": args.record_id,
            "model_state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()},
            "training_metadata": metadata.as_dict(),
            "schedule_seed": schedule_seed,
            "schedule_sha256": schedule_hash_value,
        }
        atomic_torch_save(run_dir / "checkpoint.pt", checkpoint)
        if 20 in captured_states:
            atomic_torch_save(run_dir / "checkpoint_iter20.pt", {
                "record_id": args.record_id,
                "completed_iterations": 20,
                "model_state_dict": captured_states[20],
                "objective_trace_first20": objective_trace[:20],
            })
        atomic_npz(run_dir / "predictions_and_targets.npz", predictions=prediction, fixed_raw_targets=raw_target)
        if scores_to_save:
            atomic_npz(run_dir / "scores.npz", **scores_to_save)
        atomic_json(run_dir / "training_trace.json", {
            "total_regularized_objective": objective_trace,
            "separated_components": component_trace,
            "checkpoint_policy": metadata.as_dict(),
        })
        if schedule is not None:
            atomic_json(run_dir / "jacobian_schedule.json", {
                "seed": schedule_seed,
                "sha256": schedule_hash_value,
                "entries": schedule,
            })
        atomic_json(run_dir / "metrics.json", metrics_payload)

        peak_allocated = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        peak_reserved = torch.cuda.max_memory_reserved(device) / (1024 ** 2)
        total_memory = torch.cuda.get_device_properties(device).total_memory / (1024 ** 2)
        resolved_payload = {
            **resolved,
            "source_release": release,
            "authorization_sha256": file_sha256(args.authorization),
            "authorization": authorization,
            "config_sha256": file_sha256(args.config),
            "run_matrix_sha256": file_sha256(args.run_matrix),
            "schedule_seed": schedule_seed,
            "schedule_sha256": schedule_hash_value,
            "data_metadata": data_metadata,
            "deterministic_settings": deterministic,
            "checkpoint_policy_metadata": metadata.as_dict(),
            "execution_device": str(device),
            "environment": {
                "python": sys.version,
                "platform": platform.platform(),
                "numpy": np.__version__,
                "torch": torch.__version__,
                "cuda_runtime": torch.version.cuda,
                "cuda_device_name": torch.cuda.get_device_name(device),
                "cuda_device_capability": list(torch.cuda.get_device_capability(device)),
            },
        }
        atomic_json(run_dir / "resolved_config.json", resolved_payload)
        runtime = {
            "training_seconds": training_seconds,
            "evaluation_seconds": evaluation_seconds,
            "total_seconds": time.time() - started,
            "projected_2000_iteration_training_seconds": training_seconds * 2000 / max(int(resolved["max_iter"]), 1),
            "cuda_peak_memory_allocated_mb": peak_allocated,
            "cuda_peak_memory_reserved_mb": peak_reserved,
            "cuda_total_memory_mb": total_memory,
            "peak_reserved_fraction": peak_reserved / max(total_memory, 1e-12),
        }
        atomic_json(run_dir / "runtime.json", runtime)
        artifacts = {}
        for path in sorted(run_dir.iterdir()):
            if path.is_file() and path.name not in {"artifact_manifest.json", "status.json"}:
                artifacts[path.name] = file_sha256(path)
        atomic_json(run_dir / "artifact_manifest.json", {"files": artifacts})
        atomic_json(complete_path, {
            "status": "complete",
            "record_id": args.record_id,
            "no_nan_inf": all(np.isfinite(value) for value in [
                separated["fixed_target_prediction_mse"],
                separated["jacobian_penalty"],
                separated["total_regularized_objective"],
                output_variance,
                target_variance,
            ]),
            "output_complete": True,
            "formal_result": bool(resolved["formal_result"]),
            "phase": resolved["phase"],
            "block": resolved["block"],
            "method": method,
            "completed_at_unix": time.time(),
        })
        print(json.dumps({"record_id": args.record_id, "status": "complete", "runtime": runtime}, indent=2))
        return 0
    except Exception as exc:
        atomic_json(complete_path, {
            "status": "failed",
            "record_id": args.record_id,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "failed_at_unix": time.time(),
        })
        raise


if __name__ == "__main__":
    raise SystemExit(run_record(parse_args()))
