"""Development-only Lorenz strong-operating-point audit preflight.

This runner uses only the already observed Lorenz-96 seed 0 fixture. Outputs
are non-evidentiary and cannot enter the manuscript. Its purpose is to verify
the dataset contract, baseline operating point, raw-chain audit semantics,
determinism, runtime, and VRAM before a separately frozen 901 protocol.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
for _path in (PROJECT_ROOT, SRC_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from phase8_coverage import Phase8ModelConfig, fixed_target_concat_interventions
from phase9_audit_generalization import (
    build_audit_profile,
    condition_coordinate_mixing_audit,
    deterministic_audit_targets,
    fixed_target_baseline_interventions,
    sampled_raw_chain_audit,
)
from phase9_audit_stageb import (
    horizon_summary,
    make_adapter,
    reset_predictor,
    state_subset_sha256,
    train_legacy_with_metadata,
)
from knowledge_metrics import topk_edges_exact


METHODS = {"baseline", "mamba_concat"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--tag", default="preflight")
    parser.add_argument("--methods", default="baseline,mamba_concat")
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--compare-a", type=Path)
    parser.add_argument("--compare-b", type=Path)
    parser.add_argument("--comparison-output", type=Path)
    return parser.parse_args()


def json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    raise TypeError(type(value).__name__)


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=json_default) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha256(path: Path) -> str:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def git_output(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def load_config(path: Path) -> Dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if payload.get("protocol_name") != "phase9_lorenz_strong_operating_preflight_v1":
        raise RuntimeError("Unexpected preflight protocol")
    boundaries = payload.get("boundaries", {})
    required_false = (
        "formal_result",
        "manuscript_evidence",
        "formal_seed_generation_authorized",
        "autodl_execution_authorized",
    )
    if boundaries.get("development_only") is not True:
        raise RuntimeError("Preflight must be development-only")
    for key in required_false:
        if boundaries.get(key) is not False:
            raise RuntimeError(f"Boundary must remain false: {key}")
    return payload


def load_lorenz_fixture(
    data_root: Path,
    config: Mapping[str, object],
) -> tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    spec = config["dataset"]
    x_path = data_root / str(spec["x_relative_path"])
    graph_path = data_root / str(spec["graph_relative_path"])
    for path, expected in (
        (x_path, str(spec["x_sha256"])),
        (graph_path, str(spec["graph_sha256"])),
    ):
        if not path.is_file():
            raise RuntimeError(f"Missing frozen fixture: {path}")
        actual = file_sha256(path)
        if actual.lower() != expected.lower():
            raise RuntimeError(
                f"Frozen fixture SHA256 mismatch: {path}; "
                f"actual={actual}, expected={expected}"
            )
    x = np.asarray(np.load(x_path, allow_pickle=False), dtype=np.float64)
    graph = np.asarray(np.load(graph_path, allow_pickle=False))
    expected_shape = (int(spec["d"]), int(spec["T"]))
    if x.shape != expected_shape:
        raise RuntimeError(f"Lorenz x shape mismatch: {x.shape} != {expected_shape}")
    if graph.shape != (expected_shape[0], expected_shape[0], int(spec["lag"])):
        raise RuntimeError(f"Lorenz graph shape mismatch: {graph.shape}")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(graph)):
        raise RuntimeError("Nonfinite Lorenz fixture")
    metadata = {
        "name": spec["name"],
        "family": spec["family"],
        "data_seed": int(spec["data_seed"]),
        "d": int(x.shape[0]),
        "T": int(x.shape[1]),
        "lag": int(spec["lag"]),
        "forcing": int(spec["forcing"]),
        "x_path": str(x_path),
        "x_sha256": file_sha256(x_path),
        "graph_path": str(graph_path),
        "graph_sha256": file_sha256(graph_path),
        "x_mean": float(np.mean(x)),
        "x_std": float(np.std(x)),
        "normalization": spec["normalization"],
        "development_seed_previously_observed": True,
        "development_only": True,
        "formal_result": False,
        "manuscript_evidence": False,
    }
    return x, graph, metadata


def make_model_config(
    config: Mapping[str, object],
    *,
    d: int,
) -> Phase8ModelConfig:
    model = config["model"]
    evaluation = config["evaluation"]
    return Phase8ModelConfig(
        d=d,
        lag=int(model["lag"]),
        layers=int(model["layers"]),
        hidden=int(model["hidden"]),
        dropout=float(model["dropout"]),
        d_cond=int(model["d_cond"]),
        d_state=int(model["d_state"]),
        d_conv=int(model["d_conv"]),
        expand=int(model["expand"]),
        jacobian_lam=float(model["jacobian_lam"]),
        attribution_horizon=int(evaluation["primary_horizon"]),
        dtype="float32",
    )


def configure_determinism(seed: int, device: torch.device) -> Dict[str, object]:
    torch.manual_seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats(device)
    return {
        "master_seed": int(seed),
        "torch_deterministic_algorithms": True,
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }


def _all_finite(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, Mapping):
        return all(_all_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_all_finite(item) for item in value)
    if isinstance(value, (float, np.floating)):
        return bool(np.isfinite(value))
    return True


def _save_checkpoint(path: Path, model: torch.nn.Module, metadata: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.pt")
    torch.save(
        {
            "model_state": {
                name: value.detach().cpu()
                for name, value in model.state_dict().items()
            },
            **dict(metadata),
        },
        temporary,
    )
    os.replace(temporary, path)


def evaluate_method_gates(
    *,
    method: str,
    primary_summary: Mapping[str, object],
    primary_arrays: Mapping[str, np.ndarray],
    interventions: Mapping[str, object],
    mixing: Optional[Mapping[str, object]],
    horizon: Optional[Mapping[str, object]],
    gates: Mapping[str, object],
) -> Dict[str, object]:
    checks: Dict[str, bool] = {}
    if method == "baseline":
        total_metrics = primary_summary["total_nominal_metrics"]
        partial_total_max_abs = float(np.max(np.abs(
            primary_arrays["s_total_nominal"]
            - primary_arrays["s_partial_nominal"]
        )))
        missing = primary_summary["missing_route_relative_magnitude"]
        checks = {
            "strong_operating_point": (
                total_metrics is not None
                and float(total_metrics["auroc"])
                >= float(gates["baseline_total_nominal_auroc_min"])
            ),
            "partial_total_identity": (
                partial_total_max_abs
                <= float(gates["baseline_partial_total_max_abs"])
            ),
            "missing_route_zero": (
                missing is not None
                and float(missing) <= float(gates["baseline_missing_route_max"])
            ),
        }
        details = {
            "total_nominal_auroc": (
                None if total_metrics is None else float(total_metrics["auroc"])
            ),
            "partial_total_max_abs": partial_total_max_abs,
            "missing_route_relative_magnitude": missing,
        }
    else:
        pearson = primary_summary["partial_total_nominal_pearson"]
        jaccard = primary_summary["partial_total_nominal_topk_jaccard"]
        missing = primary_summary["missing_route_relative_magnitude"]
        tail = primary_summary["temporal_tail_statistics"]["median"]
        mask_delta = interventions["fixed_target_prediction_mse_delta"]["mask_c"]
        entropy = mixing["normalized_source_entropy"]["median"] if mixing else None
        ratios = horizon["offdiagonal_cumulative_mass_ratio_vs_H128"] if horizon else {}
        nominal = horizon["nominal_score_max_abs_difference_vs_H128"] if horizon else {}
        discrepancy = (
            (pearson is not None and float(pearson) <= float(gates["partial_total_pearson_max"]))
            or (jaccard is not None and float(jaccard) <= float(gates["topk_jaccard_max"]))
        )
        checks = {
            "missing_route": (
                missing is not None and float(missing) >= float(gates["missing_route_min"])
            ),
            "mask_c": float(mask_delta) >= float(gates["mask_c_delta_min"]),
            "partial_total_discrepancy": discrepancy,
            "coordinate_entropy": (
                entropy is not None
                and float(entropy) >= float(gates["coordinate_entropy_median_min"])
            ),
            "temporal_tail": (
                tail is not None
                and float(tail) >= float(gates["temporal_tail_median_min"])
            ),
            "h64_h128_mass": (
                ratios.get("64") is not None
                and float(ratios["64"]) >= float(gates["h64_h128_ratio_min"])
            ),
            "nominal_horizon_stability": (
                float(nominal.get("64", float("inf")))
                <= float(gates["nominal_horizon_max_abs"])
            ),
        }
        details = {
            "missing_route_relative_magnitude": missing,
            "mask_c_fixed_target_mse_delta": float(mask_delta),
            "partial_total_nominal_pearson": pearson,
            "partial_total_nominal_topk_jaccard": jaccard,
            "coordinate_entropy_median": entropy,
            "temporal_tail_median": tail,
            "h64_h128_mass_ratio": ratios.get("64"),
            "nominal_h64_h128_max_abs": nominal.get("64"),
        }
    return {
        "method": method,
        "checks": checks,
        "details": details,
        "passed": all(checks.values()),
        "development_only": True,
        "formal_result": False,
        "manuscript_evidence": False,
    }


def run_one(
    *,
    method: str,
    tag: str,
    iterations: int,
    config_path: Path,
    config: Mapping[str, object],
    x: np.ndarray,
    graph: np.ndarray,
    dataset_metadata: Mapping[str, object],
    output_root: Path,
    device: torch.device,
    resume: bool,
) -> Dict[str, object]:
    run_id = f"{tag}__lorenz_f40_seed0__{method}"
    run_dir = output_root / "runs" / run_id
    status_path = run_dir / "status.json"
    if resume and status_path.is_file():
        existing = json.loads(status_path.read_text(encoding="utf-8"))
        if existing.get("status") == "complete":
            return existing
    if run_dir.exists() and any(run_dir.iterdir()):
        raise RuntimeError(f"Refusing nonempty run directory: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    atomic_json(
        status_path,
        {
            "status": "running",
            "run_id": run_id,
            "development_only": True,
            "formal_result": False,
            "manuscript_evidence": False,
        },
    )

    cfg = make_model_config(config, d=x.shape[0])
    seeds = config["seeds"]
    predictor_seed = int(seeds["predictor_seed"])
    preprocessor_seed = (
        None if method == "baseline" else int(seeds["preprocessor_seed"])
    )
    deterministic = configure_determinism(int(seeds["master_seed"]), device)
    adapter = make_adapter(
        method,
        cfg,
        predictor_seed=predictor_seed,
        preprocessor_seed=preprocessor_seed,
    )
    predictor_initial_sha = state_subset_sha256(
        adapter.model,
        prefix_excluded="preprocessor",
    )
    adapter.model.to(device)

    training_cfg = config["training"]
    train_started = time.perf_counter()
    training = train_legacy_with_metadata(
        adapter.model,
        x,
        max_iter=iterations,
        learning_rate=float(training_cfg["learning_rate"]),
        weight_decay=float(training_cfg["weight_decay"]),
        gradient_clip_norm=float(training_cfg["gradient_clip_norm"]),
        check_every=int(training_cfg["check_every"]),
        lookback=int(training_cfg["lookback"]),
    )
    training_seconds = time.perf_counter() - train_started

    evaluation_cfg = config["evaluation"]
    target_indices = deterministic_audit_targets(
        T=x.shape[1],
        lag=cfg.lag,
        attribution_horizon=int(evaluation_cfg["common_target_minimum_horizon"]),
        count=int(evaluation_cfg["audit_window_count"]),
        seed=int(seeds["score_window_seed"]),
    )
    eval_started = time.perf_counter()
    if method == "baseline":
        primary = sampled_raw_chain_audit(
            adapter,
            x,
            target_indices=target_indices,
            attribution_horizon=int(evaluation_cfg["primary_horizon"]),
            graph=graph,
        )
        audits = {64: primary}
        horizon = None
        interventions = fixed_target_baseline_interventions(
            adapter,
            x,
            perturbation_seed=int(seeds["perturbation_seed"]),
        )
        mixing = None
    else:
        audits = {
            horizon_value: sampled_raw_chain_audit(
                adapter,
                x,
                target_indices=target_indices,
                attribution_horizon=horizon_value,
                graph=graph,
            )
            for horizon_value in evaluation_cfg["horizon_sensitivity"]
        }
        primary = audits[int(evaluation_cfg["primary_horizon"])]
        horizon = horizon_summary(audits)
        interventions = fixed_target_concat_interventions(
            adapter,
            x,
            perturbation_seed=int(seeds["perturbation_seed"]),
        )
        mixing = condition_coordinate_mixing_audit(
            adapter,
            x,
            target_indices=target_indices,
            attribution_horizon=int(evaluation_cfg["primary_horizon"]),
        )
    evaluation_seconds = time.perf_counter() - eval_started

    primary_summary = primary.summary()
    primary_arrays = primary.arrays()
    gates = evaluate_method_gates(
        method=method,
        primary_summary=primary_summary,
        primary_arrays=primary_arrays,
        interventions=interventions,
        mixing=mixing,
        horizon=horizon,
        gates=config["gates"],
    )
    profile = build_audit_profile(
        architecture=(
            "baseline_jrngc"
            if method == "baseline"
            else "legacy_mamba_concat_jrngc"
        ),
        sampled_audit=primary,
        has_auxiliary_route=method != "baseline",
    )
    profile.update({
        "development_only": True,
        "formal_result": False,
        "manuscript_evidence": False,
    })
    pure_mse = float(adapter.pure_mse(x).detach().cpu())
    payloads = {
        "training": training,
        "primary": primary_summary,
        "interventions": interventions,
        "mixing": mixing,
        "horizon": horizon,
        "pure_mse": pure_mse,
        "gates": gates,
    }
    if not _all_finite(payloads):
        raise RuntimeError(f"Nonfinite output: {run_id}")

    record_config = {
        "protocol_name": config["protocol_name"],
        "config_path": str(config_path),
        "config_canonical_sha256": canonical_json_sha256(config_path),
        "git_commit": git_output("rev-parse", "HEAD"),
        "git_worktree_clean_at_start": git_output("status", "--porcelain") == "",
        "run_id": run_id,
        "tag": tag,
        "method": method,
        "iterations": int(iterations),
        "model": asdict(cfg),
        "target_indices": target_indices.tolist(),
        "predictor_initial_sha256": predictor_initial_sha,
        "determinism": deterministic,
        "development_only": True,
        "formal_result": False,
        "manuscript_evidence": False,
    }
    atomic_json(run_dir / "config.json", record_config)
    atomic_json(run_dir / "dataset_metadata.json", dataset_metadata)
    atomic_json(run_dir / "training.json", training)
    atomic_json(run_dir / "sampled_attribution_audit.json", primary_summary)
    atomic_npz(run_dir / "sampled_attribution_objects.npz", **primary_arrays)
    for horizon_value, audit in audits.items():
        atomic_json(
            run_dir / f"sampled_attribution_audit_H{horizon_value}.json",
            audit.summary(),
        )
        atomic_npz(
            run_dir / f"sampled_attribution_objects_H{horizon_value}.npz",
            **audit.arrays(),
        )
    atomic_json(run_dir / "horizon_sensitivity.json", horizon)
    atomic_json(run_dir / "fixed_target_interventions.json", interventions)
    atomic_json(run_dir / "coordinate_mixing_audit.json", mixing)
    atomic_json(run_dir / "audit_profile.json", profile)
    atomic_json(run_dir / "gate_report.json", gates)
    _save_checkpoint(
        run_dir / "checkpoint.pt",
        adapter.model,
        {
            "run_id": run_id,
            "model_config": asdict(cfg),
            "predictor_initial_sha256": predictor_initial_sha,
            "development_only": True,
            "formal_result": False,
            "manuscript_evidence": False,
        },
    )
    status = {
        "status": "complete",
        "run_id": run_id,
        "method": method,
        "iterations": int(iterations),
        "selected_iteration": int(training["selected_iteration"]),
        "best_total_regularized_objective": float(
            training["best_total_regularized_objective"]
        ),
        "fixed_target_prediction_mse": pure_mse,
        "predictor_initial_sha256": predictor_initial_sha,
        "gate_passed": bool(gates["passed"]),
        "training_seconds": training_seconds,
        "evaluation_seconds": evaluation_seconds,
        "wall_seconds": time.perf_counter() - started,
        "cuda_max_memory_allocated_mb": (
            float(torch.cuda.max_memory_allocated(device) / 1024**2)
            if device.type == "cuda"
            else None
        ),
        "device": str(device),
        "cuda_device_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else None
        ),
        "hostname": platform.node(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "config_canonical_sha256": canonical_json_sha256(config_path),
        "git_commit": git_output("rev-parse", "HEAD"),
        "no_nan_inf": True,
        "development_only": True,
        "formal_result": False,
        "manuscript_evidence": False,
    }
    atomic_json(status_path, status)
    return status


def _nested_numeric_max_abs(left: object, right: object) -> float:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        common = set(left) & set(right)
        values = [_nested_numeric_max_abs(left[key], right[key]) for key in common]
        return max(values, default=0.0)
    if isinstance(left, Sequence) and not isinstance(left, (str, bytes)):
        if not isinstance(right, Sequence) or len(left) != len(right):
            return float("inf")
        return max(
            (_nested_numeric_max_abs(a, b) for a, b in zip(left, right)),
            default=0.0,
        )
    if (
        isinstance(left, (int, float, np.integer, np.floating))
        and not isinstance(left, bool)
        and isinstance(right, (int, float, np.integer, np.floating))
        and not isinstance(right, bool)
    ):
        return abs(float(left) - float(right))
    return 0.0 if left == right else float("inf")


def _checkpoint_max_abs(path_a: Path, path_b: Path) -> float:
    left = torch.load(path_a, map_location="cpu", weights_only=False)["model_state"]
    right = torch.load(path_b, map_location="cpu", weights_only=False)["model_state"]
    if set(left) != set(right):
        return float("inf")
    return max(
        (
            float(torch.max(torch.abs(left[name] - right[name])))
            for name in left
        ),
        default=0.0,
    )


def compare_runs(
    run_a: Path,
    run_b: Path,
    *,
    threshold: float,
) -> Dict[str, object]:
    status_a = json.loads((run_a / "status.json").read_text(encoding="utf-8"))
    status_b = json.loads((run_b / "status.json").read_text(encoding="utf-8"))
    config_a = json.loads((run_a / "config.json").read_text(encoding="utf-8"))
    config_b = json.loads((run_b / "config.json").read_text(encoding="utf-8"))
    training_a = json.loads((run_a / "training.json").read_text(encoding="utf-8"))
    training_b = json.loads((run_b / "training.json").read_text(encoding="utf-8"))
    audit_a = json.loads(
        (run_a / "sampled_attribution_audit.json").read_text(encoding="utf-8")
    )
    audit_b = json.loads(
        (run_b / "sampled_attribution_audit.json").read_text(encoding="utf-8")
    )
    arrays_a = np.load(run_a / "sampled_attribution_objects.npz")
    arrays_b = np.load(run_b / "sampled_attribution_objects.npz")
    array_differences = {
        key: float(np.max(np.abs(arrays_a[key] - arrays_b[key])))
        for key in arrays_a.files
        if key != "target_indices"
    }
    target_indices_equal = np.array_equal(
        arrays_a["target_indices"],
        arrays_b["target_indices"],
    )
    score_a = arrays_a["s_total_nominal"]
    score_b = arrays_b["s_total_nominal"]
    graph_path = Path(
        json.loads((run_a / "dataset_metadata.json").read_text(encoding="utf-8"))[
            "graph_path"
        ]
    )
    graph = np.asarray(np.load(graph_path, allow_pickle=False))
    graph_2d = np.any(graph != 0, axis=2) if graph.ndim == 3 else graph != 0
    graph_2d = graph_2d.copy()
    np.fill_diagonal(graph_2d, 0)
    k = int(np.sum(graph_2d))
    edges_equal = topk_edges_exact(score_a, k) == topk_edges_exact(score_b, k)
    checks = {
        "same_method": status_a["method"] == status_b["method"] == "mamba_concat",
        "same_iterations": status_a["iterations"] == status_b["iterations"],
        "same_config_hash": (
            status_a["config_canonical_sha256"]
            == status_b["config_canonical_sha256"]
        ),
        "same_predictor_initialization": (
            status_a["predictor_initial_sha256"]
            == status_b["predictor_initial_sha256"]
        ),
        "same_target_indices": target_indices_equal,
        "same_model_config": config_a["model"] == config_b["model"],
        "training_trace": (
            _nested_numeric_max_abs(training_a["trace"], training_b["trace"])
            <= threshold
        ),
        "checkpoint_state": (
            _checkpoint_max_abs(
                run_a / "checkpoint.pt",
                run_b / "checkpoint.pt",
            )
            <= threshold
        ),
        "attribution_arrays": max(array_differences.values(), default=0.0) <= threshold,
        "audit_metrics": _nested_numeric_max_abs(audit_a, audit_b) <= threshold,
        "exact_topk_edges": edges_equal,
    }
    return {
        "run_a": str(run_a),
        "run_b": str(run_b),
        "threshold": float(threshold),
        "checks": checks,
        "training_trace_max_abs": _nested_numeric_max_abs(
            training_a["trace"],
            training_b["trace"],
        ),
        "checkpoint_state_max_abs": _checkpoint_max_abs(
            run_a / "checkpoint.pt",
            run_b / "checkpoint.pt",
        ),
        "attribution_array_max_abs": array_differences,
        "audit_metric_max_abs": _nested_numeric_max_abs(audit_a, audit_b),
        "exact_topk_edges_equal": edges_equal,
        "passed": all(checks.values()),
        "development_only": True,
        "formal_result": False,
        "manuscript_evidence": False,
    }


def main() -> int:
    args = parse_args()
    if args.compare_a or args.compare_b:
        if not args.compare_a or not args.compare_b or not args.comparison_output:
            raise RuntimeError(
                "Comparison requires --compare-a, --compare-b, and --comparison-output"
            )
        config = load_config(args.config.resolve())
        report = compare_runs(
            args.compare_a.resolve(),
            args.compare_b.resolve(),
            threshold=float(config["gates"]["determinism_max_abs"]),
        )
        atomic_json(args.comparison_output.resolve(), report)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if report["passed"] else 2

    config_path = args.config.resolve()
    config = load_config(config_path)
    requested_methods = [
        value.strip() for value in args.methods.split(",") if value.strip()
    ]
    if not requested_methods or not set(requested_methods).issubset(METHODS):
        raise RuntimeError(f"Invalid methods: {requested_methods}")
    if args.iterations <= 0:
        raise RuntimeError("iterations must be positive")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)
    x, graph, dataset_metadata = load_lorenz_fixture(
        args.data_root.resolve(),
        config,
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    environment = {
        "protocol_name": config["protocol_name"],
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
        "git_commit": git_output("rev-parse", "HEAD"),
        "git_status_porcelain": git_output("status", "--porcelain"),
        "config_canonical_sha256": canonical_json_sha256(config_path),
        "development_only": True,
        "formal_result": False,
        "manuscript_evidence": False,
    }
    atomic_json(args.output_root / "environment.json", environment)
    atomic_json(args.output_root / "config_snapshot.json", config)
    statuses = []
    for method in requested_methods:
        statuses.append(run_one(
            method=method,
            tag=args.tag,
            iterations=int(args.iterations),
            config_path=config_path,
            config=config,
            x=x,
            graph=graph,
            dataset_metadata=dataset_metadata,
            output_root=args.output_root.resolve(),
            device=device,
            resume=args.resume,
        ))
    summary = {
        "tag": args.tag,
        "iterations": int(args.iterations),
        "methods": requested_methods,
        "runs": statuses,
        "all_complete": all(row.get("status") == "complete" for row in statuses),
        "all_finite": all(row.get("no_nan_inf") is True for row in statuses),
        "all_method_gates_passed": all(row.get("gate_passed") is True for row in statuses),
        "total_wall_seconds": float(sum(row["wall_seconds"] for row in statuses)),
        "development_only": True,
        "formal_result": False,
        "manuscript_evidence": False,
    }
    atomic_json(args.output_root / f"{args.tag}_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
