"""Release-locked Phase 9 Lorenz-96 confirmation runner.

The formal run matrix contains only baseline JRNGC and the frozen legacy
Mamba-concat x-only architecture. The study evaluates score-route coverage at
a strong known-graph operating point; it does not test a new repair method.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import shutil
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence

import numpy as np
import scipy
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
for _path in (PROJECT_ROOT, SRC_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from lorenz96_frozen import generate_lorenz96, lorenz96_direct_graph
from phase8_coverage import fixed_target_concat_interventions
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
    state_subset_sha256,
    train_legacy_with_metadata,
)
from phase9_lorenz_strong_operating_preflight import (
    _all_finite,
    _save_checkpoint,
    atomic_json,
    atomic_npz,
    canonical_json_sha256,
    compare_runs,
    configure_determinism,
    evaluate_method_gates,
    file_sha256,
    git_output,
    make_model_config,
)


METHODS = ("baseline", "mamba_concat")
PROTOCOL_NAME = "phase9_lorenz_901_confirmation_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=(
        "dry-run",
        "generate-data",
        "smoke",
        "validate-smoke",
        "formal",
    ))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--release-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--tag", default="formal")
    parser.add_argument("--methods", default="baseline,mamba_concat")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke-root-a", type=Path)
    parser.add_argument("--smoke-root-b", type=Path)
    return parser.parse_args()


def load_protocol(path: Path) -> Dict[str, object]:
    config = json.loads(path.read_text(encoding="utf-8-sig"))
    if config.get("protocol_name") != PROTOCOL_NAME:
        raise RuntimeError("Unexpected confirmation protocol")
    boundaries = config.get("boundaries", {})
    if boundaries.get("formal_result") is not True:
        raise RuntimeError("Formal-result boundary is not locked")
    if boundaries.get("manuscript_evidence_pending_aggregate") is not True:
        raise RuntimeError("Manuscript evidence must remain pending aggregation")
    if boundaries.get("phase7_stage1b_authorized") is not False:
        raise RuntimeError("Phase 7 Stage 1b must remain unauthorized")
    if boundaries.get("new_method_training_authorized") is not False:
        raise RuntimeError("New-method training must remain unauthorized")
    return config


def load_run_matrix(
    config: Mapping[str, object],
) -> list[Dict[str, object]]:
    design = config["formal_design"]
    matrix_path = PROJECT_ROOT / str(design["run_matrix_file"])
    if not matrix_path.is_file():
        raise RuntimeError(f"Missing run matrix: {matrix_path}")
    if file_sha256(matrix_path) != str(design["run_matrix_sha256"]):
        raise RuntimeError("Run-matrix SHA256 mismatch")
    with matrix_path.open("r", encoding="utf-8-sig", newline="") as handle:
        records = list(csv.DictReader(handle))
    parsed = []
    for row in records:
        parsed.append({
            "run_index": int(row["run_index"]),
            "formal_result": row["formal_result"].lower() == "true",
            "manuscript_evidence_pending": (
                row["manuscript_evidence_pending"].lower() == "true"
            ),
            "data_seed": int(row["data_seed"]),
            "train_seed": int(row["train_seed"]),
            "method": row["method"],
            "iterations": int(row["iterations"]),
        })
    expected = {
        (int(data_seed), int(train_seed), method)
        for data_seed in design["data_seeds"]
        for train_seed in design["train_seeds"]
        for method in design["methods"]
    }
    actual = {
        (row["data_seed"], row["train_seed"], row["method"])
        for row in parsed
    }
    if actual != expected or len(parsed) != int(design["run_count"]):
        raise RuntimeError("Run matrix does not equal the frozen Cartesian product")
    if [row["run_index"] for row in parsed] != list(range(1, len(parsed) + 1)):
        raise RuntimeError("Run-matrix indices are not contiguous")
    if any(
        row["method"] not in METHODS
        or row["iterations"] != int(design["iterations"])
        or row["formal_result"] is not True
        or row["manuscript_evidence_pending"] is not True
        for row in parsed
    ):
        raise RuntimeError("Invalid formal run-matrix record")
    return parsed


def load_release_manifest(path: Path) -> Dict[str, object]:
    manifest = json.loads(path.read_text(encoding="utf-8-sig"))
    required = {
        "approved_commit",
        "config_sha256",
        "canonical_config_sha256",
        "run_matrix_sha256",
        "critical_files",
        "source_manifest_sha256",
    }
    if not required.issubset(manifest):
        raise RuntimeError("Incomplete release manifest")
    return manifest


def validate_release_lock(
    *,
    config_path: Path,
    release_manifest_path: Path,
    config: Mapping[str, object],
) -> Dict[str, object]:
    manifest = load_release_manifest(release_manifest_path)
    actual_commit = git_output("rev-parse", "HEAD")
    status = git_output("status", "--porcelain")
    checks = {
        "commit": actual_commit == manifest["approved_commit"],
        "clean_worktree": status == "",
        "config_sha256": (
            file_sha256(config_path) == manifest["config_sha256"]
        ),
        "canonical_config_sha256": (
            canonical_json_sha256(config_path)
            == manifest["canonical_config_sha256"]
        ),
        "run_matrix_sha256": (
            str(config["formal_design"]["run_matrix_sha256"])
            == manifest["run_matrix_sha256"]
        ),
    }
    critical_actual = {}
    for relative_path, expected_hash in manifest["critical_files"].items():
        path = PROJECT_ROOT / relative_path
        actual_hash = file_sha256(path) if path.is_file() else None
        critical_actual[relative_path] = actual_hash
        checks[f"critical:{relative_path}"] = actual_hash == expected_hash
    if not all(checks.values()):
        failed = [key for key, passed in checks.items() if not passed]
        raise RuntimeError(f"Release-lock failure: {failed}")
    return {
        "release_lock_mode": "git_commit_and_source_manifest",
        "approved_commit": manifest["approved_commit"],
        "actual_commit": actual_commit,
        "clean_worktree": True,
        "config_sha256": manifest["config_sha256"],
        "canonical_config_sha256": manifest["canonical_config_sha256"],
        "run_matrix_sha256": manifest["run_matrix_sha256"],
        "source_manifest_sha256": manifest["source_manifest_sha256"],
        "critical_files": critical_actual,
        "checks": checks,
        "passed": True,
    }


def write_traceability_snapshots(
    *,
    output_root: Path,
    config: Mapping[str, object],
    release_lock: Mapping[str, object],
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    atomic_json(output_root / "config_snapshot.json", config)
    atomic_json(output_root / "release_lock.json", release_lock)
    source = PROJECT_ROOT / str(config["formal_design"]["run_matrix_file"])
    target = output_root / "run_matrix_snapshot.csv"
    if target.exists():
        if file_sha256(target) != file_sha256(source):
            raise RuntimeError("Existing run-matrix snapshot mismatch")
    else:
        temporary = target.with_suffix(".tmp.csv")
        shutil.copyfile(source, temporary)
        os.replace(temporary, target)


def dataset_directory(output_root: Path, data_seed: int) -> Path:
    return output_root / "datasets" / f"data_seed_{data_seed}"


def generate_dataset(
    *,
    output_root: Path,
    config: Mapping[str, object],
    data_seed: int,
    formal_result: bool,
) -> Dict[str, object]:
    directory = dataset_directory(output_root, data_seed)
    x_path = directory / "_x.npy"
    graph_path = directory / "_gc.npy"
    metadata_path = directory / "metadata.json"
    if any(path.exists() for path in (x_path, graph_path, metadata_path)):
        if not all(path.is_file() for path in (x_path, graph_path, metadata_path)):
            raise RuntimeError(f"Partial dataset artifact: {directory}")
        existing = json.loads(metadata_path.read_text(encoding="utf-8"))
        if (
            file_sha256(x_path) != existing.get("x_sha256")
            or file_sha256(graph_path) != existing.get("graph_sha256")
            or int(existing.get("data_seed", -1)) != int(data_seed)
        ):
            raise RuntimeError(f"Existing dataset integrity failure: {directory}")
        return existing

    spec = config["dataset"]
    x, x_eval, graph = generate_lorenz96(
        d=int(spec["d"]),
        t=int(spec["T"]),
        t_eval=int(spec["t_eval"]),
        forcing=float(spec["forcing"]),
        seed=int(data_seed),
        delta_t=float(spec["delta_t"]),
        observation_noise_sd=float(spec["observation_noise_sd"]),
        burn_in=int(spec["burn_in"]),
    )
    if x_eval.shape[1] != int(spec["t_eval"]):
        raise RuntimeError("Unexpected Lorenz evaluation shape")
    expected_graph = lorenz96_direct_graph(int(spec["d"]))
    if not np.array_equal(graph, expected_graph):
        raise RuntimeError("Lorenz direct-graph support mismatch")
    if x.shape != (int(spec["d"]), int(spec["T"])):
        raise RuntimeError(f"Unexpected Lorenz data shape: {x.shape}")
    if not np.all(np.isfinite(x)):
        raise RuntimeError("Nonfinite generated Lorenz data")
    variable_means = np.mean(x, axis=1)
    variable_stds = np.std(x, axis=1)
    if (
        float(np.max(np.abs(variable_means))) > 1e-5
        or float(np.max(np.abs(variable_stds - 1.0))) > 1e-5
    ):
        raise RuntimeError("Generated Lorenz normalization gate failed")

    directory.mkdir(parents=True, exist_ok=False)
    np.save(x_path, x, allow_pickle=False)
    np.save(graph_path, graph, allow_pickle=False)
    metadata = {
        "family": spec["family"],
        "data_seed": int(data_seed),
        "d": int(spec["d"]),
        "T": int(spec["T"]),
        "lag": int(spec["lag"]),
        "forcing": float(spec["forcing"]),
        "delta_t": float(spec["delta_t"]),
        "observation_noise_sd": float(spec["observation_noise_sd"]),
        "burn_in": int(spec["burn_in"]),
        "normalization": spec["normalization"],
        "x_path": str(x_path.resolve()),
        "x_sha256": file_sha256(x_path),
        "graph_path": str(graph_path.resolve()),
        "graph_sha256": file_sha256(graph_path),
        "graph_nonzero_including_diagonal": int(np.sum(graph != 0)),
        "graph_nonzero_off_diagonal": int(
            np.sum(graph[:, :, 0] != 0) - np.trace(graph[:, :, 0] != 0)
        ),
        "variable_mean_max_abs": float(np.max(np.abs(variable_means))),
        "variable_std_max_abs_difference_from_one": float(
            np.max(np.abs(variable_stds - 1.0))
        ),
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "generator_source_sha256": file_sha256(
            PROJECT_ROOT / "src" / "lorenz96_frozen.py"
        ),
        "formal_result": bool(formal_result),
        "manuscript_evidence": False,
        "manuscript_evidence_pending_aggregate": bool(formal_result),
    }
    atomic_json(metadata_path, metadata)
    return metadata


def load_dataset(
    *,
    output_root: Path,
    data_seed: int,
) -> tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    directory = dataset_directory(output_root, data_seed)
    metadata = json.loads(
        (directory / "metadata.json").read_text(encoding="utf-8")
    )
    x_path = directory / "_x.npy"
    graph_path = directory / "_gc.npy"
    if (
        file_sha256(x_path) != metadata["x_sha256"]
        or file_sha256(graph_path) != metadata["graph_sha256"]
    ):
        raise RuntimeError(f"Dataset hash failure: {directory}")
    x = np.asarray(np.load(x_path, allow_pickle=False), dtype=np.float64)
    graph = np.asarray(np.load(graph_path, allow_pickle=False))
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(graph)):
        raise RuntimeError(f"Nonfinite dataset: {directory}")
    return x, graph, metadata


def run_identifier(
    *,
    prefix: str,
    run_index: Optional[int],
    data_seed: int,
    train_seed: int,
    method: str,
) -> str:
    index = "smoke" if run_index is None else f"r{run_index:03d}"
    return (
        f"{prefix}__{index}__lorenz_f40"
        f"__dseed{data_seed}__tseed{train_seed}__{method}"
    )


def method_seed_bundle(
    *,
    config: Mapping[str, object],
    data_seed: int,
    train_seed: int,
) -> Dict[str, int]:
    policy = config["seed_policy"]
    return {
        "master_seed": int(train_seed),
        "predictor_seed": int(train_seed),
        "preprocessor_seed": int(train_seed) + 1_000_000,
        "score_window_seed": int(
            policy["score_window_seed_by_data_seed"].get(
                str(data_seed),
                int(data_seed) + 1_000_000,
            )
        ),
        "perturbation_seed": int(
            policy["perturbation_seed_by_data_seed"].get(
                str(data_seed),
                int(data_seed) + 2_000_000,
            )
        ),
    }


def run_one(
    *,
    run_index: Optional[int],
    prefix: str,
    method: str,
    iterations: int,
    data_seed: int,
    train_seed: int,
    formal_result: bool,
    config_path: Path,
    config: Mapping[str, object],
    release_lock: Mapping[str, object],
    output_root: Path,
    device: torch.device,
    resume: bool,
) -> Dict[str, object]:
    run_id = run_identifier(
        prefix=prefix,
        run_index=run_index,
        data_seed=data_seed,
        train_seed=train_seed,
        method=method,
    )
    run_dir = output_root / "runs" / run_id
    status_path = run_dir / "status.json"
    if resume and status_path.is_file():
        existing = json.loads(status_path.read_text(encoding="utf-8"))
        if existing.get("status") == "complete":
            return existing
    if run_dir.exists() and any(run_dir.iterdir()):
        raise RuntimeError(f"Refusing nonempty run directory: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(status_path, {
        "status": "running",
        "run_id": run_id,
        "formal_result": bool(formal_result),
        "manuscript_evidence": False,
        "manuscript_evidence_pending_aggregate": bool(formal_result),
    })
    started = time.perf_counter()

    x, graph, dataset_metadata = load_dataset(
        output_root=output_root,
        data_seed=data_seed,
    )
    cfg = make_model_config(config, d=x.shape[0])
    seeds = method_seed_bundle(
        config=config,
        data_seed=data_seed,
        train_seed=train_seed,
    )
    deterministic = configure_determinism(seeds["master_seed"], device)
    adapter = make_adapter(
        method,
        cfg,
        predictor_seed=seeds["predictor_seed"],
        preprocessor_seed=(
            None if method == "baseline" else seeds["preprocessor_seed"]
        ),
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
        max_iter=int(iterations),
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
        seed=seeds["score_window_seed"],
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
        audits = {int(evaluation_cfg["primary_horizon"]): primary}
        horizon = None
        interventions = fixed_target_baseline_interventions(
            adapter,
            x,
            perturbation_seed=seeds["perturbation_seed"],
        )
        mixing = None
    else:
        audits = {
            int(horizon_value): sampled_raw_chain_audit(
                adapter,
                x,
                target_indices=target_indices,
                attribution_horizon=int(horizon_value),
                graph=graph,
            )
            for horizon_value in evaluation_cfg["horizon_sensitivity"]
        }
        primary = audits[int(evaluation_cfg["primary_horizon"])]
        horizon = horizon_summary(audits)
        interventions = fixed_target_concat_interventions(
            adapter,
            x,
            perturbation_seed=seeds["perturbation_seed"],
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
    gate_report = evaluate_method_gates(
        method=method,
        primary_summary=primary_summary,
        primary_arrays=primary_arrays,
        interventions=interventions,
        mixing=mixing,
        horizon=horizon,
        gates=config["per_run_diagnostic_gates"],
    )
    gate_report.update({
        "formal_result": bool(formal_result),
        "development_only": not bool(formal_result),
        "manuscript_evidence": False,
        "manuscript_evidence_pending_aggregate": bool(formal_result),
        "gate_role": "per-run diagnostic; aggregate protocol decides evidence eligibility",
    })
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
        "formal_result": bool(formal_result),
        "development_only": not bool(formal_result),
        "manuscript_evidence": False,
        "manuscript_evidence_pending_aggregate": bool(formal_result),
    })
    pure_mse = float(adapter.pure_mse(x).detach().cpu())
    payloads = {
        "training": training,
        "primary": primary_summary,
        "interventions": interventions,
        "mixing": mixing,
        "horizon": horizon,
        "pure_mse": pure_mse,
        "gate_report": gate_report,
    }
    if not _all_finite(payloads):
        raise RuntimeError(f"Nonfinite output: {run_id}")

    record_config = {
        "protocol_name": config["protocol_name"],
        "config_path": str(config_path),
        "config_file_sha256": file_sha256(config_path),
        "config_canonical_sha256": canonical_json_sha256(config_path),
        "run_id": run_id,
        "run_index": run_index,
        "prefix": prefix,
        "method": method,
        "iterations": int(iterations),
        "data_seed": int(data_seed),
        "train_seed": int(train_seed),
        "seed_bundle": seeds,
        "model": asdict(cfg),
        "target_indices": target_indices.tolist(),
        "predictor_initial_sha256": predictor_initial_sha,
        "determinism": deterministic,
        "checkpoint_policy": config["training"]["checkpoint_policy"],
        "gating_checkpoint": config["training"]["gating_checkpoint"],
        "release_lock": release_lock,
        "formal_result": bool(formal_result),
        "development_only": not bool(formal_result),
        "manuscript_evidence": False,
        "manuscript_evidence_pending_aggregate": bool(formal_result),
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
    atomic_json(run_dir / "gate_report.json", gate_report)
    _save_checkpoint(
        run_dir / "checkpoint.pt",
        adapter.model,
        {
            "run_id": run_id,
            "run_index": run_index,
            "model_config": asdict(cfg),
            "data_seed": int(data_seed),
            "train_seed": int(train_seed),
            "predictor_initial_sha256": predictor_initial_sha,
            "release_commit": release_lock["approved_commit"],
            "formal_result": bool(formal_result),
            "manuscript_evidence": False,
            "manuscript_evidence_pending_aggregate": bool(formal_result),
        },
    )
    status = {
        "status": "complete",
        "run_id": run_id,
        "run_index": run_index,
        "method": method,
        "data_seed": int(data_seed),
        "train_seed": int(train_seed),
        "iterations": int(iterations),
        "selected_iteration": int(training["selected_iteration"]),
        "best_total_regularized_objective": float(
            training["best_total_regularized_objective"]
        ),
        "fixed_target_prediction_mse": pure_mse,
        "predictor_initial_sha256": predictor_initial_sha,
        "per_run_diagnostic_gate_passed": bool(gate_report["passed"]),
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
        "release_commit": release_lock["approved_commit"],
        "source_manifest_sha256": release_lock["source_manifest_sha256"],
        "no_nan_inf": True,
        "formal_result": bool(formal_result),
        "development_only": not bool(formal_result),
        "manuscript_evidence": False,
        "manuscript_evidence_pending_aggregate": bool(formal_result),
    }
    atomic_json(status_path, status)
    return status


def environment_record(
    *,
    config_path: Path,
    release_lock: Mapping[str, object],
    device: torch.device,
    formal_result: bool,
) -> Dict[str, object]:
    return {
        "protocol_name": PROTOCOL_NAME,
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": sys.version,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "device": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else None
        ),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "config_file_sha256": file_sha256(config_path),
        "config_canonical_sha256": canonical_json_sha256(config_path),
        "release_lock": release_lock,
        "formal_result": bool(formal_result),
        "development_only": not bool(formal_result),
        "manuscript_evidence": False,
    }


def validate_smoke(
    *,
    root_a: Path,
    root_b: Path,
    config: Mapping[str, object],
    output_path: Path,
) -> Dict[str, object]:
    train_seed = int(config["development_smoke"]["train_seed"])
    baseline_a = root_a / "runs" / run_identifier(
        prefix="smoke_a",
        run_index=None,
        data_seed=0,
        train_seed=train_seed,
        method="baseline",
    )
    concat_a = root_a / "runs" / run_identifier(
        prefix="smoke_a",
        run_index=None,
        data_seed=0,
        train_seed=train_seed,
        method="mamba_concat",
    )
    concat_b = root_b / "runs" / run_identifier(
        prefix="smoke_b",
        run_index=None,
        data_seed=0,
        train_seed=train_seed,
        method="mamba_concat",
    )
    runs = [baseline_a, concat_a, concat_b]
    statuses = [
        json.loads((run / "status.json").read_text(encoding="utf-8"))
        for run in runs
    ]
    baseline_gate = json.loads(
        (baseline_a / "gate_report.json").read_text(encoding="utf-8")
    )
    baseline_arrays = np.load(
        baseline_a / "sampled_attribution_objects.npz",
        allow_pickle=False,
    )
    determinism = compare_runs(
        concat_a,
        concat_b,
        threshold=float(config["development_smoke"]["determinism_max_abs"]),
    )
    checks = {
        "exactly_three_runs": len(statuses) == 3,
        "all_complete": all(row["status"] == "complete" for row in statuses),
        "all_finite": all(row["no_nan_inf"] is True for row in statuses),
        "all_cuda": all(str(row["device"]).startswith("cuda") for row in statuses),
        "all_nonformal": all(row["formal_result"] is False for row in statuses),
        "all_nonmanuscript": all(
            row["manuscript_evidence"] is False for row in statuses
        ),
        "deterministic_algorithms": all(
            row["deterministic_algorithms"] is True for row in statuses
        ),
        "baseline_partial_total_identity": float(np.max(np.abs(
            baseline_arrays["s_total_nominal"]
            - baseline_arrays["s_partial_nominal"]
        ))) <= 1e-7,
        "baseline_missing_route_zero": (
            float(baseline_gate["details"]["missing_route_relative_magnitude"])
            <= 1e-7
        ),
        "concat_duplicate": determinism["passed"] is True,
    }
    report = {
        "checks": checks,
        "determinism": determinism,
        "passed": all(checks.values()),
        "formal_result": False,
        "manuscript_evidence": False,
    }
    atomic_json(output_path, report)
    return report


def main() -> int:
    args = parse_args()
    config_path = args.config.resolve()
    config = load_protocol(config_path)
    matrix = load_run_matrix(config)
    release_lock = validate_release_lock(
        config_path=config_path,
        release_manifest_path=args.release_manifest.resolve(),
        config=config,
    )
    output_root = args.output_root.resolve()

    if args.mode == "dry-run":
        write_traceability_snapshots(
            output_root=output_root,
            config=config,
            release_lock=release_lock,
        )
        report = {
            "protocol_name": PROTOCOL_NAME,
            "run_count": len(matrix),
            "data_seeds": sorted({row["data_seed"] for row in matrix}),
            "train_seeds": sorted({row["train_seed"] for row in matrix}),
            "methods": sorted({row["method"] for row in matrix}),
            "iterations": sorted({row["iterations"] for row in matrix}),
            "release_lock": release_lock,
            "passed": True,
            "formal_result": False,
            "manuscript_evidence": False,
        }
        atomic_json(output_root / "dry_run_validation.json", report)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0

    if args.mode == "generate-data":
        write_traceability_snapshots(
            output_root=output_root,
            config=config,
            release_lock=release_lock,
        )
        metadata = [
            generate_dataset(
                output_root=output_root,
                config=config,
                data_seed=int(data_seed),
                formal_result=True,
            )
            for data_seed in config["formal_design"]["data_seeds"]
        ]
        report = {
            "dataset_count": len(metadata),
            "datasets": metadata,
            "all_finite": all(
                row["variable_mean_max_abs"] < 1e-5
                and row["variable_std_max_abs_difference_from_one"] < 1e-5
                for row in metadata
            ),
            "formal_result": True,
            "manuscript_evidence": False,
            "manuscript_evidence_pending_aggregate": True,
        }
        atomic_json(output_root / "formal_dataset_manifest.json", report)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0

    if args.mode == "validate-smoke":
        if args.smoke_root_a is None or args.smoke_root_b is None:
            raise RuntimeError("Smoke validation requires both smoke roots")
        report = validate_smoke(
            root_a=args.smoke_root_a.resolve(),
            root_b=args.smoke_root_b.resolve(),
            config=config,
            output_path=output_root / "gpu_smoke_validation.json",
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if report["passed"] else 2

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if args.device.startswith("cuda") and os.environ.get(
        "CUBLAS_WORKSPACE_CONFIG"
    ) != ":4096:8":
        raise RuntimeError("CUBLAS_WORKSPACE_CONFIG must equal :4096:8")
    device = torch.device(args.device)
    write_traceability_snapshots(
        output_root=output_root,
        config=config,
        release_lock=release_lock,
    )
    requested_methods = [
        value.strip() for value in args.methods.split(",") if value.strip()
    ]
    if not requested_methods or not set(requested_methods).issubset(METHODS):
        raise RuntimeError(f"Invalid methods: {requested_methods}")

    if args.mode == "smoke":
        smoke = config["development_smoke"]
        data_seed = int(smoke["data_seed"])
        generate_dataset(
            output_root=output_root,
            config=config,
            data_seed=data_seed,
            formal_result=False,
        )
        atomic_json(
            output_root / "environment.json",
            environment_record(
                config_path=config_path,
                release_lock=release_lock,
                device=device,
                formal_result=False,
            ),
        )
        statuses = [
            run_one(
                run_index=None,
                prefix=args.tag,
                method=method,
                iterations=int(smoke["iterations"]),
                data_seed=data_seed,
                train_seed=int(smoke["train_seed"]),
                formal_result=False,
                config_path=config_path,
                config=config,
                release_lock=release_lock,
                output_root=output_root,
                device=device,
                resume=args.resume,
            )
            for method in requested_methods
        ]
    elif args.mode == "formal":
        dataset_manifest = output_root / "formal_dataset_manifest.json"
        if not dataset_manifest.is_file():
            raise RuntimeError("Formal data must be generated and frozen first")
        manifest_payload = json.loads(
            dataset_manifest.read_text(encoding="utf-8")
        )
        if manifest_payload.get("dataset_count") != len(
            config["formal_design"]["data_seeds"]
        ):
            raise RuntimeError("Formal dataset manifest is incomplete")
        atomic_json(
            output_root / "environment.json",
            environment_record(
                config_path=config_path,
                release_lock=release_lock,
                device=device,
                formal_result=True,
            ),
        )
        statuses = []
        for record in matrix:
            if record["method"] not in requested_methods:
                continue
            statuses.append(run_one(
                run_index=int(record["run_index"]),
                prefix=args.tag,
                method=str(record["method"]),
                iterations=int(record["iterations"]),
                data_seed=int(record["data_seed"]),
                train_seed=int(record["train_seed"]),
                formal_result=True,
                config_path=config_path,
                config=config,
                release_lock=release_lock,
                output_root=output_root,
                device=device,
                resume=args.resume,
            ))
    else:
        raise RuntimeError(f"Unhandled mode: {args.mode}")

    summary = {
        "mode": args.mode,
        "tag": args.tag,
        "methods": requested_methods,
        "run_count_in_invocation": len(statuses),
        "runs": statuses,
        "all_complete": all(row["status"] == "complete" for row in statuses),
        "all_finite": all(row["no_nan_inf"] is True for row in statuses),
        "total_wall_seconds": float(sum(row["wall_seconds"] for row in statuses)),
        "formal_result": args.mode == "formal",
        "development_only": args.mode != "formal",
        "manuscript_evidence": False,
        "manuscript_evidence_pending_aggregate": args.mode == "formal",
    }
    atomic_json(output_root / f"{args.tag}_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
