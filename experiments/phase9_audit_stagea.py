"""Release-locked Phase 9 Stage A audit-generality validation runner."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
for _path in (PROJECT_ROOT, SRC_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from phase8_coverage import (  # noqa: E402
    Phase8ModelConfig,
    fixed_target_concat_interventions,
    make_legacy_baseline,
    make_legacy_concat,
)
from phase9_audit_generalization import (  # noqa: E402
    build_audit_profile,
    condition_coordinate_mixing_audit,
    deterministic_audit_targets,
    fixed_target_baseline_interventions,
    make_tcn_concat_adapter,
    sampled_raw_chain_audit,
)


STAGE = "A_4090_VALIDATION"
PRIMARY_ROLES = {
    "internal_prospective_go_no_go",
    "non_primary_determinism_duplicate",
}
METHODS = {"baseline", "mamba_concat", "tcn_concat"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--release-token", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--run-ids", default="")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke-iterations", type=int)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_text_sha256(path: Path) -> str:
    content = path.read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(content).hexdigest()


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


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def atomic_json(path: Path, payload: object) -> None:
    atomic_text(
        path,
        json.dumps(payload, indent=2, sort_keys=True, default=json_default) + "\n",
    )


def atomic_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def load_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_matrix(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def git_output(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def validate_release_lock(
    *,
    config_path: Path,
    matrix_path: Path,
    token_path: Path,
) -> Dict[str, object]:
    token = load_json(token_path)
    if token.get("protocol_name") != "phase9_audit_generality_stagea_v1":
        raise RuntimeError("Release token protocol name mismatch")
    if token.get("authorized_stage") != STAGE:
        raise RuntimeError("Release token does not authorize Stage A")
    if token.get("execution_authorized") is not True:
        raise RuntimeError("Release token is not authorized")
    actual_commit = git_output("rev-parse", "HEAD")
    if actual_commit != token.get("approved_commit"):
        raise RuntimeError(
            f"Commit mismatch: actual={actual_commit}, "
            f"approved={token.get('approved_commit')}"
        )
    worktree_status = git_output("status", "--porcelain")
    if worktree_status:
        raise RuntimeError("Stage A requires a clean Git worktree")
    config_sha = canonical_text_sha256(config_path)
    matrix_sha = canonical_text_sha256(matrix_path)
    if config_sha != token.get("config_sha256"):
        raise RuntimeError("Stage A config SHA256 mismatch")
    if matrix_sha != token.get("matrix_sha256"):
        raise RuntimeError("Stage A matrix SHA256 mismatch")
    critical_files = token.get("critical_files")
    if not isinstance(critical_files, dict) or not critical_files:
        raise RuntimeError("Release token has no critical-file manifest")
    actual_files = {}
    for relative, expected in critical_files.items():
        path = PROJECT_ROOT / str(relative)
        if not path.is_file():
            raise RuntimeError(f"Missing critical source: {relative}")
        actual = canonical_text_sha256(path)
        actual_files[str(relative)] = actual
        if actual != expected:
            raise RuntimeError(f"Critical source SHA mismatch: {relative}")
    return {
        "release_lock_mode": "git_commit_clean_worktree_and_source_manifest",
        "approved_commit": token["approved_commit"],
        "actual_commit": actual_commit,
        "clean_worktree": True,
        "config_sha256": config_sha,
        "matrix_sha256": matrix_sha,
        "release_token_sha256": file_sha256(token_path),
        "critical_files": actual_files,
    }


def validate_matrix(
    rows: Sequence[Mapping[str, str]],
    config: Mapping[str, object],
) -> List[Dict[str, str]]:
    selected = [
        dict(row)
        for row in rows
        if row.get("stage") == STAGE
        and row.get("evidence_role") in PRIMARY_ROLES
    ]
    if len(selected) != 56:
        raise RuntimeError(f"Expected 56 Stage A records, found {len(selected)}")
    run_ids = [row["run_id"] for row in selected]
    if len(set(run_ids)) != len(run_ids):
        raise RuntimeError("Stage A run IDs are not unique")
    primary = [
        row
        for row in selected
        if row["evidence_role"] == "internal_prospective_go_no_go"
    ]
    duplicates = [
        row
        for row in selected
        if row["evidence_role"] == "non_primary_determinism_duplicate"
    ]
    if len(primary) != 54 or len(duplicates) != 2:
        raise RuntimeError("Stage A primary/duplicate cardinality mismatch")
    if {row["method"] for row in primary} != METHODS:
        raise RuntimeError("Stage A method set mismatch")
    units = set(config["datasets"])  # type: ignore[arg-type]
    if {row["data_unit"] for row in primary} != units:
        raise RuntimeError("Stage A data-unit set mismatch")
    for row in selected:
        if row["environment"] != "windows_rtx4090":
            raise RuntimeError(f"Unexpected Stage A environment: {row['run_id']}")
        if row["method"] not in METHODS:
            raise RuntimeError(f"Unexpected method: {row['run_id']}")
        if int(row["max_iter"]) != int(config["training"]["max_iter"]):  # type: ignore[index]
            raise RuntimeError(f"Iteration mismatch: {row['run_id']}")
        if int(row["lag"]) != int(config["model"]["lag"]):  # type: ignore[index]
            raise RuntimeError(f"Lag mismatch: {row['run_id']}")
    return selected


def load_dataset(
    name: str,
    *,
    data_root: Path,
    config: Mapping[str, object],
) -> tuple[np.ndarray, Optional[np.ndarray], Dict[str, object]]:
    datasets = config["datasets"]
    spec = datasets[name]  # type: ignore[index]
    path = data_root / str(spec["relative_path"])
    actual_sha = file_sha256(path)
    if actual_sha != spec["sha256"]:
        raise RuntimeError(f"Dataset SHA mismatch for {name}: {actual_sha}")
    start, stop = [int(value) for value in spec["segment"]]
    with np.load(path, allow_pickle=False) as archive:
        if spec["kind"] == "netsim":
            full = np.asarray(archive["X_np"], dtype=np.float64)
            raw_segment = full[:, start:stop]
            graph = np.asarray(archive["Gref"]).copy()
            variable_names = [
                f"node_{index:02d}" for index in range(raw_segment.shape[0])
            ]
        else:
            full = np.asarray(archive["time_series_data"], dtype=np.float64).T
            raw_segment = full[:, start:stop]
            graph = None
            variable_names = [str(value) for value in archive["joint_info"].tolist()]
    if raw_segment.shape[1] != stop - start:
        raise RuntimeError(f"Frozen segment is incomplete for {name}")
    mean = np.mean(raw_segment, axis=1, keepdims=True)
    std = np.std(raw_segment, axis=1, keepdims=True)
    safe_std = np.where(std < 1e-8, 1.0, std)
    normalized = (raw_segment - mean) / safe_std
    if not np.all(np.isfinite(normalized)):
        raise RuntimeError(f"Nonfinite normalized data for {name}")
    metadata = {
        "data_unit": name,
        "source_path": str(path),
        "source_sha256": actual_sha,
        "kind": spec["kind"],
        "segment_start_inclusive": start,
        "segment_stop_exclusive": stop,
        "segment_length": int(normalized.shape[1]),
        "d": int(normalized.shape[0]),
        "normalization": "per-variable z-score over frozen validation segment",
        "normalization_scope": (
            "prospective in-sample architecture audit; not a forecasting claim"
        ),
        "normalization_mean": mean[:, 0].tolist(),
        "normalization_std": safe_std[:, 0].tolist(),
        "variable_names": variable_names,
        "ground_truth_available": graph is not None,
        "validation_only": True,
        "formal_result": False,
        "manuscript_evidence": False,
    }
    return normalized, graph, metadata


def model_config(
    *,
    d: int,
    config: Mapping[str, object],
) -> Phase8ModelConfig:
    model = config["model"]
    evaluation = config["evaluation"]
    return Phase8ModelConfig(  # type: ignore[index]
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


def _leaf_modules(model: nn.Module) -> Iterable[tuple[str, nn.Module]]:
    for name, module in model.named_modules():
        if name and not any(True for _ in module.children()):
            yield name, module


def reset_predictor(model: nn.Module, seed: int) -> None:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        for name, module in _leaf_modules(model):
            if name == "preprocessor" or name.startswith("preprocessor."):
                continue
            reset = getattr(module, "reset_parameters", None)
            if callable(reset):
                reset()


def state_subset_sha256(model: nn.Module, *, prefix_excluded: str) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        if name == prefix_excluded or name.startswith(prefix_excluded + "."):
            continue
        array = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def make_adapter(
    method: str,
    cfg: Phase8ModelConfig,
    *,
    predictor_seed: int,
    preprocessor_seed: Optional[int],
):
    construction_seed = predictor_seed if preprocessor_seed is None else preprocessor_seed
    torch.manual_seed(int(construction_seed))
    np.random.seed(int(construction_seed) % (2**32 - 1))
    if method == "baseline":
        adapter = make_legacy_baseline(cfg)
    elif method == "mamba_concat":
        adapter = make_legacy_concat(cfg)
    elif method == "tcn_concat":
        adapter = make_tcn_concat_adapter(
            d=cfg.d,
            lag=cfg.lag,
            layers=cfg.layers,
            hidden=cfg.hidden,
            dropout=cfg.dropout,
            jacobian_lam=cfg.jacobian_lam,
            d_cond=cfg.d_cond,
        )
    else:
        raise ValueError(f"Unknown method: {method}")
    reset_predictor(adapter.model, predictor_seed)
    return adapter


def train_legacy_with_metadata(
    model: nn.Module,
    x: np.ndarray,
    *,
    max_iter: int,
    learning_rate: float,
    weight_decay: float,
    gradient_clip_norm: float,
    check_every: int,
    lookback: int,
) -> Dict[str, object]:
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    best_loss = float("inf")
    best_iteration: Optional[int] = None
    best_state = None
    trace = []
    final_iteration = -1
    for iteration in range(max_iter):
        final_iteration = iteration
        model.train()
        optimizer.zero_grad()
        loss = model.compute_loss(x)
        if not bool(torch.isfinite(loss)):
            raise RuntimeError(f"Nonfinite training loss at iteration {iteration}")
        loss.backward()
        for name, parameter in model.named_parameters():
            if parameter.grad is not None and not bool(torch.all(torch.isfinite(parameter.grad))):
                raise RuntimeError(
                    f"Nonfinite gradient at iteration {iteration}: {name}"
                )
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            gradient_clip_norm,
        )
        if not bool(torch.isfinite(gradient_norm)):
            raise RuntimeError(f"Nonfinite gradient norm at iteration {iteration}")
        optimizer.step()
        if iteration % check_every == 0:
            value = float(loss.detach().cpu())
            trace.append(
                {
                    "iteration": iteration,
                    "total_regularized_objective": value,
                    "preclip_gradient_norm": float(gradient_norm.detach().cpu()),
                }
            )
            if value < best_loss:
                best_loss = value
                best_iteration = iteration
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in model.state_dict().items()
                }
            elif (
                best_iteration is not None
                and iteration - best_iteration >= lookback * check_every
                and iteration > 1000
            ):
                break
    if best_state is None or best_iteration is None:
        raise RuntimeError("No finite checked checkpoint was created")
    model.load_state_dict(best_state)
    return {
        "best_total_regularized_objective": best_loss,
        "selected_iteration": best_iteration,
        "final_iteration_before_restore": final_iteration,
        "trace": trace,
    }


def _offdiagonal_mass(jacobian: np.ndarray) -> float:
    d = jacobian.shape[0]
    return float(np.sum(jacobian[~np.eye(d, dtype=bool), :]))


def horizon_summary(audits: Mapping[int, object]) -> Dict[str, object]:
    reference = audits[128]
    ref_mass = _offdiagonal_mass(reference.j_bar_total)
    ratios = {}
    nominal_differences = {}
    for horizon in (32, 64, 128):
        audit = audits[horizon]
        ratios[str(horizon)] = (
            None
            if ref_mass <= 1e-12
            else _offdiagonal_mass(audit.j_bar_total) / ref_mass
        )
        nominal_differences[str(horizon)] = float(
            np.max(
                np.abs(
                    audit.s_total_nominal
                    - reference.s_total_nominal
                )
            )
        )
    return {
        "same_target_indices": True,
        "target_indices": reference.target_indices.tolist(),
        "horizons": [32, 64, 128],
        "offdiagonal_cumulative_mass_ratio_vs_H128": ratios,
        "nominal_score_max_abs_difference_vs_H128": nominal_differences,
        "mass_beyond_H128_assessed": False,
        "horizon_label": "HORIZON-TRUNCATED",
    }


def all_finite(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, Mapping):
        return all(all_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(all_finite(item) for item in value)
    if isinstance(value, (float, np.floating)):
        return bool(np.isfinite(value))
    return True


def run_record(
    row: Mapping[str, str],
    *,
    config: Mapping[str, object],
    release_lock: Mapping[str, object],
    data_root: Path,
    output_root: Path,
    device: torch.device,
    smoke_iterations: Optional[int],
    resume: bool,
) -> Dict[str, object]:
    run_id = row["run_id"]
    if smoke_iterations is not None:
        run_id = f"smoke_it{smoke_iterations}__{run_id}"
    run_dir = output_root / "runs" / run_id
    status_path = run_dir / "status.json"
    if resume and status_path.is_file():
        existing = load_json(status_path)
        if existing.get("status") == "complete":
            if existing.get("release_token_sha256") != release_lock[
                "release_token_sha256"
            ]:
                raise RuntimeError(f"Resume release mismatch: {run_id}")
            return existing
    if run_dir.exists() and any(run_dir.iterdir()):
        raise RuntimeError(f"Refusing nonempty run directory without resume: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    atomic_json(
        status_path,
        {
            "status": "running",
            "run_id": run_id,
            "validation_only": True,
            "formal_result": False,
            "manuscript_evidence": False,
        },
    )
    x, graph, data_metadata = load_dataset(
        row["data_unit"],
        data_root=data_root,
        config=config,
    )
    cfg = model_config(d=x.shape[0], config=config)
    predictor_seed = int(row["predictor_seed"])
    preprocessor_seed = (
        None if not row["preprocessor_seed"] else int(row["preprocessor_seed"])
    )
    adapter = make_adapter(
        row["method"],
        cfg,
        predictor_seed=predictor_seed,
        preprocessor_seed=preprocessor_seed,
    )
    predictor_initial_sha = state_subset_sha256(
        adapter.model,
        prefix_excluded="preprocessor",
    )
    adapter.model.to(device)
    master_seed = int(row["master_seed"])
    torch.manual_seed(master_seed)
    np.random.seed(master_seed % (2**32 - 1))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(master_seed)
        torch.cuda.reset_peak_memory_stats(device)
    training_cfg = config["training"]
    effective_iterations = (
        int(training_cfg["max_iter"])
        if smoke_iterations is None
        else int(smoke_iterations)
    )
    train_started = time.perf_counter()
    training = train_legacy_with_metadata(  # type: ignore[index]
        adapter.model,
        x,
        max_iter=effective_iterations,
        learning_rate=float(training_cfg["learning_rate"]),
        weight_decay=float(training_cfg["weight_decay"]),
        gradient_clip_norm=float(training_cfg["gradient_clip_norm"]),
        check_every=int(training_cfg["check_every"]),
        lookback=int(training_cfg["lookback"]),
    )
    training_seconds = time.perf_counter() - train_started

    evaluation_cfg = config["evaluation"]
    target_indices = deterministic_audit_targets(  # type: ignore[index]
        T=x.shape[1],
        lag=cfg.lag,
        attribution_horizon=int(evaluation_cfg["common_target_minimum_horizon"]),
        count=int(evaluation_cfg["audit_window_count"]),
        seed=int(row["score_window_seed"]),
    )
    eval_started = time.perf_counter()
    if row["method"] == "baseline":
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
            perturbation_seed=int(row["perturbation_seed"]),
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
            for horizon_value in (32, 64, 128)
        }
        primary = audits[64]
        horizon = horizon_summary(audits)
        interventions = fixed_target_concat_interventions(
            adapter,
            x,
            perturbation_seed=int(row["perturbation_seed"]),
        )
        mixing = condition_coordinate_mixing_audit(
            adapter,
            x,
            target_indices=target_indices,
            attribution_horizon=64,
        )
    evaluation_seconds = time.perf_counter() - eval_started
    architecture = {
        "baseline": "baseline_jrngc",
        "mamba_concat": "legacy_mamba_concat_jrngc",
        "tcn_concat": "causal_tcn_concat_jrngc",
    }[row["method"]]
    profile = build_audit_profile(
        architecture=architecture,
        sampled_audit=primary,
        has_auxiliary_route=row["method"] in {"mamba_concat", "tcn_concat"},
    )
    pure_mse = float(adapter.pure_mse(x).detach().cpu())
    payloads = {
        "primary_audit": primary.summary(),
        "interventions": interventions,
        "mixing": mixing,
        "horizon": horizon,
        "pure_mse": pure_mse,
    }
    if not all_finite(payloads):
        raise RuntimeError(f"Nonfinite evaluation output: {run_id}")

    record_config = {
        "run_id": run_id,
        "matrix_run_id": row["run_id"],
        "matrix_row": dict(row),
        "model": asdict(cfg),
        "effective_iterations": effective_iterations,
        "smoke": smoke_iterations is not None,
        "target_indices": target_indices.tolist(),
        "predictor_initial_sha256": predictor_initial_sha,
        "release_lock": dict(release_lock),
        "validation_only": True,
        "formal_result": False,
        "manuscript_evidence": False,
    }
    atomic_json(run_dir / "config.json", record_config)
    atomic_json(run_dir / "dataset_metadata.json", data_metadata)
    atomic_json(run_dir / "training.json", training)
    atomic_json(
        run_dir / "sampled_attribution_audit.json",
        primary.summary(),
    )
    atomic_npz(
        run_dir / "sampled_attribution_objects.npz",
        **primary.arrays(),
    )
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
    torch.save(
        {
            "model_state": adapter.model.state_dict(),
            "model_config": asdict(cfg),
            "run_id": run_id,
            "predictor_initial_sha256": predictor_initial_sha,
            "validation_only": True,
            "manuscript_evidence": False,
        },
        run_dir / "checkpoint.pt",
    )
    status = {
        "status": "complete",
        "run_id": run_id,
        "matrix_run_id": row["run_id"],
        "data_unit": row["data_unit"],
        "dataset_kind": row["dataset_kind"],
        "method": row["method"],
        "replicate": int(row["replicate"]),
        "evidence_role": row["evidence_role"],
        "duplicate_of": row["duplicate_of"] or None,
        "effective_iterations": effective_iterations,
        "selected_iteration": training["selected_iteration"],
        "best_total_regularized_objective": training[
            "best_total_regularized_objective"
        ],
        "fixed_target_prediction_mse": pure_mse,
        "predictor_initial_sha256": predictor_initial_sha,
        "training_seconds": training_seconds,
        "evaluation_seconds": evaluation_seconds,
        "wall_seconds": time.perf_counter() - started,
        "cuda_max_memory_allocated_mb": (
            float(torch.cuda.max_memory_allocated(device) / 1024**2)
            if device.type == "cuda"
            else None
        ),
        "device": str(device),
        "hostname": platform.node(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "release_token_sha256": release_lock["release_token_sha256"],
        "approved_commit": release_lock["approved_commit"],
        "validation_only": True,
        "formal_result": False,
        "manuscript_evidence": False,
        "smoke": smoke_iterations is not None,
        "no_nan_inf": True,
    }
    atomic_json(status_path, status)
    return status


def main() -> int:
    args = parse_args()
    config_path = args.config.resolve()
    matrix_path = args.matrix.resolve()
    token_path = args.release_token.resolve()
    config = load_json(config_path)
    if config.get("stage") != STAGE:
        raise RuntimeError("Config is not Stage A")
    release_lock = validate_release_lock(
        config_path=config_path,
        matrix_path=matrix_path,
        token_path=token_path,
    )
    rows = validate_matrix(read_matrix(matrix_path), config)
    requested = {
        value.strip() for value in args.run_ids.split(",") if value.strip()
    }
    if requested:
        unknown = requested - {row["run_id"] for row in rows}
        if unknown:
            raise RuntimeError(f"Unknown requested run IDs: {sorted(unknown)}")
        rows = [row for row in rows if row["run_id"] in requested]
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Stage A execution requires CUDA")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("CUBLAS_WORKSPACE_CONFIG must equal :4096:8")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    args.output_root.mkdir(parents=True, exist_ok=True)
    if not any(args.output_root.iterdir()):
        shutil.copy2(config_path, args.output_root / "config_snapshot.json")
        shutil.copy2(matrix_path, args.output_root / "matrix_snapshot.csv")
        shutil.copy2(token_path, args.output_root / "release_token_snapshot.json")
        atomic_json(args.output_root / "release_lock.json", release_lock)
    else:
        stored = load_json(args.output_root / "release_lock.json")
        if stored != release_lock:
            raise RuntimeError("Output root release lock mismatch")
    statuses = [
        run_record(
            row,
            config=config,
            release_lock=release_lock,
            data_root=args.data_root,
            output_root=args.output_root,
            device=device,
            smoke_iterations=args.smoke_iterations,
            resume=args.resume,
        )
        for row in rows
    ]
    atomic_json(
        args.output_root / "execution_summary.json",
        {
            "status": "complete",
            "requested_records": len(rows),
            "completed_records": sum(
                status["status"] == "complete" for status in statuses
            ),
            "smoke": args.smoke_iterations is not None,
            "validation_only": True,
            "formal_result": False,
            "manuscript_evidence": False,
            "statuses": statuses,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
