"""Development-only NetSim/MoCap Jacobian coverage audit runner."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Mapping, Optional

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
for _path in (PROJECT_ROOT, SRC_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from mamba_jrngc_pilot import train_model  # noqa: E402
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


DATASET_SPECS = {
    "netsim48": {
        "relative_path": "netsim/sim3_subject_48.npz",
        "sha256": "2f2d923a96f04c4d8f976efc97dd1b7a2bfc56d3c66b5f40b9783874fbce434f",
        "kind": "netsim",
        "segment": [0, 200],
        "model_seed": 24048,
        "perturbation_seed": 34048,
    },
    "netsim49": {
        "relative_path": "netsim/sim3_subject_49.npz",
        "sha256": "e394b2c0646a71693389e1c4307da1c15bf5299367ec4faaca8c2e847c71f73c",
        "kind": "netsim",
        "segment": [0, 200],
        "model_seed": 24049,
        "perturbation_seed": 34049,
    },
    "mocap_run": {
        "relative_path": "mocap/mocap_time_series_run_angles.npz",
        "sha256": "cc180d486832fbeb08a7be8de69494dccdc31615e0155022c96aef674fe17569",
        "kind": "mocap",
        "segment": [0, 600],
        "model_seed": 25001,
        "perturbation_seed": 35001,
    },
    "mocap_salsa": {
        "relative_path": "mocap/mocap_time_series_salsa_angles.npz",
        "sha256": "9f368e638d2c22327d8b67901c6eaab1a46ee44b249dc88faa49f220ab28dd39",
        "kind": "mocap",
        "segment": [1000, 1600],
        "model_seed": 25002,
        "perturbation_seed": 35002,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--datasets", default=",".join(DATASET_SPECS))
    parser.add_argument("--methods", default="baseline,concat")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--lag", type=int, default=3)
    parser.add_argument("--audit-horizon", type=int, default=64)
    parser.add_argument("--audit-window-count", type=int, default=32)
    parser.add_argument("--train-seeds", default="0")
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
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_list(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def load_dataset(
    name: str,
    *,
    data_root: Path,
) -> tuple[np.ndarray, Optional[np.ndarray], Dict[str, object]]:
    spec = DATASET_SPECS[name]
    path = data_root / str(spec["relative_path"])
    actual_sha = file_sha256(path)
    if actual_sha.lower() != str(spec["sha256"]).lower():
        raise RuntimeError(f"Dataset SHA mismatch for {name}: {actual_sha}")
    archive = np.load(path, allow_pickle=False)
    start, stop = [int(value) for value in spec["segment"]]
    if spec["kind"] == "netsim":
        full = np.asarray(archive["X_np"], dtype=np.float64)
        raw_segment = full[:, start:stop]
        graph = np.asarray(archive["Gref"])
        variable_names = [f"node_{index:02d}" for index in range(raw_segment.shape[0])]
    else:
        full = np.asarray(archive["time_series_data"], dtype=np.float64).T
        raw_segment = full[:, start:stop]
        graph = None
        variable_names = [str(value) for value in archive["joint_info"].tolist()]
    mean = np.mean(raw_segment, axis=1, keepdims=True)
    std = np.std(raw_segment, axis=1, keepdims=True)
    safe_std = np.where(std < 1e-8, 1.0, std)
    x = (raw_segment - mean) / safe_std
    metadata = {
        "dataset": name,
        "source_path": str(path),
        "source_sha256": actual_sha,
        "kind": spec["kind"],
        "segment_start_inclusive": start,
        "segment_stop_exclusive": stop,
        "segment_length": int(x.shape[1]),
        "d": int(x.shape[0]),
        "normalization": "per-variable z-score over frozen development segment",
        "normalization_scope": "development in-sample audit; not a forecasting claim",
        "normalization_mean": mean[:, 0].tolist(),
        "normalization_std": safe_std[:, 0].tolist(),
        "variable_names": variable_names,
        "ground_truth_available": graph is not None,
        "development_only": True,
        "formal_result": False,
    }
    return x, graph, metadata


def make_model_config(d: int, lag: int) -> Phase8ModelConfig:
    return Phase8ModelConfig(
        d=d,
        lag=lag,
        layers=2,
        hidden=32,
        dropout=0.0,
        d_cond=4,
        d_state=4,
        d_conv=4,
        expand=2,
        jacobian_lam=0.01,
        attribution_horizon=64,
        dtype="float32",
    )


def run_record(
    *,
    name: str,
    method: str,
    data_root: Path,
    output_root: Path,
    device: torch.device,
    max_iter: int,
    train_seed: int,
    lag: int,
    audit_horizon: int,
    audit_window_count: int,
    resume: bool,
) -> Dict[str, object]:
    spec = DATASET_SPECS[name]
    run_id = (
        f"{name}__{method}__ts{train_seed}"
        f"__it{max_iter}__H{audit_horizon}"
    )
    run_dir = output_root / run_id
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
    x, graph, data_metadata = load_dataset(name, data_root=data_root)
    model_seed = int(spec["model_seed"]) + 100 * int(train_seed)
    torch.manual_seed(model_seed)
    np.random.seed(model_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(model_seed)
        torch.cuda.reset_peak_memory_stats(device)
    cfg = make_model_config(x.shape[0], lag)
    if method == "baseline":
        adapter = make_legacy_baseline(cfg)
    elif method == "concat":
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
    adapter.model.to(device)
    train_started = time.perf_counter()
    adapter.model, best_total_objective = train_model(
        adapter.model,
        x,
        max_iter=max_iter,
        lr=1e-3,
        weight_decay=0.0,
        lookback=10,
        check_every=50,
        verbose=False,
    )
    training_seconds = time.perf_counter() - train_started
    target_idx = deterministic_audit_targets(
        T=x.shape[1],
        lag=lag,
        attribution_horizon=audit_horizon,
        count=audit_window_count,
        seed=int(spec["perturbation_seed"]) + 101,
    )
    eval_started = time.perf_counter()
    sampled = sampled_raw_chain_audit(
        adapter,
        x,
        target_indices=target_idx,
        attribution_horizon=audit_horizon,
        graph=graph,
    )
    if method == "baseline":
        interventions = fixed_target_baseline_interventions(
            adapter,
            x,
            perturbation_seed=int(spec["perturbation_seed"]),
        )
        coordinate_mixing = None
    else:
        interventions = fixed_target_concat_interventions(
            adapter,
            x,
            perturbation_seed=int(spec["perturbation_seed"]),
        )
        coordinate_mixing = condition_coordinate_mixing_audit(
            adapter,
            x,
            target_indices=target_idx,
            attribution_horizon=audit_horizon,
        )
    evaluation_seconds = time.perf_counter() - eval_started
    profile = build_audit_profile(
        architecture=(
            "baseline_jrngc"
            if method == "baseline"
            else (
                "legacy_mamba_concat_jrngc"
                if method == "concat"
                else "development_tcn_concat_jrngc"
            )
        ),
        sampled_audit=sampled,
        has_auxiliary_route=method == "concat",
    )
    pure_mse = adapter.pure_mse(x)
    atomic_json(run_dir / "config.json", {
        "run_id": run_id,
        "dataset": name,
        "method": method,
        "max_iter": max_iter,
        "model": asdict(cfg),
        "audit_horizon": audit_horizon,
        "audit_window_count": audit_window_count,
        "target_indices": target_idx.tolist(),
        "model_seed": model_seed,
        "train_seed": int(train_seed),
        "perturbation_seed": int(spec["perturbation_seed"]),
        "development_only": True,
        "formal_result": False,
    })
    atomic_json(run_dir / "dataset_metadata.json", data_metadata)
    atomic_json(run_dir / "sampled_attribution_audit.json", sampled.summary())
    atomic_npz(run_dir / "sampled_attribution_objects.npz", **sampled.arrays())
    atomic_json(run_dir / "fixed_target_interventions.json", interventions)
    atomic_json(run_dir / "coordinate_mixing_audit.json", coordinate_mixing)
    atomic_json(run_dir / "audit_profile.json", profile)
    torch.save({
        "model_state": adapter.model.state_dict(),
        "model_config": asdict(cfg),
        "run_id": run_id,
        "development_only": True,
    }, run_dir / "checkpoint.pt")
    status = {
        "status": "complete",
        "run_id": run_id,
        "dataset": name,
        "method": method,
        "train_seed": int(train_seed),
        "max_iter": max_iter,
        "best_total_regularized_objective": float(best_total_objective),
        "fixed_target_prediction_mse": float(pure_mse.detach().cpu()),
        "training_seconds": training_seconds,
        "evaluation_seconds": evaluation_seconds,
        "wall_seconds": time.perf_counter() - started,
        "cuda_max_memory_allocated_mb": (
            float(torch.cuda.max_memory_allocated(device) / 1024**2)
            if device.type == "cuda"
            else None
        ),
        "development_only": True,
        "formal_result": False,
    }
    atomic_json(status_path, status)
    return status


def main() -> int:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    datasets = parse_list(args.datasets)
    methods = parse_list(args.methods)
    train_seeds = [int(value) for value in parse_list(args.train_seeds)]
    if set(datasets) - set(DATASET_SPECS):
        raise ValueError(f"Unknown datasets: {sorted(set(datasets) - set(DATASET_SPECS))}")
    known_methods = {"baseline", "concat", "tcn_concat"}
    if set(methods) - known_methods:
        raise ValueError(f"Unknown methods: {sorted(set(methods) - known_methods)}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output_root / "protocol.json", {
        "development_only": True,
        "formal_result": False,
        "datasets": datasets,
        "methods": methods,
        "max_iter": args.max_iter,
        "lag": args.lag,
        "audit_horizon": args.audit_horizon,
        "audit_window_count": args.audit_window_count,
        "train_seeds": train_seeds,
        "full_prefix_claim_allowed": False,
        "ground_truth_claim_scope": "NetSim diagnostic only; MoCap has no graph truth",
        "phase7_seeds_4_to_8_accessed": False,
        "manuscript_modified": False,
        "hostname": platform.node(),
        "device": str(device),
        "torch": torch.__version__,
    })
    statuses = []
    for dataset in datasets:
        for train_seed in train_seeds:
            for method in methods:
                statuses.append(run_record(
                    name=dataset,
                    method=method,
                    data_root=args.data_root,
                    output_root=args.output_root,
                    device=device,
                    max_iter=args.max_iter,
                    train_seed=train_seed,
                    lag=args.lag,
                    audit_horizon=args.audit_horizon,
                    audit_window_count=args.audit_window_count,
                    resume=args.resume,
                ))
    atomic_json(args.output_root / "run_summary.json", {
        "development_only": True,
        "formal_result": False,
        "complete_count": sum(row["status"] == "complete" for row in statuses),
        "run_count": len(statuses),
        "statuses": statuses,
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
