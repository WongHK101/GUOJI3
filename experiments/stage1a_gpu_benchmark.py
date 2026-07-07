"""Frozen Stage 1a execution runner for repaired ISTF methods.

This runner is execution infrastructure only. Scientific parameters are read
from a frozen JSON config; CLI may only choose config path, output root, device,
resume, plan-only, and smoke mode.
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
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from factorial_data import FACTORIAL_CELLS, FACTORIAL_SETTINGS, generate_factorial_cell  # noqa: E402
from minimal_mamba import MambaBlock  # noqa: E402
from repaired_istf import (  # noqa: E402
    RepairedISTFConfig,
    deterministic_sample_indices,
    eligible_target_indices,
    evaluate_repaired_model_chunked,
    finite_values_ok,
    horizon_sensitivity_audit,
    instantiate_repaired_method,
    make_cyclic_schedule,
    model_metadata,
    raw_chain_jacobian_for_windows,
    schedule_hash,
)


CELL_FLAGS = {name: (stationary, linear) for name, stationary, linear in FACTORIAL_CELLS}
FORMAL_METHOD_LABELS = {
    "baseline": "Baseline",
    "cp_depthwise": "CP-depthwise",
    "fixed_fir3": "FixedFIR3",
    "fixed_ema": "FixedEMA",
    "raw_chain_mamba": "RawChainMamba",
}
APPROVED_CONFIG_SHA_FILES = {
    "stage1a_frozen_config": PROJECT_ROOT / "configs" / "approved_stage1a_frozen_config_sha256.txt",
    "stage1a_cpu_smoke_config": PROJECT_ROOT / "configs" / "approved_stage1a_smoke_config_sha256.txt",
    "stage1a_gpu_infrastructure_smoke_config": PROJECT_ROOT / "configs" / "approved_stage1a_gpu_infrastructure_smoke_config_sha256.txt",
}
PREDICTOR_PREFIXES = ("inputgate.", "outputgate.", "encoders.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Frozen Stage 1 config JSON.")
    parser.add_argument("--output-root", default=str(PROJECT_ROOT / "results_kbs" / "stage1a_execution"))
    parser.add_argument("--device", default="cpu", help="cpu or cuda. Defaults to cpu for safety.")
    parser.add_argument("--resume", action="store_true", help="Skip completed runs with matching config hash.")
    parser.add_argument("--plan-only", action="store_true", help="Write run manifest without training.")
    parser.add_argument("--smoke", action="store_true", help="Require formal_result=false smoke config.")
    return parser.parse_args()


def load_json(path: Path) -> Dict[str, object]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def atomic_write_json(path: Path, payload: object) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, default=str))


def atomic_save_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.npy")
    np.save(tmp, array)
    os.replace(tmp, path)


def atomic_torch_save(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def canonical_config_hash(config: Dict[str, object]) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_approved_sha(path: Path) -> str:
    if not path.exists():
        raise ValueError(f"Approved config SHA file is missing: {path}")
    return path.read_text(encoding="utf-8").strip().split()[0].lower()


def git_commit_hash() -> str:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=30,
        )
        if proc.returncode == 0:
            return proc.stdout.strip()
    except Exception:
        pass
    return "unknown"


def rss_mb() -> float | None:
    try:
        import psutil  # type: ignore
    except Exception:
        return None
    return float(psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2))


def dir_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    return int(sum(p.stat().st_size for p in path.rglob("*") if p.is_file()))


def jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def get_path(config: Dict[str, object], dotted: str):
    cur = config
    for part in dotted.split("."):
        cur = cur[part]  # type: ignore[index]
    return cur


def require_equal(config: Dict[str, object], dotted: str, expected) -> None:
    actual = get_path(config, dotted)
    if actual != expected:
        raise ValueError(f"Frozen config mismatch: {dotted}={actual!r}, expected {expected!r}")


def validate_config(config: Dict[str, object], smoke: bool, config_path: Optional[Path] = None) -> None:
    required = [
        "config_name", "config_version", "formal_result", "frozen", "data", "model",
        "methods", "training", "target_windows", "attribution", "evaluation",
        "semantic_audit", "stage1a_go_no_go_gates", "run_count", "gpu_budget_estimate",
    ]
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"Config missing required keys: {missing}")
    if config_path is not None:
        name = str(config.get("config_name"))
        approved_path = APPROVED_CONFIG_SHA_FILES.get(name)
        if approved_path is None:
            raise ValueError(f"No approved SHA whitelist entry for config_name={name!r}")
        actual_sha = file_sha256(config_path)
        approved_sha = read_approved_sha(approved_path)
        if actual_sha.lower() != approved_sha:
            raise ValueError(
                f"Frozen config SHA mismatch for {name}: actual {actual_sha}, approved {approved_sha}"
            )
    if config["frozen"] is not True:
        raise ValueError("Config must have frozen=true")
    if bool(config["formal_result"]) == bool(smoke):
        raise ValueError("--smoke requires formal_result=false; formal mode requires formal_result=true")
    for dotted, expected in {
        "data.setting": "D2",
        "data.d": 10,
        "data.T": 600,
        "data.lag": 3,
        "data.sparsity": 0.2,
        "data.beta": 0.5,
        "data.s0": 0.075,
        "training.optimizer": "Adam",
        "training.lr": 0.001,
        "training.weight_decay": 0.0,
        "training.grad_clip": 1.0,
        "training.dtype": "float32",
        "training.jacobian_lam": 0.01,
        "target_windows.common_start": 64,
        "target_windows.common_stop_inclusive": 599,
        "target_windows.count": 536,
        "evaluation.chunk_size": 64,
        "evaluation.full_window_chunked_exact": True,
        "evaluation.accumulator_dtype": "float64",
    }.items():
        require_equal(config, dotted, expected)
    if smoke:
        require_equal(config, "training.max_iter", 20)
        require_equal(config, "training.primary_checkpoint", 20)
        require_equal(config, "data.cells", ["NS+Nonlinear"])
        require_equal(config, "data.data_seeds", [1])
        require_equal(config, "data.train_seeds", [0])
        if config["config_name"] == "stage1a_gpu_infrastructure_smoke_config":
            require_equal(config, "run_count.formal", 4)
            require_equal(config, "run_count.limited_ablation", 1)
            require_equal(config, "run_count.total", 5)
        else:
            require_equal(config, "run_count.formal", 4)
            require_equal(config, "run_count.limited_ablation", 0)
            require_equal(config, "run_count.total", 4)
    else:
        require_equal(config, "training.max_iter", 500)
        require_equal(config, "training.primary_checkpoint", 500)
        require_equal(config, "data.cells", ["Stat+Linear", "Stat+Nonlinear", "NS+Linear", "NS+Nonlinear"])
        require_equal(config, "data.data_seeds", [1, 2, 3])
        require_equal(config, "data.train_seeds", [0, 1])
        require_equal(config, "run_count.total", 100)
    require_equal(config, "training.seed_rules.predictor_seed", "100000 + 1000*data_seed + 100*train_seed")
    require_equal(config, "training.seed_rules.raw_chain_mamba_filter_seed", "200000 + 1000*data_seed + 100*train_seed")
    require_equal(config, "semantic_audit.audit_window_count", 32)
    require_equal(config, "semantic_audit.audit_window_seed", 9301)
    require_equal(config, "semantic_audit.audit_window_min_target", 64)
    d2 = FACTORIAL_SETTINGS["D2"]
    if d2["nonlinear_strength"] != config["data"]["beta"] or d2["nonlinear_scale"] != config["data"]["s0"]:  # type: ignore[index]
        raise ValueError("Local FACTORIAL_SETTINGS['D2'] no longer matches frozen config beta/s0")


def resolve_device(device_name: str) -> torch.device:
    if device_name == "cpu":
        return torch.device("cpu")
    if device_name.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(f"Requested {device_name}, but CUDA is unavailable")
        return torch.device(device_name)
    raise ValueError(f"Unsupported device: {device_name}")


def slug(text: str) -> str:
    return text.replace("+", "_").replace(" ", "_")


def build_run_manifest(config: Dict[str, object], output_root: Path) -> List[Dict[str, object]]:
    runs: List[Dict[str, object]] = []
    for method in config["methods"]["formal"]:  # type: ignore[index]
        for cell in config["data"]["cells"]:  # type: ignore[index]
            for data_seed in config["data"]["data_seeds"]:  # type: ignore[index]
                for train_seed in config["data"]["train_seeds"]:  # type: ignore[index]
                    run_id = f"stage1a__{method}__{slug(cell)}__data{data_seed}__train{train_seed}"
                    runs.append({
                        "run_id": run_id,
                        "role": "formal",
                        "method": method,
                        "method_label": FORMAL_METHOD_LABELS.get(method, method),
                        "cell": cell,
                        "data_seed": int(data_seed),
                        "train_seed": int(train_seed),
                        "output_path": str(output_root / "runs" / run_id),
                    })
    for item in config["methods"].get("limited_ablation", []):  # type: ignore[union-attr]
        method = item["method"]
        for cell in item["cells"]:
            for data_seed in item["data_seeds"]:
                for train_seed in item["train_seeds"]:
                    run_id = f"stage1a_limited__{method}__{slug(cell)}__data{data_seed}__train{train_seed}"
                    runs.append({
                        "run_id": run_id,
                        "role": "limited_ablation",
                        "method": method,
                        "method_label": FORMAL_METHOD_LABELS.get(method, method),
                        "cell": cell,
                        "data_seed": int(data_seed),
                        "train_seed": int(train_seed),
                        "output_path": str(output_root / "runs" / run_id),
                    })
    expected = int(config["run_count"]["total"])  # type: ignore[index]
    if len(runs) != expected:
        raise ValueError(f"Run manifest count {len(runs)} does not match frozen config total {expected}")
    return runs


def method_horizon(config: Dict[str, object], method: str) -> int:
    return int(config["attribution"]["horizons"][method])  # type: ignore[index]


def method_cfg(config: Dict[str, object], method: str) -> RepairedISTFConfig:
    return RepairedISTFConfig(
        d=int(config["data"]["d"]),  # type: ignore[index]
        lag=int(config["data"]["lag"]),  # type: ignore[index]
        attribution_horizon=method_horizon(config, method),
        layers=int(config["model"]["predictor"]["layers"]),  # type: ignore[index]
        hidden=int(config["model"]["predictor"]["hidden"]),  # type: ignore[index]
        dropout=float(config["model"]["predictor"]["dropout"]),  # type: ignore[index]
        jacobian_lam=float(config["training"]["jacobian_lam"]),  # type: ignore[index]
        identity_lam=float(config["training"]["identity_lam"][method]),  # type: ignore[index]
        residual_gain=float(config["model"]["cp_depthwise"]["residual_gain"]),  # type: ignore[index]
        depthwise_kernel_size=int(config["model"]["cp_depthwise"]["kernel_size"]),  # type: ignore[index]
        d_state=int(config["model"]["raw_chain_mamba"]["d_state"]),  # type: ignore[index]
        mamba_expand=int(config["model"]["raw_chain_mamba"]["expand"]),  # type: ignore[index]
        mamba_d_conv=int(config["model"]["raw_chain_mamba"]["d_conv"]),  # type: ignore[index]
        ema_alpha=float(config["model"]["fixed_ema"]["alpha"]),  # type: ignore[index]
        fir3_gamma=float(config["model"]["fixed_fir3"]["gamma"]),  # type: ignore[index]
        dtype=str(config["training"]["dtype"]),  # type: ignore[index]
    )


def common_target_indices(config: Dict[str, object]) -> np.ndarray:
    start = int(config["target_windows"]["common_start"])  # type: ignore[index]
    stop = int(config["target_windows"]["common_stop_inclusive"])  # type: ignore[index]
    idx = np.arange(start, stop + 1, dtype=np.int64)
    if len(idx) != int(config["target_windows"]["count"]):  # type: ignore[index]
        raise ValueError("common_target_indices count mismatch")
    return idx


def schedule_seed(data_seed: int, train_seed: int) -> int:
    return 7101 + 1000 * int(data_seed) + int(train_seed)


def predictor_seed(data_seed: int, train_seed: int) -> int:
    return 100000 + 1000 * int(data_seed) + 100 * int(train_seed)


def raw_chain_mamba_filter_seed(data_seed: int, train_seed: int) -> int:
    return 200000 + 1000 * int(data_seed) + 100 * int(train_seed)


def configure_torch_determinism(seed: int, device: torch.device) -> Dict[str, object]:
    torch.manual_seed(int(seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))
        torch.cuda.reset_peak_memory_stats(device)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return {
        "seed": int(seed),
        "torch_use_deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cuda_manual_seed_all": bool(device.type == "cuda"),
    }


def predictor_state_dict(model) -> Dict[str, torch.Tensor]:
    return {
        key: tensor.detach().cpu().clone()
        for key, tensor in model.state_dict().items()
        if key.startswith(PREDICTOR_PREFIXES)
    }


def instantiate_paired_method(
    method: str,
    cfg: RepairedISTFConfig,
    data_seed: int,
    train_seed: int,
) -> tuple[torch.nn.Module, Dict[str, Optional[int]]]:
    p_seed = predictor_seed(data_seed, train_seed)
    torch.manual_seed(p_seed)
    model = instantiate_repaired_method(method, cfg)
    seeds: Dict[str, Optional[int]] = {"predictor_seed": p_seed, "filter_seed": None}
    if method == "raw_chain_mamba":
        f_seed = raw_chain_mamba_filter_seed(data_seed, train_seed)
        torch.manual_seed(f_seed)
        dtype = next(model.parameters()).dtype
        model.filter = MambaBlock(
            d_model=cfg.d,
            d_state=cfg.d_state,
            d_conv=cfg.mamba_d_conv,
            expand=cfg.mamba_expand,
            residual_scale=cfg.residual_gain,
        ).to(dtype=dtype)
        seeds["filter_seed"] = f_seed
    return model, seeds


def generate_cell(config: Dict[str, object], cell: str, data_seed: int):
    stationary, linear = CELL_FLAGS[cell]
    params = FACTORIAL_SETTINGS["D2"]
    return generate_factorial_cell(
        d=int(config["data"]["d"]),  # type: ignore[index]
        T=int(config["data"]["T"]),  # type: ignore[index]
        lag=int(config["data"]["lag"]),  # type: ignore[index]
        seed=int(data_seed),
        stationary=stationary,
        linear=linear,
        coeff_scale=params["coeff_scale"],
        noise_scale=params["noise_scale"],
        regime_shift_strength=0.0 if stationary else params["regime_shift_strength"],
        nonlinear_strength=0.0 if linear else params["nonlinear_strength"],
        nonlinear_scale=params["nonlinear_scale"],
        sparsity=float(config["data"]["sparsity"]),  # type: ignore[index]
        return_metadata=True,
    )


def environment_payload(
    config: Dict[str, object],
    device: torch.device,
    deterministic_settings: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "cwd": str(PROJECT_ROOT),
        "device": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
        "formal_result": bool(config["formal_result"]),
        "deterministic_settings": deterministic_settings or {
            "torch_use_deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
            "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
            "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        },
    }


def train_step(model, optimizer, x, schedule_entry, target_indices, grad_clip: float) -> Dict[str, float]:
    optimizer.zero_grad(set_to_none=True)
    comp = model.compute_loss_components(x, schedule_entry=schedule_entry, target_indices=target_indices)
    comp["total_training_objective"].backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(grad_clip))
    optimizer.step()
    return {key: float(val.detach().cpu()) for key, val in comp.items()}


def audit_target_indices(config: Dict[str, object], target_indices: Sequence[int]) -> np.ndarray:
    audit_cfg = config["semantic_audit"]  # type: ignore[index]
    return deterministic_sample_indices(
        target_indices,
        n=int(audit_cfg["audit_window_count"]),  # type: ignore[index]
        seed=int(audit_cfg["audit_window_seed"]),  # type: ignore[index]
        require_min_target=int(audit_cfg["audit_window_min_target"]),  # type: ignore[index]
    )


def full_prefix_omitted_mass(model, x, target_indices: Sequence[int], horizon: int) -> Dict[str, object]:
    idx = [int(i) for i in target_indices if int(i) >= horizon]
    if not idx:
        raise ValueError("full-prefix omitted-mass audit has no target indices")
    _, per_full = raw_chain_jacobian_for_windows(
        model,
        x,
        target_indices=idx,
        attribution_horizon=horizon,
        create_graph=False,
        full_prefix=True,
    )
    vals = []
    for u, jac_u in zip(idx, per_full):
        early = jac_u[:, :, :max(0, u - horizon)]
        omitted = float(torch.sum(torch.abs(early)).detach().cpu())
        total = float(torch.sum(torch.abs(jac_u)).detach().cpu())
        vals.append(omitted / (total + 1e-12))
    arr = np.asarray(vals, dtype=np.float64)
    return {
        "horizon": int(horizon),
        "target_indices": idx,
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(np.max(arr)),
        "per_window": [float(v) for v in arr],
    }


def _corr_value(comp: Dict[str, object]) -> Optional[float]:
    pearson = comp.get("pearson")
    if isinstance(pearson, dict) and pearson.get("value") is not None:
        return float(pearson["value"])
    spearman = comp.get("spearman")
    if spearman is not None:
        return float(spearman)
    return None


def _semantic_thresholds(config: Dict[str, object]) -> Dict[str, object]:
    return config["semantic_audit"]["thresholds"]  # type: ignore[index]


def build_semantic_audit(
    method: str,
    model,
    x,
    graph,
    target_indices: Sequence[int],
    eval_out: Dict[str, object],
    config: Dict[str, object],
) -> Dict[str, object]:
    thresholds = _semantic_thresholds(config)
    audit_idx = audit_target_indices(config, target_indices)
    audit: Dict[str, object] = {
        "method": method,
        "audit_window_indices": [int(i) for i in audit_idx],
        "thresholds": thresholds,
        "passed": True,
        "failures": [],
    }
    failures: List[str] = []
    if method in {"cp_depthwise", "fixed_fir3"}:
        horizon = horizon_sensitivity_audit(
            model,
            x,
            graph,
            target_indices=audit_idx,
            h_small=32,
            h_large=64,
        )
        leak = float(eval_out["cross_variable_leakage"]["cross_variable_leakage"])  # type: ignore[index]
        temporal = eval_out["temporal_horizon_mass"]  # type: ignore[assignment]
        filt = eval_out["filter_diagnostics"]  # type: ignore[assignment]
        if leak >= float(thresholds["leakage_max"]):
            failures.append(f"cross_variable_leakage={leak:.6g} >= {thresholds['leakage_max']}")
        if float(temporal["median"]) > float(thresholds["temporal_horizon_median_max"]):  # type: ignore[index]
            failures.append(f"temporal_horizon_median={temporal['median']} > {thresholds['temporal_horizon_median_max']}")  # type: ignore[index]
        if float(temporal["max"]) > float(thresholds["temporal_horizon_max_max"]):  # type: ignore[index]
            failures.append(f"temporal_horizon_max={temporal['max']} > {thresholds['temporal_horizon_max_max']}")  # type: ignore[index]
        for label in ["H32_vs_H64", "H32_vs_full_prefix"]:
            comp = horizon[label]
            corr = _corr_value(comp)  # type: ignore[arg-type]
            if corr is None or corr < float(thresholds["score_corr_min"]):
                failures.append(f"{label}_score_corr={corr} < {thresholds['score_corr_min']}")
            if float(comp["topk_jaccard"]) < float(thresholds["topk_jaccard_min"]):  # type: ignore[index]
                failures.append(f"{label}_topk_jaccard={comp['topk_jaccard']} < {thresholds['topk_jaccard_min']}")  # type: ignore[index]
        omitted_max = float(horizon["omitted_gradient_mass"]["max"])  # type: ignore[index]
        if omitted_max >= float(thresholds["omitted_gradient_mass_max"]):
            failures.append(f"omitted_gradient_mass_max={omitted_max:.6g} >= {thresholds['omitted_gradient_mass_max']}")
        if method == "cp_depthwise":
            kernel_norm = float(filt["kernel_frobenius_norm"])  # type: ignore[index]
            identity = float(filt["identity_deviation"])  # type: ignore[index]
            if kernel_norm <= 0.0:
                failures.append("kernel_frobenius_norm <= 0")
            if not np.isfinite(identity) or identity <= 0.0:
                failures.append("identity_deviation not finite and nonzero")
        audit.update({
            "track_role": "nominal_lag_candidate",
            "cross_variable_leakage": leak,
            "temporal_horizon_mass": jsonable(temporal),
            "horizon_sensitivity": jsonable(horizon),
            "filter_diagnostics": jsonable(filt),
        })
    elif method == "fixed_ema":
        omitted = full_prefix_omitted_mass(model, x, audit_idx, horizon=64)
        if float(omitted["max"]) >= float(thresholds["ema_omitted_gradient_mass_max"]):
            failures.append(
                f"ema_full_prefix_omitted_mass_max={omitted['max']} >= {thresholds['ema_omitted_gradient_mass_max']}"
            )
        audit.update({
            "track_role": "full_H_reference",
            "metric_track_required": "full_H",
            "full_prefix_omitted_mass": jsonable(omitted),
        })
    elif method == "raw_chain_mamba":
        audit.update({
            "track_role": "limited_diagnostic_only",
            "not_a_formal_nominal_lag_candidate": True,
        })
    else:
        audit.update({"track_role": "baseline_no_filter_semantic_gate"})
    audit["failures"] = failures
    audit["passed"] = len(failures) == 0
    return audit


def output_complete(run_dir: Path, checkpoint_iter: int) -> bool:
    required = [
        run_dir / "config_snapshot.json",
        run_dir / "config_sha256.txt",
        run_dir / "commit_hash.txt",
        run_dir / "environment.json",
        run_dir / "generator_metadata.json",
        run_dir / "schedule.json",
        run_dir / "loss_trace.json",
        run_dir / "checkpoints" / f"iter_{checkpoint_iter:04d}.pt",
        run_dir / "scores" / "raw_chain_j_bar.npy",
        run_dir / "scores" / "score_nominal.npy",
        run_dir / "scores" / "score_full_H.npy",
        run_dir / "metrics.json",
        run_dir / "diagnostics.json",
        run_dir / "runtime.json",
        run_dir / "status.json",
    ]
    return all(p.exists() and p.stat().st_size > 0 for p in required)


def status_matches_complete(run_dir: Path, config_hash: str, checkpoint_iter: int) -> bool:
    status_path = run_dir / "status.json"
    if not status_path.exists():
        return False
    status = load_json(status_path)
    if status.get("config_sha256") != config_hash:
        raise RuntimeError(f"Existing run {run_dir} has config hash mismatch; refusing reuse")
    return status.get("status") == "complete" and output_complete(run_dir, checkpoint_iter)


def run_one(
    run: Dict[str, object],
    config: Dict[str, object],
    config_hash: str,
    output_root: Path,
    device: torch.device,
    resume: bool,
) -> Dict[str, object]:
    run_dir = Path(str(run["output_path"]))
    checkpoint_iter = int(config["training"]["primary_checkpoint"])  # type: ignore[index]
    if run_dir.exists() and (run_dir / "status.json").exists():
        if status_matches_complete(run_dir, config_hash, checkpoint_iter):
            if resume:
                status = load_json(run_dir / "status.json")
                status["resume_action"] = "skipped_complete_matching_hash"
                return status
            raise RuntimeError(f"Run already complete; use --resume to skip: {run['run_id']}")
    run_dir.mkdir(parents=True, exist_ok=True)
    run_started = time.perf_counter()
    peak = rss_mb()
    commit = git_commit_hash()
    status = {
        "run_id": run["run_id"],
        "status": "running",
        "config_sha256": config_hash,
        "formal_result": bool(config["formal_result"]),
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    atomic_write_json(run_dir / "status.json", status)
    try:
        atomic_write_json(run_dir / "config_snapshot.json", config)
        atomic_write_text(run_dir / "config_sha256.txt", config_hash + "\n")
        atomic_write_text(run_dir / "commit_hash.txt", commit + "\n")
        x, graph, metadata = generate_cell(config, str(run["cell"]), int(run["data_seed"]))
        atomic_write_json(run_dir / "generator_metadata.json", jsonable(metadata))

        cfg = method_cfg(config, str(run["method"]))
        idx = common_target_indices(config)
        audit_idx = audit_target_indices(config, idx)
        seeds = {
            "predictor_seed": predictor_seed(int(run["data_seed"]), int(run["train_seed"])),
            "filter_seed": (
                raw_chain_mamba_filter_seed(int(run["data_seed"]), int(run["train_seed"]))
                if str(run["method"]) == "raw_chain_mamba"
                else None
            ),
        }
        deterministic_settings = configure_torch_determinism(int(seeds["predictor_seed"]), device)
        atomic_write_json(run_dir / "environment.json", environment_payload(config, device, deterministic_settings))
        schedule = make_cyclic_schedule(
            idx,
            d=cfg.d,
            max_iter=int(config["training"]["max_iter"]),  # type: ignore[index]
            windows_per_step=int(config["training"]["jacobian_estimator"]["sampled_windows_per_step"]),  # type: ignore[index]
            targets_per_step=int(config["training"]["jacobian_estimator"]["sampled_output_targets_per_step"]),  # type: ignore[index]
            seed=schedule_seed(int(run["data_seed"]), int(run["train_seed"])),
        )
        digest = schedule_hash(schedule)
        atomic_write_json(run_dir / "schedule.json", {
            "schedule_hash": digest,
            "schedule_seed": schedule_seed(int(run["data_seed"]), int(run["train_seed"])),
            "predictor_seed": seeds["predictor_seed"],
            "filter_seed": seeds["filter_seed"],
            "common_target_indices": [int(i) for i in idx],
            "semantic_audit_target_indices": [int(i) for i in audit_idx],
            "schedule": schedule,
        })

        model, init_seeds = instantiate_paired_method(
            str(run["method"]),
            cfg,
            int(run["data_seed"]),
            int(run["train_seed"]),
        )
        model = model.to(device)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=float(config["training"]["lr"]),  # type: ignore[index]
            weight_decay=float(config["training"]["weight_decay"]),  # type: ignore[index]
        )
        loss_trace = []
        train_started = time.perf_counter()
        for it in range(1, int(config["training"]["max_iter"]) + 1):  # type: ignore[index]
            loss_payload = train_step(
                model,
                optimizer,
                x,
                schedule[it - 1],
                idx,
                grad_clip=float(config["training"]["grad_clip"]),  # type: ignore[index]
            )
            loss_trace.append({"iter": it, **loss_payload})
            mem = rss_mb()
            if mem is not None:
                peak = mem if peak is None else max(peak, mem)
        train_elapsed = time.perf_counter() - train_started
        atomic_write_json(run_dir / "loss_trace.json", loss_trace)

        ckpt_path = run_dir / "checkpoints" / f"iter_{checkpoint_iter:04d}.pt"
        atomic_torch_save(ckpt_path, {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "config": asdict(cfg),
            "run": run,
            "iteration": checkpoint_iter,
            "schedule_hash": digest,
            "commit_hash": commit,
            "formal_result": bool(config["formal_result"]),
            "predictor_seed": init_seeds["predictor_seed"],
            "filter_seed": init_seeds["filter_seed"],
        })

        eval_started = time.perf_counter()
        eval_out = evaluate_repaired_model_chunked(
            model,
            x,
            graph,
            target_indices=idx,
            attribution_horizon=cfg.attribution_horizon,
            chunk_size=int(config["evaluation"]["chunk_size"]),  # type: ignore[index]
            include_filtered_coordinate=False,
            prediction_target_indices=idx,
            leakage_target_indices=idx[: min(32, len(idx))],
        )
        eval_elapsed = time.perf_counter() - eval_started
        mem = rss_mb()
        if mem is not None:
            peak = mem if peak is None else max(peak, mem)
        semantic_audit = build_semantic_audit(
            str(run["method"]),
            model,
            x,
            graph,
            target_indices=idx,
            eval_out=eval_out,
            config=config,
        )

        scores_dir = run_dir / "scores"
        atomic_save_npy(scores_dir / "raw_chain_j_bar.npy", eval_out["raw_chain_j_bar"])
        atomic_save_npy(scores_dir / "score_nominal.npy", eval_out["score_nominal"])
        atomic_save_npy(scores_dir / "score_full_H.npy", eval_out["score_full_H"])
        atomic_write_json(run_dir / "metrics.json", {
            "metrics_nominal": jsonable(eval_out["metrics_nominal"]),
            "metrics_full_H": jsonable(eval_out["metrics_full_H"]),
            "eval_raw_prediction_loss": eval_out["eval_raw_prediction_loss"],
            "eval_filtered_prediction_loss": eval_out["eval_filtered_prediction_loss"],
            "metric_track_for_aggregation": "full_H" if run["method"] == "fixed_ema" else "nominal",
        })
        atomic_write_json(run_dir / "diagnostics.json", {
            "model_metadata": model_metadata(model),
            "temporal_horizon_mass": jsonable(eval_out["temporal_horizon_mass"]),
            "cross_variable_leakage": jsonable(eval_out["cross_variable_leakage"]),
            "filter_diagnostics": jsonable(eval_out["filter_diagnostics"]),
            "chunked_evaluator": jsonable(eval_out["chunked_evaluator"]),
            "semantic_audit": jsonable(semantic_audit),
            "semantic_diagnostic_note": "graph metrics use all 536 common windows; horizon/full-prefix audit uses preregistered deterministic 32 common target indices",
        })
        score_size = dir_size_bytes(scores_dir)
        checkpoint_size = ckpt_path.stat().st_size
        sec_per_iter = train_elapsed / max(1, int(config["training"]["max_iter"]))  # type: ignore[arg-type,index]
        runtime = {
            "training_wall_time_seconds": train_elapsed,
            "training_wall_time_per_iteration_seconds": sec_per_iter,
            "full_evaluation_wall_time_seconds": eval_elapsed,
            "cpu_peak_rss_mb": peak,
            "checkpoint_size_bytes": checkpoint_size,
            "score_size_bytes": score_size,
            "estimated_500_iteration_runtime_hours": sec_per_iter * 500.0 / 3600.0,
            "jacobian_output_buffer_lower_bound_mb": (
                int(config["evaluation"]["chunk_size"]) * cfg.d * cfg.d * cfg.attribution_horizon * 8 / (1024 ** 2)  # type: ignore[index]
            ),
            "cuda_max_memory_allocated_mb": (
                float(torch.cuda.max_memory_allocated(device) / (1024 ** 2)) if device.type == "cuda" else None
            ),
            "cuda_max_memory_reserved_mb": (
                float(torch.cuda.max_memory_reserved(device) / (1024 ** 2)) if device.type == "cuda" else None
            ),
        }
        atomic_write_json(run_dir / "runtime.json", runtime)
        no_nan_inf = finite_values_ok(loss_trace) and finite_values_ok({
            "metrics": eval_out["metrics_nominal"],
            "metrics_full_H": eval_out["metrics_full_H"],
            "diag": eval_out["temporal_horizon_mass"],
            "loss": eval_out["eval_raw_prediction_loss"],
            "semantic_audit": semantic_audit,
        })
        complete = output_complete(run_dir, checkpoint_iter)
        status = {
            **status,
            "status": "complete" if (no_nan_inf and complete) else "failed",
            "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "run_wall_time_seconds": time.perf_counter() - run_started,
            "config_sha256": config_hash,
            "commit_hash": commit,
            "schedule_hash": digest,
            "no_nan_inf": bool(no_nan_inf),
            "output_complete": bool(complete),
            "formal_result": bool(config["formal_result"]),
            "checkpoint_iteration": checkpoint_iter,
            "device": str(device),
            "predictor_seed": init_seeds["predictor_seed"],
            "filter_seed": init_seeds["filter_seed"],
            "deterministic_settings": deterministic_settings,
            "semantic_audit_passed": bool(semantic_audit["passed"]),
        }
        atomic_write_json(run_dir / "status.json", status)
        return status
    except Exception as exc:
        status = {
            **status,
            "status": "failed",
            "failed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "run_wall_time_seconds": time.perf_counter() - run_started,
        }
        atomic_write_json(run_dir / "status.json", status)
        return status


def write_manifest(output_root: Path, config: Dict[str, object], config_hash: str, runs: List[Dict[str, object]], statuses=None) -> None:
    payload = {
        "config_sha256": config_hash,
        "formal_result": bool(config["formal_result"]),
        "run_count": len(runs),
        "expected_run_count": int(config["run_count"]["total"]),  # type: ignore[index]
        "runs": runs,
        "statuses": statuses or [],
    }
    atomic_write_json(output_root / "run_manifest.json", payload)


def write_smoke_report(output_root: Path, statuses: Sequence[Dict[str, object]]) -> None:
    rows = []
    for status in statuses:
        run_dir = output_root / "runs" / str(status["run_id"])
        runtime = load_json(run_dir / "runtime.json") if (run_dir / "runtime.json").exists() else {}
        rows.append({
            "run_id": status["run_id"],
            "method": str(status["run_id"]).split("__")[1] if "__" in str(status["run_id"]) else "",
            "training_wall_time_per_iteration_seconds": runtime.get("training_wall_time_per_iteration_seconds"),
            "full_evaluation_wall_time_seconds": runtime.get("full_evaluation_wall_time_seconds"),
            "cpu_peak_rss_mb": runtime.get("cpu_peak_rss_mb"),
            "checkpoint_size_bytes": runtime.get("checkpoint_size_bytes"),
            "score_size_bytes": runtime.get("score_size_bytes"),
            "estimated_500_iteration_runtime_hours": runtime.get("estimated_500_iteration_runtime_hours"),
            "jacobian_output_buffer_lower_bound_mb": runtime.get("jacobian_output_buffer_lower_bound_mb"),
            "no_nan_inf": status.get("no_nan_inf"),
            "output_complete": status.get("output_complete"),
            "formal_result": status.get("formal_result"),
        })
    atomic_write_json(output_root / "cpu_smoke_report.json", {
        "formal_result": False,
        "stage1_scale": "d=10,T=600,lag=3,NS+Nonlinear,data_seed=1,train_seed=0,common_windows=536",
        "rows": rows,
        "all_passed": all(r["no_nan_inf"] and r["output_complete"] for r in rows),
    })


def main() -> int:
    args = parse_args()
    config_path = Path(args.config)
    config = load_json(config_path)
    validate_config(config, smoke=args.smoke, config_path=config_path)
    device = resolve_device(args.device)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    config_hash = canonical_config_hash(config)
    atomic_write_json(output_root / "config_snapshot.json", config)
    atomic_write_text(output_root / "config_sha256.txt", config_hash + "\n")
    atomic_write_json(output_root / "environment.json", environment_payload(config, device))
    runs = build_run_manifest(config, output_root)
    write_manifest(output_root, config, config_hash, runs)
    atomic_write_json(output_root / "failed_runs.json", {"failed_runs": []})
    if args.plan_only:
        print(json.dumps({
            "plan_only": True,
            "output_root": str(output_root),
            "run_count": len(runs),
            "run_ids": [r["run_id"] for r in runs],
        }, indent=2))
        return 0
    statuses = []
    failures = []
    for run in runs:
        status = run_one(run, config, config_hash, output_root, device, resume=args.resume)
        statuses.append(status)
        if status.get("status") != "complete":
            failures.append(status)
        write_manifest(output_root, config, config_hash, runs, statuses)
        atomic_write_json(output_root / "failed_runs.json", {"failed_runs": failures})
    if args.smoke:
        write_smoke_report(output_root, statuses)
    print(json.dumps({
        "output_root": str(output_root),
        "run_count": len(runs),
        "failed_count": len(failures),
        "status": "complete" if not failures else "failed",
    }, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
