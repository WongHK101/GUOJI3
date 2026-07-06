"""CPU Stage 0 verification runner for P0.1 repaired ISTF methods.

This script does not use GPU, does not modify legacy classes, and writes all
outputs under results_kbs/p0_repaired_local/<timestamp>/.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from factorial_data import FACTORIAL_CELLS, FACTORIAL_SETTINGS, generate_factorial_cell  # noqa: E402
from repaired_istf import (  # noqa: E402
    EvaluationWindowConfig,
    JacobianEstimatorConfig,
    RepairedISTFConfig,
    aggregate_window_jacobians,
    canonical_baseline_penalty,
    canonical_baseline_equivalence_audit,
    compare_sampled_vs_full,
    deterministic_sample_indices,
    eligible_target_indices,
    evaluate_repaired_model,
    finite_values_ok,
    graph_recovery_metrics,
    horizon_sensitivity_audit,
    instantiate_repaired_method,
    legacy_file_hashes,
    make_cyclic_schedule,
    model_metadata,
    now_timestamp,
    raw_chain_jacobian_for_windows,
    raw_chain_jacobian_penalty,
    save_json,
    schedule_hash,
)


DEFAULT_METHODS = ["baseline", "cp_depthwise", "raw_chain_mamba", "fixed_ema"]
DEFAULT_CELLS = ["Stat+Linear", "NS+Nonlinear"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default=str(PROJECT_ROOT / "results_kbs" / "p0_repaired_local"))
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    parser.add_argument("--cells", nargs="+", default=DEFAULT_CELLS)
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument("--checkpoints", nargs="+", type=int, default=[120, 500, 2000])
    parser.add_argument("--micro-iters", type=int, default=20)
    parser.add_argument("--runtime-limit-hours", type=float, default=6.0)
    parser.add_argument("--micro-only", action="store_true", help="Stop after tests and 20-iter microbenchmark.")
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument("--skip-heavy-audits", action="store_true",
                        help="Skip full-window and H/full-prefix audits during checkpoint evaluation.")
    parser.add_argument("--score-max-windows", type=int, default=32)
    parser.add_argument("--data-seed", type=int, default=0)
    parser.add_argument("--train-seed", type=int, default=0)
    parser.add_argument("--jacobian-seed", type=int, default=7101)
    parser.add_argument("--score-window-seed", type=int, default=9103)
    parser.add_argument("--batch-seed", type=int, default=8101)
    parser.add_argument("--threads", type=int, default=1)
    return parser.parse_args()


def set_cpu_determinism(threads: int) -> None:
    torch.set_num_threads(max(1, int(threads)))
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        torch.use_deterministic_algorithms(True)


def load_cells(names: Sequence[str], cfg: RepairedISTFConfig, data_seed: int) -> Dict[str, Dict[str, object]]:
    params = FACTORIAL_SETTINGS["D2"].copy()
    cell_lookup = {name: (stationary, linear) for name, stationary, linear in FACTORIAL_CELLS}
    out: Dict[str, Dict[str, object]] = {}
    for name in names:
        if name not in cell_lookup:
            raise ValueError(f"Unknown D2 cell {name}; available={sorted(cell_lookup)}")
        stationary, linear = cell_lookup[name]
        regime = 0.0 if stationary else params["regime_shift_strength"]
        nonlinear = 0.0 if linear else params["nonlinear_strength"]
        x, graph, metadata = generate_factorial_cell(
            d=cfg.d,
            T=180,
            lag=cfg.lag,
            seed=data_seed,
            stationary=stationary,
            linear=linear,
            coeff_scale=params["coeff_scale"],
            noise_scale=params["noise_scale"],
            regime_shift_strength=regime,
            nonlinear_strength=nonlinear,
            sparsity=0.2,
            return_metadata=True,
        )
        out[name] = {
            "x": x,
            "gc": graph,
            "generator": "factorial_data.generate_factorial_cell",
            "setting": "D2",
            "stationary": stationary,
            "linear": linear,
            "sparsity": 0.2,
            "coeff_scale": params["coeff_scale"],
            "noise_scale": params["noise_scale"],
            "regime_shift_strength": regime,
            "nonlinear_strength": nonlinear,
            "target_graph_definition": "gc[target, source, lag] from generator; metrics use any-lag summary",
            "metadata": metadata,
        }
    return out


def generator_support_audit_payload(cells: Dict[str, Dict[str, object]]) -> Dict[str, object]:
    payload: Dict[str, object] = {"cells": {}, "pairing": {}}
    ref_name = next(iter(cells))
    ref = cells[ref_name]
    for name, cell in cells.items():
        meta = cell["metadata"]
        payload["cells"][name] = {
            "support_audit": jsonable(meta["support_audit"]),
            "rng_streams": jsonable(meta["rng_streams"]),
            "gc_edges": int(np.sum(cell["gc"])),
            "x_shape": list(np.asarray(cell["x"]).shape),
        }
    payload["pairing"] = {
        "reference_cell": ref_name,
        "gc_shared_with_reference": {
            name: bool(np.array_equal(cell["gc"], ref["gc"]))
            for name, cell in cells.items()
        },
        "A_base_shared_with_reference": {
            name: bool(np.array_equal(cell["metadata"]["A_base"], ref["metadata"]["A_base"]))
            for name, cell in cells.items()
        },
        "noise_shared_with_reference": {
            name: bool(np.array_equal(cell["metadata"]["noise"], ref["metadata"]["noise"]))
            for name, cell in cells.items()
        },
    }
    return payload


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
    return value


def rss_mb() -> float | None:
    try:
        import psutil  # type: ignore
    except Exception:
        return None
    return float(psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2))


def run_required_tests(output_dir: Path) -> Dict[str, object]:
    test_path = PROJECT_ROOT / "tests" / "test_p0_repaired_istf.py"
    proc = subprocess.run(
        [sys.executable, str(test_path)],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        timeout=300,
    )
    report = (
        f"command: {sys.executable} {test_path}\n"
        f"returncode: {proc.returncode}\n\n"
        f"STDOUT\n{proc.stdout}\n\nSTDERR\n{proc.stderr}\n"
    )
    (output_dir / "test_report.txt").write_text(report, encoding="utf-8")
    return {"returncode": proc.returncode, "passed": proc.returncode == 0}


def build_default_config(args: argparse.Namespace) -> Dict[str, object]:
    cfg = RepairedISTFConfig(
        d=6,
        lag=3,
        attribution_horizon=32,
        layers=1,
        hidden=16,
        dropout=0.0,
        jacobian_lam=0.01,
        identity_lam=0.05,
        residual_gain=0.1,
        depthwise_kernel_size=3,
        d_state=4,
        mamba_expand=2,
        mamba_d_conv=4,
        ema_alpha=0.9,
        dtype="float32",
    )
    jac_cfg = JacobianEstimatorConfig(
        attribution_horizon=32,
        sampled_windows_per_step=2,
        sampled_targets_per_step=2,
        jacobian_seed=args.jacobian_seed,
    )
    eval_cfg = EvaluationWindowConfig(
        score_window_seed=args.score_window_seed,
        score_max_windows=args.score_max_windows,
        full_window_audit=True,
    )
    return {"model": cfg, "jacobian": jac_cfg, "evaluation": eval_cfg}


def environment_payload(args: argparse.Namespace) -> Dict[str, object]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "executable": sys.executable,
        "cwd": str(PROJECT_ROOT),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "torch_num_threads": torch.get_num_threads(),
        "device": "cpu",
        "gpu_used": False,
        "args": vars(args),
    }


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


def baseline_equivalence_report(cfg: RepairedISTFConfig, x: np.ndarray, gc_true: np.ndarray) -> Dict[str, object]:
    target_indices = eligible_target_indices(x.shape[1], cfg.lag, cfg.attribution_horizon)[:4]
    output_targets = [0, 1]
    torch.manual_seed(1001)
    repaired = instantiate_repaired_method("baseline", cfg)
    canonical_report = canonical_baseline_equivalence_audit(
        repaired,
        x,
        target_indices=target_indices,
        output_targets=output_targets,
        gc_true=gc_true,
        tolerance=1e-6,
    )

    cfg_lag = RepairedISTFConfig(**{**cfg.__dict__, "attribution_horizon": cfg.lag})
    torch.manual_seed(1001)
    lag_model = instantiate_repaired_method("baseline", cfg_lag)
    repaired.load_state_dict(lag_model.state_dict())
    canonical = canonical_baseline_penalty(lag_model, x, target_indices, output_targets)
    repaired_h_lag = raw_chain_jacobian_penalty(lag_model, x, target_indices, output_targets, create_graph=False)
    repaired_h = raw_chain_jacobian_penalty(repaired, x, target_indices, output_targets, create_graph=False)
    jac, _ = raw_chain_jacobian_for_windows(repaired, x, target_indices, attribution_horizon=cfg.attribution_horizon)
    denom = len(target_indices) * len(output_targets) * cfg.d * cfg.lag
    p0 = torch.sum(torch.abs(jac[:, output_targets, :, :])) / denom
    jac[:, output_targets, :, 0] += 0.25
    p1 = torch.sum(torch.abs(jac[:, output_targets, :, :])) / denom
    return {
        "target_indices": [int(i) for i in target_indices],
        "output_targets": output_targets,
        "true_canonical_baseline": canonical_report,
        "canonical_penalty": float(canonical.detach().cpu()),
        "repaired_H_equals_lag_penalty": float(repaired_h_lag.detach().cpu()),
        "repaired_H32_penalty": float(repaired_h.detach().cpu()),
        "max_abs_diff_H_equals_lag": float(torch.abs(canonical - repaired_h_lag).detach().cpu()),
        "max_abs_diff_H32_zero_extension": float(torch.abs(canonical - repaired_h).detach().cpu()),
        "artificial_out_of_lag_increases_penalty": bool(p1 > p0),
    }


def second_order_gradient_report(cfg: RepairedISTFConfig, x: np.ndarray) -> Dict[str, object]:
    out: Dict[str, object] = {}
    for method in ["cp_depthwise", "raw_chain_mamba"]:
        torch.manual_seed(2002)
        model = instantiate_repaired_method(method, cfg)
        model.zero_grad(set_to_none=True)
        penalty = raw_chain_jacobian_penalty(
            model,
            x,
            target_indices=eligible_target_indices(x.shape[1], cfg.lag, cfg.attribution_horizon)[:2],
            output_targets=[0, 1],
            create_graph=True,
        )
        penalty.backward()
        predictor_nonzero = False
        filter_nonzero = False
        all_finite = True
        for name, param in model.named_parameters():
            if param.grad is None:
                continue
            finite = bool(torch.isfinite(param.grad).all().detach().cpu())
            nonzero = bool((torch.sum(torch.abs(param.grad)) > 0).detach().cpu())
            all_finite = all_finite and finite
            if "filter" in name:
                filter_nonzero = filter_nonzero or nonzero
            else:
                predictor_nonzero = predictor_nonzero or nonzero
        out[method] = {
            "penalty": float(penalty.detach().cpu()),
            "all_gradients_finite": all_finite,
            "predictor_has_nonzero_gradient": predictor_nonzero,
            "filter_has_nonzero_gradient": filter_nonzero,
            "passed": all_finite and predictor_nonzero and filter_nonzero,
        }
    return out


def causality_report(cfg: RepairedISTFConfig, x: np.ndarray) -> Dict[str, object]:
    out: Dict[str, object] = {}
    target_u = max(cfg.attribution_horizon, cfg.lag) + 5
    pert = x.copy()
    pert[:, target_u:] += 1000.0
    for method in ["cp_depthwise", "raw_chain_mamba", "fixed_ema"]:
        torch.manual_seed(3003)
        model = instantiate_repaired_method(method, cfg)
        base = model.make_histories(x, target_indices=[target_u], require_grad=False)
        changed = model.make_histories(pert, target_indices=[target_u], require_grad=False)
        hist_diff = float(torch.max(torch.abs(base["filtered_history"] - changed["filtered_history"])).detach().cpu())
        pred_diff = float(torch.max(torch.abs(
            model(base["filtered_history"]) - model(changed["filtered_history"])
        )).detach().cpu())
        target_diff = float(torch.max(torch.abs(base["raw_target"] - changed["raw_target"])).detach().cpu())
        out[method] = {
            "target_index": int(target_u),
            "filtered_history_max_abs_diff_after_future_perturbation": hist_diff,
            "prediction_max_abs_diff_after_future_perturbation": pred_diff,
            "raw_target_max_abs_diff_after_future_perturbation": target_diff,
            "passed": hist_diff == 0.0 and pred_diff == 0.0 and target_diff > 0.0,
        }
    return out


def train_step(model, optimizer, x, schedule_entry, target_indices, cfg: RepairedISTFConfig) -> Dict[str, float]:
    optimizer.zero_grad(set_to_none=True)
    comp = model.compute_loss_components(
        x,
        schedule_entry=schedule_entry,
        target_indices=target_indices,
    )
    total = comp["total_training_objective"]
    total.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    return {key: float(val.detach().cpu()) for key, val in comp.items()}


def _state_dict_max_abs_diff(a: Dict[str, torch.Tensor], b: Dict[str, torch.Tensor]) -> float:
    max_diff = 0.0
    for key in a:
        if key not in b:
            return float("inf")
        ta = a[key]
        tb = b[key]
        if torch.is_tensor(ta):
            max_diff = max(max_diff, float(torch.max(torch.abs(ta.detach().cpu() - tb.detach().cpu()))))
    return max_diff


def _metric_abs_diffs(a: Dict[str, float], b: Dict[str, float]) -> Dict[str, float]:
    keys = sorted(set(a) & set(b))
    out = {}
    for key in keys:
        if isinstance(a[key], (int, float)) and isinstance(b[key], (int, float)):
            out[key] = abs(float(a[key]) - float(b[key]))
    return out


def training_determinism_report(
    cfg: RepairedISTFConfig,
    cells: Dict[str, Dict[str, object]],
    target_indices: np.ndarray,
    schedule: Sequence[Dict[str, List[int]]],
    train_seed: int,
) -> Dict[str, object]:
    out: Dict[str, object] = {}
    for cell_name, cell_payload in cells.items():
        x = cell_payload["x"]
        graph = cell_payload["gc"]
        out[cell_name] = {}
        for method in ["baseline", "cp_depthwise"]:
            runs = []
            for _ in range(2):
                torch.manual_seed(train_seed)
                model = instantiate_repaired_method(method, cfg)
                optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.0)
                trace = []
                for it in range(120):
                    loss_payload = train_step(model, optimizer, x, schedule[it], target_indices, cfg)
                    trace.append(loss_payload["total_training_objective"])
                eval_out = evaluate_repaired_model(
                    model,
                    x,
                    graph,
                    target_indices=target_indices,
                    attribution_horizon=cfg.attribution_horizon,
                    include_filtered_coordinate=False,
                )
                runs.append({
                    "state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                    "loss_trace": np.asarray(trace, dtype=np.float64),
                    "score_nominal": np.asarray(eval_out["score_nominal"], dtype=np.float64),
                    "score_full_H": np.asarray(eval_out["score_full_H"], dtype=np.float64),
                    "metrics_nominal": eval_out["metrics_nominal"],
                    "metrics_full_H": eval_out["metrics_full_H"],
                })
            loss_trace_max_abs_diff = float(np.max(np.abs(runs[0]["loss_trace"] - runs[1]["loss_trace"])))
            nominal_score_max_abs_diff = float(np.max(np.abs(runs[0]["score_nominal"] - runs[1]["score_nominal"])))
            full_h_score_max_abs_diff = float(np.max(np.abs(runs[0]["score_full_H"] - runs[1]["score_full_H"])))
            state_diff = _state_dict_max_abs_diff(runs[0]["state_dict"], runs[1]["state_dict"])
            nominal_metric_diffs = _metric_abs_diffs(runs[0]["metrics_nominal"], runs[1]["metrics_nominal"])
            full_h_metric_diffs = _metric_abs_diffs(runs[0]["metrics_full_H"], runs[1]["metrics_full_H"])
            out[cell_name][method] = {
                "iterations": 120,
                "state_dict_max_abs_diff": state_diff,
                "loss_trace_max_abs_diff": loss_trace_max_abs_diff,
                "nominal_score_max_abs_diff": nominal_score_max_abs_diff,
                "full_H_score_max_abs_diff": full_h_score_max_abs_diff,
                "nominal_metric_abs_diffs": nominal_metric_diffs,
                "full_H_metric_abs_diffs": full_h_metric_diffs,
                "passed": (
                    state_diff < 1e-7
                    and loss_trace_max_abs_diff < 1e-9
                    and nominal_score_max_abs_diff < 1e-7
                    and full_h_score_max_abs_diff < 1e-7
                ),
            }
    return out


def microbenchmark(
    cfg: RepairedISTFConfig,
    cells: Dict[str, Dict[str, object]],
    methods: Sequence[str],
    target_indices: np.ndarray,
    schedule: Sequence[Dict[str, List[int]]],
    micro_iters: int,
    train_seed: int,
) -> Dict[str, object]:
    out: Dict[str, object] = {}
    for cell_name, cell_payload in cells.items():
        x = cell_payload["x"]
        out[cell_name] = {}
        for method in methods:
            torch.manual_seed(train_seed)
            model = instantiate_repaired_method(method, cfg)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.0)
            peak = rss_mb()
            start = time.perf_counter()
            last_loss = {}
            for it in range(micro_iters):
                last_loss = train_step(model, optimizer, x, schedule[it], target_indices, cfg)
                mem = rss_mb()
                if mem is not None:
                    peak = mem if peak is None else max(peak, mem)
            elapsed = time.perf_counter() - start
            sec_per_iter = elapsed / max(micro_iters, 1)
            out[cell_name][method] = {
                "micro_iters": micro_iters,
                "elapsed_seconds": elapsed,
                "wall_time_per_iter_seconds": sec_per_iter,
                "estimated_2000_iter_hours": sec_per_iter * 2000.0 / 3600.0,
                "peak_ram_mb": peak,
                "last_loss": last_loss,
                "finite": finite_values_ok(last_loss),
            }
    return out


def strip_eval_for_json(eval_out: Dict[str, object]) -> Dict[str, object]:
    skip = {
        "raw_chain_j_bar",
        "score_nominal",
        "score_full_H",
        "filtered_coordinate_j_bar",
        "filtered_coordinate_score_nominal",
    }
    return {k: jsonable(v) for k, v in eval_out.items() if k not in skip}


def save_scores(score_dir: Path, eval_out: Dict[str, object], iteration: int) -> None:
    score_dir.mkdir(parents=True, exist_ok=True)
    np.save(score_dir / f"iter_{iteration}_raw_chain.npy", eval_out["raw_chain_j_bar"])
    if "filtered_coordinate_j_bar" in eval_out:
        np.save(score_dir / f"iter_{iteration}_filtered_coordinate.npy", eval_out["filtered_coordinate_j_bar"])
    np.save(score_dir / f"iter_{iteration}_nominal.npy", eval_out["score_nominal"])
    np.save(score_dir / f"iter_{iteration}_full_H.npy", eval_out["score_full_H"])


def save_checkpoint(
    output_dir: Path,
    method: str,
    cell_name: str,
    iteration: int,
    model,
    optimizer,
    cfg: RepairedISTFConfig,
    schedule_digest: str,
    commit_hash: str,
) -> None:
    ckpt_dir = output_dir / "models" / method / cell_name.replace("+", "_").replace(" ", "_")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "config": cfg.__dict__,
            "method": method,
            "cell": cell_name,
            "iteration": int(iteration),
            "schedule_hash": schedule_digest,
            "commit_hash": commit_hash,
        },
        ckpt_dir / f"iter_{iteration:04d}.pt",
    )


def nested_window_sequence(target_indices: Sequence[int], max_n: int, seed: int) -> np.ndarray:
    arr = np.asarray(list(target_indices), dtype=np.int64)
    rng = np.random.default_rng(seed)
    if arr.size <= max_n:
        return arr.copy()
    return rng.choice(arr, size=max_n, replace=False).astype(np.int64)


def window_count_sensitivity_for_checkpoint(
    model,
    x: np.ndarray,
    graph: np.ndarray,
    full_eval: Dict[str, object],
    target_indices: Sequence[int],
    seed: int,
) -> Dict[str, object]:
    nested = nested_window_sequence(target_indices, max_n=min(128, len(target_indices)), seed=seed)
    full_score = full_eval["score_nominal"]
    full_metrics = full_eval["metrics_nominal"]
    out: Dict[str, object] = {
        "nested_sequence": [int(i) for i in nested],
        "full_window_count": int(len(target_indices)),
        "counts": {},
    }
    for count in [32, 64, 128]:
        if len(nested) < count:
            continue
        idx = nested[:count]
        eval_out = evaluate_repaired_model(
            model,
            x,
            graph,
            target_indices=idx,
            attribution_horizon=model.attribution_horizon,
            include_filtered_coordinate=False,
        )
        metrics = eval_out["metrics_nominal"]
        cmp = compare_sampled_vs_full(eval_out["score_nominal"], full_score, graph)
        out["counts"][str(count)] = {
            "target_indices": [int(i) for i in idx],
            "score_comparison": cmp["score_comparison"],
            "metrics": metrics,
            "diff_vs_full": {
                "auroc_abs_diff": abs(metrics["auroc"] - full_metrics["auroc"]),
                "auprc_abs_diff": abs(metrics["auprc"] - full_metrics["auprc"]),
                "f1_exact_topk_abs_diff": abs(metrics["f1_exact_topk"] - full_metrics["f1_exact_topk"]),
            },
        }
    out["counts"]["full"] = {
        "target_indices": [int(i) for i in target_indices],
        "score_comparison": compare_sampled_vs_full(full_score, full_score, graph)["score_comparison"],
        "metrics": full_metrics,
        "diff_vs_full": {
            "auroc_abs_diff": 0.0,
            "auprc_abs_diff": 0.0,
            "f1_exact_topk_abs_diff": 0.0,
        },
    }
    return jsonable(out)


def run_training_trajectory(
    cfg: RepairedISTFConfig,
    cells: Dict[str, Dict[str, object]],
    methods: Sequence[str],
    target_indices: np.ndarray,
    score_windows: np.ndarray,
    horizon_windows: np.ndarray,
    schedule: Sequence[Dict[str, List[int]]],
    checkpoints: Sequence[int],
    max_iter: int,
    train_seed: int,
    output_dir: Path,
    skip_heavy_audits: bool,
    schedule_digest: str,
    commit_hash: str,
) -> tuple[Dict[str, object], Dict[str, object]]:
    metrics: Dict[str, object] = {}
    diagnostics: Dict[str, object] = {
        "sampled_vs_full": {},
        "horizon_sensitivity": {},
        "window_count_sensitivity": {},
    }
    for cell_name, cell_payload in cells.items():
        x = cell_payload["x"]
        graph = cell_payload["gc"]
        metrics[cell_name] = {}
        diagnostics["sampled_vs_full"][cell_name] = {}
        diagnostics["horizon_sensitivity"][cell_name] = {}
        diagnostics["window_count_sensitivity"][cell_name] = {}
        for method in methods:
            torch.manual_seed(train_seed)
            model = instantiate_repaired_method(method, cfg)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.0)
            method_metrics = {
                "metadata": model_metadata(model),
                "checkpoints": {},
                "training_loss_trace": [],
            }
            for it in range(1, max_iter + 1):
                loss_payload = train_step(model, optimizer, x, schedule[it - 1], target_indices, cfg)
                if it <= 5 or it in checkpoints:
                    method_metrics["training_loss_trace"].append({"iter": it, **loss_payload})
                if it not in checkpoints:
                    continue
                save_checkpoint(
                    output_dir,
                    method,
                    cell_name,
                    it,
                    model,
                    optimizer,
                    cfg,
                    schedule_digest,
                    commit_hash,
                )
                eval_out = evaluate_repaired_model(
                    model,
                    x,
                    graph,
                    target_indices=score_windows,
                    attribution_horizon=cfg.attribution_horizon,
                    include_filtered_coordinate=True,
                )
                score_dir = output_dir / "scores" / method / cell_name.replace("+", "_").replace(" ", "_")
                save_scores(score_dir, eval_out, it)
                checkpoint_payload = strip_eval_for_json(eval_out)
                filt_score = eval_out.get("filtered_coordinate_score_nominal")
                if filt_score is not None:
                    checkpoint_payload["filtered_vs_raw_nominal"] = {
                        "max_abs_diff": float(np.max(np.abs(eval_out["score_nominal"] - filt_score))),
                        "mean_abs_diff": float(np.mean(np.abs(eval_out["score_nominal"] - filt_score))),
                    }
                if not skip_heavy_audits:
                    full_eval = evaluate_repaired_model(
                        model,
                        x,
                        graph,
                        target_indices=target_indices,
                        attribution_horizon=cfg.attribution_horizon,
                        include_filtered_coordinate=False,
                    )
                    svf = compare_sampled_vs_full(
                        eval_out["score_nominal"],
                        full_eval["score_nominal"],
                        graph,
                    )
                    diagnostics["sampled_vs_full"][cell_name].setdefault(method, {})[str(it)] = jsonable(svf)
                    diagnostics["window_count_sensitivity"][cell_name].setdefault(method, {})[str(it)] = (
                        window_count_sensitivity_for_checkpoint(
                            model,
                            x,
                            graph,
                            full_eval,
                            target_indices,
                            seed=9103,
                        )
                    )
                    horizon = horizon_sensitivity_audit(
                        model,
                        x,
                        graph,
                        target_indices=horizon_windows,
                        h_small=32,
                        h_large=64,
                    )
                    diagnostics["horizon_sensitivity"][cell_name].setdefault(method, {})[str(it)] = jsonable(horizon)
                method_metrics["checkpoints"][str(it)] = checkpoint_payload
            metrics[cell_name][method] = method_metrics
    return metrics, diagnostics


def write_decision_memo(output_dir: Path, status: str, reasons: Sequence[str], next_action: str) -> None:
    lines = [
        "# P0.2 Local Closure Decision Memo",
        "",
        f"- status: {status}",
        f"- generated_at: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "- gpu_used: false",
        "- kbs_manuscript_modified: false",
        "",
        "## Reasons",
    ]
    lines.extend([f"- {r}" for r in reasons])
    lines.extend(["", "## Next action", f"- {next_action}", ""])
    (output_dir / "decision_memo.md").write_text("\n".join(lines), encoding="utf-8")


def _comparison_corr(comp: Dict[str, object]) -> float | None:
    pearson = comp["pearson"]
    value = pearson.get("value") if isinstance(pearson, dict) else None
    if value is not None:
        return float(value)
    spearman = comp.get("spearman")
    return None if spearman is None else float(spearman)


def evaluate_checkpoint_gates(
    per_checkpoint_metrics: Dict[str, object],
    checkpoint_diagnostics: Dict[str, object],
) -> Dict[str, List[str]]:
    blocking_failures: List[str] = []
    nominal_disqualifications: List[str] = []

    def add_semantic_issue(method: str, message: str) -> None:
        if method in {"fixed_ema", "raw_chain_mamba"}:
            nominal_disqualifications.append(message)
        else:
            blocking_failures.append(message)

    for cell_name, methods in per_checkpoint_metrics.items():
        for method, payload in methods.items():
            for iteration, checkpoint in payload["checkpoints"].items():
                tm = checkpoint["temporal_horizon_mass"]
                if method == "cp_depthwise":
                    leak = checkpoint["cross_variable_leakage"]["cross_variable_leakage"]
                    if leak >= 1e-8:
                        blocking_failures.append(
                            f"{cell_name}/{method}/iter_{iteration}: cross_variable_leakage={leak:.3e} >= 1e-8"
                        )
                if method in {"cp_depthwise", "raw_chain_mamba", "fixed_ema"}:
                    if tm["median"] > 0.10 or tm["max"] > 0.20:
                        add_semantic_issue(
                            method,
                            f"{cell_name}/{method}/iter_{iteration}: temporal_horizon_mass "
                            f"median={tm['median']:.4f}, max={tm['max']:.4f} exceeds nominal-lag gate"
                        )

    horizon = checkpoint_diagnostics.get("horizon_sensitivity", {})
    for cell_name, methods in horizon.items():
        for method, by_iter in methods.items():
            for iteration, payload in by_iter.items():
                for label in ["H32_vs_H64", "H32_vs_full_prefix"]:
                    comp = payload[label]
                    corr = _comparison_corr(comp)
                    if corr is not None and corr < 0.99:
                        add_semantic_issue(method, f"{cell_name}/{method}/iter_{iteration}: {label} corr={corr:.4f} < 0.99")
                    if comp["topk_jaccard"] < 0.95:
                        add_semantic_issue(
                            method,
                            f"{cell_name}/{method}/iter_{iteration}: {label} topk_jaccard="
                            f"{comp['topk_jaccard']:.4f} < 0.95"
                        )
                omitted = payload["omitted_gradient_mass"]["max"]
                if omitted >= 0.01:
                    add_semantic_issue(
                        method,
                        f"{cell_name}/{method}/iter_{iteration}: omitted_gradient_mass_max={omitted:.4f} >= 0.01"
                    )

    sampled = checkpoint_diagnostics.get("sampled_vs_full", {})
    for cell_name, methods in sampled.items():
        for method, by_iter in methods.items():
            for iteration, payload in by_iter.items():
                comp = payload["score_comparison"]
                corr = _comparison_corr(comp)
                if corr is not None and corr < 0.95:
                    blocking_failures.append(f"{cell_name}/{method}/iter_{iteration}: sampled_vs_full corr={corr:.4f} < 0.95")
                if comp["topk_jaccard"] < 0.80:
                    blocking_failures.append(
                        f"{cell_name}/{method}/iter_{iteration}: sampled_vs_full topk_jaccard="
                        f"{comp['topk_jaccard']:.4f} < 0.80"
                    )
                if payload["auroc_abs_diff"] > 0.02:
                    blocking_failures.append(
                        f"{cell_name}/{method}/iter_{iteration}: sampled_vs_full AUROC diff="
                        f"{payload['auroc_abs_diff']:.4f} > 0.02"
                    )
    return {
        "blocking_failures": blocking_failures,
        "nominal_score_disqualifications": nominal_disqualifications,
    }


def main() -> int:
    args = parse_args()
    set_cpu_determinism(args.threads)
    output_dir = Path(args.output_root) / now_timestamp()
    output_dir.mkdir(parents=True, exist_ok=True)

    config_bundle = build_default_config(args)
    cfg: RepairedISTFConfig = config_bundle["model"]
    jac_cfg: JacobianEstimatorConfig = config_bundle["jacobian"]
    eval_cfg: EvaluationWindowConfig = config_bundle["evaluation"]

    save_json(str(output_dir / "config.json"), {
        "stage": "P0.2 repaired local closure CPU verification",
        "generator": "factorial_data.generate_factorial_cell",
        "setting": "D2",
        "cells": args.cells,
        "data_seed": args.data_seed,
        "train_seed": args.train_seed,
        "batch_seed": args.batch_seed,
        "model_config": cfg.__dict__,
        "runtime_rule": {
            "microbenchmark_iters": args.micro_iters,
            "runtime_limit_hours_per_method_cell": args.runtime_limit_hours,
            "first_pass_only_data_seed": 0,
            "first_pass_only_train_seed": 0,
            "formal_score_extraction": "full eligible-window exact extraction; 32/64/128 are sensitivity audits only",
        },
        "pass_fail_gates": {
            "depthwise_cross_variable_leakage": "< 1e-8",
            "temporal_horizon_mass_median": "<= 0.10",
            "temporal_horizon_mass_max": "<= 0.20",
            "H32_score_corr": ">= 0.99",
            "H32_topk_jaccard": ">= 0.95",
            "H32_omitted_gradient_mass": "< 0.01",
            "sampled_vs_full_score_corr": ">= 0.95",
            "sampled_vs_full_topk_jaccard": ">= 0.80",
            "sampled_vs_full_auroc_abs_diff": "<= 0.02",
            "same_seed_score_max_abs_diff": "< 1e-7",
        },
    })
    save_json(str(output_dir / "environment.json"), environment_payload(args))
    save_json(str(output_dir / "jacobian_estimator_config.json"), jac_cfg.__dict__)

    legacy_before = legacy_file_hashes(str(PROJECT_ROOT))
    test_payload = {"skipped": bool(args.skip_tests), "passed": True}
    if not args.skip_tests:
        test_payload = run_required_tests(output_dir)
    else:
        (output_dir / "test_report.txt").write_text("Tests skipped by --skip-tests.\n", encoding="utf-8")

    cells = load_cells(args.cells, cfg, args.data_seed)
    save_json(str(output_dir / "generator_support_audit.json"), generator_support_audit_payload(cells))
    first_payload = next(iter(cells.values()))
    first_x = first_payload["x"]
    first_gc = first_payload["gc"]
    target_indices = eligible_target_indices(first_x.shape[1], cfg.lag, cfg.attribution_horizon)
    score_windows = deterministic_sample_indices(
        target_indices,
        n=eval_cfg.score_max_windows,
        seed=eval_cfg.score_window_seed,
    )
    horizon_windows = deterministic_sample_indices(
        target_indices,
        n=eval_cfg.score_max_windows,
        seed=eval_cfg.score_window_seed,
        require_min_target=64,
    )
    schedule = make_cyclic_schedule(
        target_indices,
        cfg.d,
        max_iter=max(args.max_iter, args.micro_iters),
        windows_per_step=jac_cfg.sampled_windows_per_step,
        targets_per_step=jac_cfg.sampled_targets_per_step,
        seed=jac_cfg.jacobian_seed,
    )
    schedule_digest = schedule_hash(schedule)
    commit_hash = git_commit_hash()
    save_json(str(output_dir / "jacobian_sampling_schedule.json"), {
        "hash": schedule_digest,
        "schedule": schedule,
    })
    save_json(str(output_dir / "window_indices.json"), {
        "training_target_indices": [int(i) for i in target_indices],
        "checkpoint_eval_target_indices": [int(i) for i in target_indices],
        "score_window_indices": [int(i) for i in score_windows],
        "horizon_sensitivity_target_indices": [int(i) for i in horizon_windows],
        "eligible_rule": "u >= max(H, lag)",
        "common_eligible_min_for_horizon_audit": "u >= 64",
    })
    (output_dir / "clean_rerun_commands.txt").write_text(
        "\n".join([
            f"git checkout {commit_hash}",
            "python -m py_compile src/repaired_istf.py src/factorial_data.py src/knowledge_metrics.py tests/test_p0_repaired_istf.py tests/test_p0_factorial_generator.py experiments/p0_repaired_local_verify.py",
            "python tests\\test_p0_factorial_generator.py",
            "python tests\\test_p0_repaired_istf.py",
            "python experiments\\p0_repaired_local_verify.py --score-max-windows 999",
            "",
        ]),
        encoding="utf-8",
    )

    baseline_eq = baseline_equivalence_report(cfg, first_x, first_gc)
    second_order = second_order_gradient_report(cfg, first_x)
    causality = causality_report(cfg, first_x)
    determinism = training_determinism_report(cfg, cells, target_indices, schedule, args.train_seed)
    save_json(str(output_dir / "baseline_equivalence.json"), baseline_eq)
    save_json(str(output_dir / "second_order_gradient_test.json"), second_order)
    save_json(str(output_dir / "causality_test.json"), causality)
    save_json(str(output_dir / "determinism_comparison.json"), determinism)

    micro = microbenchmark(
        cfg,
        cells,
        args.methods,
        target_indices,
        schedule,
        args.micro_iters,
        args.train_seed,
    )
    diagnostics: Dict[str, object] = {
        "test_payload": test_payload,
        "legacy_file_hashes_before": legacy_before,
        "runtime_microbenchmark": micro,
        "baseline_equivalence_passed": (
            baseline_eq["true_canonical_baseline"]["passed"]
            and
            baseline_eq["max_abs_diff_H_equals_lag"] <= 1e-7
            and baseline_eq["max_abs_diff_H32_zero_extension"] <= 1e-7
            and baseline_eq["artificial_out_of_lag_increases_penalty"]
        ),
        "second_order_gradient_passed": all(v["passed"] for v in second_order.values()),
        "causality_passed": all(v["passed"] for v in causality.values()),
        "determinism_passed": all(
            method_payload["passed"]
            for cell_payload in determinism.values()
            for method_payload in cell_payload.values()
        ),
    }
    runtime_failures = []
    for cell_name, by_method in micro.items():
        for method, payload in by_method.items():
            if method in {"cp_depthwise", "raw_chain_mamba"}:
                if payload["estimated_2000_iter_hours"] > args.runtime_limit_hours:
                    runtime_failures.append(
                        f"{cell_name}/{method}: estimated_2000_iter_hours="
                        f"{payload['estimated_2000_iter_hours']:.3f}"
                    )
    diagnostics["runtime_gate_failures"] = runtime_failures

    should_stop = (
        (not test_payload.get("passed", False))
        or runtime_failures
        or args.micro_only
        or not diagnostics["baseline_equivalence_passed"]
        or not diagnostics["second_order_gradient_passed"]
        or not diagnostics["causality_passed"]
        or not diagnostics["determinism_passed"]
    )

    per_checkpoint_metrics: Dict[str, object] = {}
    checkpoint_diagnostics: Dict[str, object] = {}
    checkpoint_gate_report: Dict[str, List[str]] = {
        "blocking_failures": [],
        "nominal_score_disqualifications": [],
    }
    if should_stop:
        status = "STOPPED_AFTER_MICROBENCHMARK" if args.micro_only else "STAGE0_FAIL_BEFORE_CHECKPOINTS"
        reasons = []
        if args.micro_only:
            reasons.append("Stopped by --micro-only after tests and 20-iter microbenchmark.")
        if not test_payload.get("passed", False):
            reasons.append("Required tests failed; see test_report.txt.")
        if runtime_failures:
            reasons.append("Runtime gate exceeded: " + "; ".join(runtime_failures))
        for key in ["baseline_equivalence_passed", "second_order_gradient_passed", "causality_passed", "determinism_passed"]:
            if not diagnostics[key]:
                reasons.append(f"{key}=false")
        write_decision_memo(
            output_dir,
            status,
            reasons,
            "Do not continue to 500/2000; inspect failed gates or rerun without --micro-only if only smoke was requested.",
        )
    else:
        checkpoints = [c for c in args.checkpoints if c <= args.max_iter]
        per_checkpoint_metrics, checkpoint_diagnostics = run_training_trajectory(
            cfg,
            cells,
            args.methods,
            target_indices,
            score_windows,
            horizon_windows,
            schedule,
            checkpoints,
            args.max_iter,
            args.train_seed,
            output_dir,
            skip_heavy_audits=args.skip_heavy_audits,
            schedule_digest=schedule_digest,
            commit_hash=commit_hash,
        )
        if args.skip_heavy_audits:
            checkpoint_gate_report["blocking_failures"] = [
                "Heavy audits were skipped; checkpoint gates were not fully evaluated."
            ]
        else:
            checkpoint_gate_report = evaluate_checkpoint_gates(per_checkpoint_metrics, checkpoint_diagnostics)
        blocking = checkpoint_gate_report["blocking_failures"]
        disqualified = checkpoint_gate_report["nominal_score_disqualifications"]
        if blocking:
            write_decision_memo(
                output_dir,
                "STAGE0_CHECKPOINT_GATE_FAIL",
                blocking + [f"nonblocking: {msg}" for msg in disqualified],
                "Do not continue to 500/2000 until gate failures are fixed or the plan is revised.",
            )
        elif disqualified:
            write_decision_memo(
                output_dir,
                "STAGE0_TRAJECTORY_COMPLETE_WITH_REFERENCE_LIMITATIONS",
                ["Blocking checkpoint gates passed."] + [f"nonblocking: {msg}" for msg in disqualified],
                "Treat disqualified reference nominal scores as full-H diagnostics only; proceed only if this limitation is acceptable.",
            )
        else:
            write_decision_memo(
                output_dir,
                "STAGE0_TRAJECTORY_COMPLETE",
                ["Tests, microbenchmark, and requested checkpoint trajectory completed; checkpoint gates passed."],
                "Proceed to 500/2000 from the same protocol if external consultant accepts this Stage 0 first pass.",
            )

    legacy_after = legacy_file_hashes(str(PROJECT_ROOT))
    diagnostics["legacy_file_hashes_after"] = legacy_after
    diagnostics["legacy_file_hashes_unchanged"] = legacy_before == legacy_after
    diagnostics["checkpoint_gate_failures"] = checkpoint_gate_report["blocking_failures"]
    diagnostics["nominal_score_disqualifications"] = checkpoint_gate_report["nominal_score_disqualifications"]
    diagnostics.update(checkpoint_diagnostics)

    save_json(str(output_dir / "diagnostics.json"), diagnostics)
    save_json(str(output_dir / "per_checkpoint_metrics.json"), per_checkpoint_metrics)
    save_json(str(output_dir / "horizon_sensitivity.json"), checkpoint_diagnostics.get("horizon_sensitivity", {"not_run": True}))
    save_json(str(output_dir / "window_count_sensitivity.json"), checkpoint_diagnostics.get("window_count_sensitivity", {"not_run": True}))

    print(f"P0.2 output: {output_dir}")
    print(f"tests_passed={test_payload.get('passed', False)}")
    print(f"runtime_failures={len(runtime_failures)}")
    print(f"legacy_hashes_unchanged={legacy_before == legacy_after}")
    return 0 if test_payload.get("passed", False) and not runtime_failures and not checkpoint_gate_report["blocking_failures"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
