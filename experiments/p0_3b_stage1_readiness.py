"""P0.3b Stage-1 readiness closure runner.

CPU only. This script does not start GPU/AutoDL and does not touch the KBS
manuscript. It freezes the Stage 1a preregistration, validates the D2
nonlinear generator at s0=0.075, verifies chunked exact evaluation, and runs
the local development closure at checkpoint 500 when gates permit.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from factorial_data import (  # noqa: E402
    FACTORIAL_CELLS,
    FACTORIAL_SETTINGS,
    audit_transition_jacobian_support,
    generate_factorial_cell,
    transition_jacobian_fd_spot_check,
)
from repaired_istf import (  # noqa: E402
    RepairedISTFConfig,
    aggregate_window_jacobians_float64,
    canonical_metric_adapter,
    eligible_target_indices,
    evaluate_repaired_model_chunked,
    finite_values_ok,
    horizon_sensitivity_audit,
    instantiate_repaired_method,
    legacy_file_hashes,
    make_cyclic_schedule,
    model_metadata,
    raw_chain_jacobian_chunked_aggregate,
    raw_chain_jacobian_for_windows,
    save_json,
    schedule_hash,
    topk_jaccard,
)
from knowledge_metrics import topk_edges_exact  # noqa: E402


FORMAL_METHODS = ["baseline", "cp_depthwise", "fixed_fir3", "fixed_ema"]
LIMITED_METHODS = ["raw_chain_mamba"]
ALL_CELLS = ["Stat+Linear", "Stat+Nonlinear", "NS+Linear", "NS+Nonlinear"]
CELL_FLAGS = {name: (stationary, linear) for name, stationary, linear in FACTORIAL_CELLS}
STAGE1_CHUNK_SIZE = 64
NOMINAL_LAG = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default=str(PROJECT_ROOT / "results_kbs" / "p0_3b_stage1_readiness"))
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--micro-iters", type=int, default=20)
    parser.add_argument("--runtime-limit-hours", type=float, default=6.0)
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument("--stop-after-runtime", action="store_true")
    return parser.parse_args()


def set_cpu_determinism(threads: int) -> None:
    torch.set_num_threads(max(1, int(threads)))
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        torch.use_deterministic_algorithms(True)


def now_timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


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
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def rss_mb() -> float | None:
    try:
        import psutil  # type: ignore
    except Exception:
        return None
    return float(psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2))


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
        "git_commit": git_commit_hash(),
    }


def base_cfg(horizon: int, method: str | None = None) -> RepairedISTFConfig:
    identity_lam = 0.05
    if method in {"baseline", "fixed_ema", "fixed_fir3"}:
        identity_lam = 0.0
    return RepairedISTFConfig(
        d=6,
        lag=NOMINAL_LAG,
        attribution_horizon=horizon,
        layers=1,
        hidden=16,
        dropout=0.0,
        jacobian_lam=0.01,
        identity_lam=identity_lam,
        residual_gain=0.1,
        depthwise_kernel_size=3,
        d_state=4,
        mamba_expand=2,
        mamba_d_conv=4,
        ema_alpha=0.9,
        fir3_gamma=0.1,
        dtype="float32",
    )


def method_horizon(method: str) -> int:
    return 64 if method == "fixed_ema" else 32


def load_dev_cells(d: int = 6, T: int = 180, lag: int = NOMINAL_LAG, data_seed: int = 0) -> Dict[str, Dict[str, object]]:
    params = FACTORIAL_SETTINGS["D2"]
    out: Dict[str, Dict[str, object]] = {}
    for name in ALL_CELLS:
        stationary, linear = CELL_FLAGS[name]
        x, graph, metadata = generate_factorial_cell(
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
        out[name] = {
            "x": x,
            "gc": graph,
            "metadata": metadata,
            "stationary": stationary,
            "linear": linear,
            "params": dict(params),
        }
    return out


def run_py_compile_and_tests(output_dir: Path) -> Dict[str, object]:
    commands = [
        [
            sys.executable,
            "-m",
            "py_compile",
            "src/factorial_data.py",
            "src/repaired_istf.py",
            "tests/test_p0_factorial_generator.py",
            "tests/test_p0_repaired_istf.py",
            "experiments/p0_3b_stage1_readiness.py",
        ],
        [sys.executable, "tests/test_p0_factorial_generator.py"],
        [sys.executable, "tests/test_p0_repaired_istf.py"],
    ]
    results = []
    lines = []
    for cmd in commands:
        started = time.perf_counter()
        proc = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=600,
        )
        elapsed = time.perf_counter() - started
        results.append({
            "command": cmd,
            "returncode": proc.returncode,
            "elapsed_seconds": elapsed,
            "passed": proc.returncode == 0,
        })
        lines.extend([
            f"COMMAND: {' '.join(cmd)}",
            f"RETURN_CODE: {proc.returncode}",
            f"ELAPSED_SECONDS: {elapsed:.3f}",
            "STDOUT:",
            proc.stdout,
            "STDERR:",
            proc.stderr,
            "",
        ])
    report = {"passed": all(r["passed"] for r in results), "commands": results}
    (output_dir / "test_report.txt").write_text("\n".join(lines), encoding="utf-8")
    return report


def run_generator_calibration() -> Dict[str, object]:
    params = FACTORIAL_SETTINGS["D2"]
    payload: Dict[str, object] = {
        "setting": "D2",
        "d": 10,
        "T": 600,
        "lag": NOMINAL_LAG,
        "data_seeds": list(range(6)),
        "beta": params["nonlinear_strength"],
        "nonlinear_scale_s0": params["nonlinear_scale"],
        "cells": {},
        "failures": [],
    }
    for name in ALL_CELLS:
        payload["cells"][name] = {}
    for seed in range(6):
        for name in ALL_CELLS:
            stationary, linear = CELL_FLAGS[name]
            x, graph, metadata = generate_factorial_cell(
                d=10,
                T=600,
                lag=NOMINAL_LAG,
                seed=seed,
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
            transition = audit_transition_jacobian_support(
                graph,
                metadata["A_t"],
                x,
                linear=linear,
                nonlinear_strength=0.0 if linear else params["nonlinear_strength"],
                nonlinear_scale=params["nonlinear_scale"],
                times=range(NOMINAL_LAG, 600),
            )
            fd = transition_jacobian_fd_spot_check(
                graph,
                metadata["A_t"],
                x,
                linear=linear,
                nonlinear_strength=0.0 if linear else params["nonlinear_strength"],
                nonlinear_scale=params["nonlinear_scale"],
                times=np.linspace(NOMINAL_LAG, 599, num=10, dtype=int).tolist(),
                entries_per_time=8,
            )
            diag = metadata["nonlinear_diagnostics"]
            finite = bool(np.isfinite(x).all() and finite_values_ok(diag) and finite_values_ok(transition))
            gates = {
                "finite_series_and_diagnostics": finite,
                "off_support_transition_derivative_lt_1e_minus_8": transition["max_abs_off_support_derivative"] < 1e-8,
                "actual_transition_support_equals_declared": transition["actual_lag_specific_support_equals_declared"],
                "transition_derivative_ratio_gate": transition["supported_derivative_over_A_gate"],
                "transition_edge_strength_gate": transition["declared_edge_strength_gate_passed"],
                "transition_jacobian_gate_passed": transition["transition_jacobian_gate_passed"],
                "finite_difference_spot_check_passed": fd["passed"],
            }
            if not linear:
                gates.update({
                    "relative_l1_deviation_mean_in_0_05_0_20": (
                        0.05 <= diag["relative_l1_deviation_mean"] <= 0.20
                    ),
                    "near_identity_fraction_mean_lt_0_50": (
                        diag["near_identity_fraction_abs_z_lt_0_1_mean"] < 0.50
                    ),
                    "saturated_fraction_mean_lt_0_10": (
                        diag["saturated_fraction_abs_z_gt_2_mean"] < 0.10
                    ),
                    "per_variable_diagnostics_saved": (
                        "per_variable_relative_l1_deviation_mean_per_variable" in diag
                    ),
                })
            passed = all(bool(v) for v in gates.values())
            record = {
                "passed": passed,
                "gates": gates,
                "nonlinear_diagnostics": jsonable(diag),
                "per_variable_diagnostics": {
                    key: jsonable(value)
                    for key, value in diag.items()
                    if "per_variable" in key
                },
                "support_audit": jsonable(metadata["support_audit"]),
                "transition_jacobian_audit": jsonable(transition),
                "finite_difference_spot_check": jsonable(fd),
            }
            payload["cells"][name][str(seed)] = record
            if not passed:
                payload["failures"].append({"cell": name, "seed": seed, "gates": gates})
    payload["passed"] = len(payload["failures"]) == 0
    return jsonable(payload)


def make_stage1a_frozen_config() -> Dict[str, object]:
    return {
        "primary_checkpoint": 500,
        "formal_methods": ["baseline", "cp_depthwise", "fixed_fir3", "fixed_ema"],
        "limited_ablation": ["raw_chain_mamba"],
        "data_seeds": [1, 2, 3],
        "train_seeds": [0, 1],
        "development_data_seed": 0,
        "four_factorial_cells": ALL_CELLS,
        "generator": {
            "setting": "D2",
            "beta": 0.5,
            "nonlinear_scale_s0": 0.075,
            "d": 10,
            "T": 600,
            "lag": 3,
            "sparsity": 0.2,
        },
        "attribution_horizons": {
            "baseline": 32,
            "cp_depthwise": 32,
            "fixed_fir3": 32,
            "fixed_ema": 64,
            "raw_chain_mamba_limited": 32,
        },
        "chunk_size": 64,
        "evaluation": "full eligible-window exact raw-chain evaluation",
        "statistical_unit": "data_seed; average train_seed=0,1 first per cell/method/data_seed",
        "inference_scope": "effect-size go/no-go triage only; no strong n=3 significance claim",
        "stage1a_go_no_go_gates": {
            "cp_delta_auroc_ge_0_03_in_at_least_two_cells": True,
            "at_least_one_qualifying_cell_nonstationary": True,
            "qualifying_cell_at_least_2_of_3_data_seeds_delta_auroc_positive": True,
            "mean_delta_auprc_ge_minus_0_02": True,
            "mean_delta_mcc_ge_minus_0_05": True,
            "cp_not_matched_or_dominated_by_fixed_fir3_in_all_cells": True,
            "cp_not_dominated_by_ema_full_H_reference_in_all_cells": True,
            "cp_leakage_temporal_horizon_truncation_gates_continue_to_pass": True,
            "learned_kernel_nonzero_change_from_zero_initialization": True,
            "identity_deviation_finite_and_nonzero_reported_no_posthoc_threshold": True,
        },
        "stage1b_confirmatory": {
            "data_seeds": [4, 5, 6, 7, 8],
            "train_seeds": [0, 1],
            "pooled_seeds_1_to_8_only_secondary_summary": True,
        },
    }


def write_stage1a_files(output_dir: Path) -> None:
    cfg = make_stage1a_frozen_config()
    save_json(str(output_dir / "stage1a_frozen_config.json"), cfg)
    commands = [
        "# Stage 1a Frozen Commands",
        "",
        "GPU remains closed until this P0.3b package is externally approved.",
        "",
        "```bash",
        "python experiments/stage1a_gpu_benchmark.py \\",
        "  --setting D2 --d 10 --T 600 --lag 3 --sparsity 0.2 \\",
        "  --data-seeds 1 2 3 --train-seeds 0 1 \\",
        "  --methods baseline cp_depthwise fixed_fir3 fixed_ema \\",
        "  --limited-ablation raw_chain_mamba \\",
        "  --checkpoint 500 --max-iter 500 \\",
        "  --beta 0.5 --nonlinear-scale 0.075 \\",
        "  --horizon-baseline-cp-fir 32 --horizon-ema 64 \\",
        "  --chunk-size 64 --full-window-exact-eval",
        "```",
        "",
    ]
    (output_dir / "stage1a_frozen_commands.md").write_text("\n".join(commands), encoding="utf-8")
    prereg = [
        "# P0.3b Revised Stage 1a Preregistration",
        "",
        "- primary checkpoint: 500",
        "- formal methods: Baseline, CP-depthwise, FixedFIR3, FixedEMA",
        "- limited ablation: RawChainMamba",
        "- data seeds: 1, 2, 3; train seeds: 0, 1",
        "- data_seed=0 is development only",
        "- four cells: Stat+Linear, Stat+Nonlinear, NS+Linear, NS+Nonlinear",
        "- nonlinear generator: beta=0.5, s0=0.075",
        "- attribution horizons: Baseline/CP/FIR H=32, EMA H=64",
        "- chunked evaluator: chunk size=64, float64 streaming accumulator, full eligible-window exact evaluation",
        "- statistical unit: data seed after averaging train seeds",
        "- Stage 1a is effect-size go/no-go triage, not a strong n=3 significance analysis",
        "",
        "## Go/No-Go Gates",
        "",
        "- at least two cells with CP mean Delta AUROC >= 0.03 relative to baseline",
        "- at least one qualifying cell must be non-stationary",
        "- each qualifying cell must have Delta AUROC > 0 in at least 2/3 data seeds",
        "- mean Delta AUPRC >= -0.02",
        "- mean Delta MCC >= -0.05",
        "- CP must not be matched or dominated by FixedFIR3 in all cells",
        "- CP must not be dominated by EMA full-H reference in all cells",
        "- CP leakage, temporal-horizon, and truncation gates must continue to pass",
        "- learned kernel must move nonzero from zero initialization",
        "- identity deviation must be finite and nonzero and fully reported; no posthoc threshold after Stage 1a",
        "",
        "## Stage 1b",
        "",
        "- confirmatory data seeds: 4, 5, 6, 7, 8",
        "- train seeds: 0, 1",
        "- pooled seeds 1..8 may be reported only as a secondary summary",
        "",
    ]
    (output_dir / "revised_preregistration.md").write_text("\n".join(prereg), encoding="utf-8")


def train_step(model, optimizer, x, schedule_entry, target_indices) -> Dict[str, float]:
    optimizer.zero_grad(set_to_none=True)
    comp = model.compute_loss_components(
        x,
        schedule_entry=schedule_entry,
        target_indices=target_indices,
    )
    comp["total_training_objective"].backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    return {key: float(val.detach().cpu()) for key, val in comp.items()}


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
    schedule_digest: str,
    commit_hash: str,
) -> None:
    ckpt_dir = output_dir / "models" / method / cell_name.replace("+", "_").replace(" ", "_")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "config": asdict(model.cfg),
            "method": method,
            "cell": cell_name,
            "iteration": int(iteration),
            "schedule_hash": schedule_digest,
            "commit_hash": commit_hash,
        },
        ckpt_dir / f"iter_{iteration:04d}.pt",
    )


def strip_eval_for_json(eval_out: Dict[str, object]) -> Dict[str, object]:
    skip = {
        "raw_chain_j_bar",
        "score_nominal",
        "score_full_H",
        "filtered_coordinate_j_bar",
        "filtered_coordinate_score_nominal",
    }
    return {k: jsonable(v) for k, v in eval_out.items() if k not in skip}


def chunked_evaluator_parity_and_microbenchmark(output_dir: Path) -> tuple[Dict[str, object], Dict[str, object]]:
    cells = load_dev_cells(d=6, T=180, lag=NOMINAL_LAG, data_seed=0)
    x = cells["NS+Nonlinear"]["x"]
    graph = cells["NS+Nonlinear"]["gc"]
    parity: Dict[str, object] = {"cell": "NS+Nonlinear", "chunk_sizes": [1, 4, 32, 64], "methods": {}, "failures": []}
    micro: Dict[str, object] = {"cell": "NS+Nonlinear", "chunk_sizes": [32, 64], "horizons": [32, 64], "records": []}
    for method in FORMAL_METHODS + LIMITED_METHODS:
        H = method_horizon(method)
        cfg = base_cfg(H, method)
        torch.manual_seed(0)
        model = instantiate_repaired_method(method, cfg)
        target_indices = eligible_target_indices(x.shape[1], cfg.lag, max(H, 64))
        ref_jac, _ = raw_chain_jacobian_for_windows(
            model,
            x,
            target_indices=target_indices,
            attribution_horizon=H,
            create_graph=False,
        )
        ref = aggregate_window_jacobians_float64(ref_jac, lag=cfg.lag)
        ref_metrics = canonical_metric_adapter(graph, ref["score_nominal"])
        k = int(np.sum(graph.sum(axis=2) > 0))
        ref_edges = topk_edges_exact(ref["score_nominal"], k=k, exclude_diag=True)
        parity["methods"][method] = {"target_indices": [int(i) for i in target_indices], "chunks": {}}
        for chunk_size in [1, 4, 32, 64]:
            got = raw_chain_jacobian_chunked_aggregate(
                model,
                x,
                target_indices=target_indices,
                attribution_horizon=H,
                chunk_size=chunk_size,
            )
            metrics = canonical_metric_adapter(graph, got["score_nominal"])
            edges = topk_edges_exact(got["score_nominal"], k=k, exclude_diag=True)
            diffs = {
                "j_bar_max_abs_diff": float(np.max(np.abs(got["j_bar"] - ref["j_bar"]))),
                "score_nominal_max_abs_diff": float(np.max(np.abs(got["score_nominal"] - ref["score_nominal"]))),
                "score_full_H_max_abs_diff": float(np.max(np.abs(got["score_full_H"] - ref["score_full_H"]))),
                "metric_abs_diff": {
                    key: abs(float(metrics[key]) - float(ref_metrics[key]))
                    for key in ["auroc", "auprc", "f1_exact_topk", "mcc_exact_topk", "shd_exact_topk", "nshd_exact_topk"]
                },
                "exact_topk_edges_equal": bool(edges == ref_edges),
            }
            passed = (
                diffs["j_bar_max_abs_diff"] < 1e-7
                and diffs["score_nominal_max_abs_diff"] < 1e-7
                and diffs["score_full_H_max_abs_diff"] < 1e-7
                and diffs["exact_topk_edges_equal"]
                and all(v <= 1e-12 for v in diffs["metric_abs_diff"].values())
            )
            parity["methods"][method]["chunks"][str(chunk_size)] = {"passed": passed, "diffs": diffs}
            if not passed:
                parity["failures"].append({"method": method, "chunk_size": chunk_size, "diffs": diffs})
        for bench_H in [32, 64]:
            bench_cfg = base_cfg(bench_H, method)
            torch.manual_seed(0)
            bench_model = instantiate_repaired_method(method, bench_cfg)
            bench_idx = eligible_target_indices(x.shape[1], bench_cfg.lag, bench_H)
            for chunk_size in [32, 64]:
                before = rss_mb()
                peak = before
                started = time.perf_counter()
                out = raw_chain_jacobian_chunked_aggregate(
                    bench_model,
                    x,
                    target_indices=bench_idx,
                    attribution_horizon=bench_H,
                    chunk_size=chunk_size,
                )
                elapsed = time.perf_counter() - started
                after = rss_mb()
                if after is not None:
                    peak = after if peak is None else max(peak, after)
                bytes_jac = int(chunk_size * bench_cfg.d * bench_cfg.d * bench_H * 8)
                micro["records"].append({
                    "method": method,
                    "H": bench_H,
                    "chunk_size": chunk_size,
                    "eligible_window_count": int(len(bench_idx)),
                    "wall_time_seconds": elapsed,
                    "cpu_peak_rss_mb": peak,
                    "jacobian_output_buffer_lower_bound_mb": bytes_jac / (1024 ** 2),
                    "score_file_size_bytes": int(out["j_bar"].nbytes + out["score_nominal"].nbytes + out["score_full_H"].nbytes),
                    "finite": finite_values_ok(out),
                })
    parity["passed"] = len(parity["failures"]) == 0
    micro["passed"] = all(bool(r["finite"]) for r in micro["records"])
    return jsonable(parity), jsonable(micro)


def training_runtime_estimate(
    cells: Dict[str, Dict[str, object]],
    common_target_indices: np.ndarray,
    schedule: Sequence[Dict[str, List[int]]],
    micro_iters: int,
    train_seed: int,
) -> Dict[str, object]:
    report: Dict[str, object] = {"micro_iters": micro_iters, "records": {}, "failures": []}
    for cell_name, cell in cells.items():
        report["records"][cell_name] = {}
        for method in FORMAL_METHODS + LIMITED_METHODS:
            H = method_horizon(method)
            cfg = base_cfg(H, method)
            torch.manual_seed(train_seed)
            model = instantiate_repaired_method(method, cfg)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.0)
            peak = rss_mb()
            started = time.perf_counter()
            last_loss = {}
            for it in range(micro_iters):
                last_loss = train_step(model, optimizer, cell["x"], schedule[it], common_target_indices)
                mem = rss_mb()
                if mem is not None:
                    peak = mem if peak is None else max(peak, mem)
            elapsed = time.perf_counter() - started
            sec_per_iter = elapsed / max(1, micro_iters)
            est_hours = sec_per_iter * 500.0 / 3600.0
            record = {
                "H": H,
                "wall_time_per_iter_seconds": sec_per_iter,
                "elapsed_seconds": elapsed,
                "estimated_500_iter_hours": est_hours,
                "peak_ram_mb": peak,
                "last_loss": last_loss,
                "finite": finite_values_ok(last_loss),
            }
            report["records"][cell_name][method] = record
    report["passed"] = True
    return jsonable(report)


def full_prefix_omitted_mass(model, x, target_indices: Sequence[int], horizon: int) -> Dict[str, object]:
    idx = [int(i) for i in target_indices if int(i) >= horizon]
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


def run_development_closure(
    output_dir: Path,
    cells: Dict[str, Dict[str, object]],
    common_target_indices: np.ndarray,
    schedule: Sequence[Dict[str, List[int]]],
    schedule_digest: str,
    commit_hash: str,
) -> tuple[Dict[str, object], Dict[str, object], Dict[str, List[str]]]:
    metrics: Dict[str, object] = {}
    diagnostics: Dict[str, object] = {
        "horizon_sensitivity": {},
        "ema_full_prefix_omitted_mass": {},
        "raw_chain_mamba_limited_diagnostic": {},
    }
    failures: List[str] = []
    for cell_name, cell in cells.items():
        x = cell["x"]
        graph = cell["gc"]
        metrics[cell_name] = {}
        diagnostics["horizon_sensitivity"][cell_name] = {}
        diagnostics["ema_full_prefix_omitted_mass"][cell_name] = {}
        for method in FORMAL_METHODS:
            H = method_horizon(method)
            cfg = base_cfg(H, method)
            torch.manual_seed(0)
            model = instantiate_repaired_method(method, cfg)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.0)
            trace = []
            for it in range(1, 501):
                loss_payload = train_step(model, optimizer, x, schedule[it - 1], common_target_indices)
                if it <= 5 or it == 500:
                    trace.append({"iter": it, **loss_payload})
            save_checkpoint(output_dir, method, cell_name, 500, model, optimizer, schedule_digest, commit_hash)
            eval_out = evaluate_repaired_model_chunked(
                model,
                x,
                graph,
                target_indices=common_target_indices,
                attribution_horizon=H,
                chunk_size=STAGE1_CHUNK_SIZE,
                include_filtered_coordinate=True,
                prediction_target_indices=common_target_indices,
                leakage_target_indices=common_target_indices,
            )
            score_dir = output_dir / "scores" / method / cell_name.replace("+", "_").replace(" ", "_")
            save_scores(score_dir, eval_out, 500)
            checkpoint = strip_eval_for_json(eval_out)
            filt_score = eval_out.get("filtered_coordinate_score_nominal")
            if filt_score is not None:
                checkpoint["filtered_vs_raw_nominal"] = {
                    "max_abs_diff": float(np.max(np.abs(eval_out["score_nominal"] - filt_score))),
                    "mean_abs_diff": float(np.mean(np.abs(eval_out["score_nominal"] - filt_score))),
                }
            metrics[cell_name][method] = {
                "metadata": model_metadata(model),
                "training_loss_trace": trace,
                "checkpoints": {"500": checkpoint},
            }
            if method in {"cp_depthwise", "fixed_fir3"}:
                horizon = horizon_sensitivity_audit(
                    model,
                    x,
                    graph,
                    target_indices=common_target_indices,
                    h_small=32,
                    h_large=64,
                )
                diagnostics["horizon_sensitivity"][cell_name][method] = jsonable(horizon)
            if method == "fixed_ema":
                diagnostics["ema_full_prefix_omitted_mass"][cell_name][method] = jsonable(
                    full_prefix_omitted_mass(model, x, common_target_indices, horizon=64)
                )

        # RawChainMamba limited semantic diagnostic: no 500-iter formal trajectory.
        cfg = base_cfg(32, "raw_chain_mamba")
        torch.manual_seed(0)
        mamba = instantiate_repaired_method("raw_chain_mamba", cfg)
        mamba_eval = evaluate_repaired_model_chunked(
            mamba,
            x,
            graph,
            target_indices=common_target_indices,
            attribution_horizon=32,
            chunk_size=STAGE1_CHUNK_SIZE,
            include_filtered_coordinate=False,
            prediction_target_indices=common_target_indices,
            leakage_target_indices=common_target_indices[: min(16, len(common_target_indices))],
        )
        diagnostics["raw_chain_mamba_limited_diagnostic"][cell_name] = strip_eval_for_json(mamba_eval)

    for cell_name, by_method in metrics.items():
        for method, payload in by_method.items():
            ckpt = payload["checkpoints"]["500"]
            if not finite_values_ok(ckpt):
                failures.append(f"{cell_name}/{method}: non-finite checkpoint metrics")
            if method == "cp_depthwise":
                leak = ckpt["cross_variable_leakage"]["cross_variable_leakage"]
                tm = ckpt["temporal_horizon_mass"]
                fd = ckpt["filter_diagnostics"]
                if leak >= 1e-8:
                    failures.append(f"{cell_name}/{method}: leakage={leak:.3e} >= 1e-8")
                if tm["median"] > 0.10 or tm["max"] > 0.20:
                    failures.append(f"{cell_name}/{method}: temporal mass median={tm['median']:.4f}, max={tm['max']:.4f}")
                if fd["kernel_frobenius_norm"] <= 0:
                    failures.append(f"{cell_name}/{method}: learned kernel did not move from zero initialization")
                if not np.isfinite(fd["identity_deviation"]) or fd["identity_deviation"] <= 0:
                    failures.append(f"{cell_name}/{method}: identity deviation not finite/nonzero")
            if method == "fixed_fir3":
                leak = ckpt["cross_variable_leakage"]["cross_variable_leakage"]
                tm = ckpt["temporal_horizon_mass"]
                if leak >= 1e-8:
                    failures.append(f"{cell_name}/{method}: leakage={leak:.3e} >= 1e-8")
                if tm["median"] > 0.10 or tm["max"] > 0.20:
                    failures.append(f"{cell_name}/{method}: temporal mass median={tm['median']:.4f}, max={tm['max']:.4f}")
    for cell_name, by_method in diagnostics["horizon_sensitivity"].items():
        for method, horizon in by_method.items():
            for label in ["H32_vs_H64", "H32_vs_full_prefix"]:
                comp = horizon[label]
                pearson = comp["pearson"].get("value") if isinstance(comp["pearson"], dict) else None
                corr_ok = (pearson is None) or pearson >= 0.99
                if not corr_ok:
                    failures.append(f"{cell_name}/{method}: {label} Pearson={pearson} < 0.99")
                if comp["topk_jaccard"] < 0.95:
                    failures.append(f"{cell_name}/{method}: {label} topk_jaccard={comp['topk_jaccard']:.4f} < 0.95")
            if horizon["omitted_gradient_mass"]["max"] >= 0.01:
                failures.append(
                    f"{cell_name}/{method}: omitted_gradient_mass_max={horizon['omitted_gradient_mass']['max']:.4f} >= 0.01"
                )
    for cell_name, by_method in diagnostics["ema_full_prefix_omitted_mass"].items():
        omitted = by_method["fixed_ema"]["max"]
        if omitted >= 0.01:
            failures.append(f"{cell_name}/fixed_ema: full-prefix omitted mass max={omitted:.4f} >= 0.01")
    gates = {"passed": len(failures) == 0, "failures": failures}
    return jsonable(metrics), jsonable(diagnostics), gates


def write_decision_memo(output_dir: Path, status: str, reasons: Sequence[str], next_action: str) -> None:
    lines = [
        "# P0.3b Stage-1 Readiness Decision Memo",
        "",
        f"- status: {status}",
        f"- generated_at: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "- gpu_used: false",
        "- kbs_manuscript_modified: false",
        "",
        "## Reasons",
    ]
    lines.extend([f"- {reason}" for reason in reasons])
    lines.extend(["", "## Next action", f"- {next_action}", ""])
    (output_dir / "decision_memo.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    set_cpu_determinism(args.threads)
    output_dir = Path(args.output_root) / now_timestamp()
    output_dir.mkdir(parents=True, exist_ok=True)
    commit_hash = git_commit_hash()
    legacy_before = legacy_file_hashes(str(PROJECT_ROOT))

    save_json(str(output_dir / "environment.json"), environment_payload(args))
    save_json(str(output_dir / "stage1a_frozen_config.json"), make_stage1a_frozen_config())
    write_stage1a_files(output_dir)
    config = {
        "plan": "P0.3b Stage-1 Readiness Closure v3",
        "gpu_used": False,
        "kbs_manuscript_modified": False,
        "formal_methods": FORMAL_METHODS,
        "limited_methods": LIMITED_METHODS,
        "development": {"d": 6, "T": 180, "lag": 3, "data_seed": 0, "train_seed": 0, "checkpoint": 500},
        "stage1_chunk_size": STAGE1_CHUNK_SIZE,
        "horizons": {"baseline": 32, "cp_depthwise": 32, "fixed_fir3": 32, "fixed_ema": 64, "raw_chain_mamba": 32},
        "generator": {"setting": "D2", "beta": 0.5, "nonlinear_scale_s0": 0.075},
    }
    save_json(str(output_dir / "config.json"), config)

    status = "P0_3B_IN_PROGRESS"
    reasons: List[str] = []
    if args.skip_tests:
        tests = {"passed": True, "skipped": True}
        (output_dir / "test_report.txt").write_text("SKIPPED by --skip-tests\n", encoding="utf-8")
    else:
        tests = run_py_compile_and_tests(output_dir)
    save_json(str(output_dir / "test_report.json"), tests)
    if not tests.get("passed"):
        status = "P0_3B_FAIL_TESTS"
        reasons.append("py_compile or required tests failed")
        write_decision_memo(output_dir, status, reasons, "Fix tests before any development closure.")
        return 1

    generator = run_generator_calibration()
    save_json(str(output_dir / "generator_nonlinearity_calibration.json"), generator)
    if not generator["passed"]:
        status = "P0_3B_FAIL_GENERATOR_CALIBRATION"
        reasons.append("Generator calibration gates failed; development trajectories stopped by protocol.")
        write_decision_memo(output_dir, status, reasons, "Send failure diagnostics to advisor; do not tune s0.")
        return 1

    parity, chunk_micro = chunked_evaluator_parity_and_microbenchmark(output_dir)
    save_json(str(output_dir / "chunked_evaluator_parity.json"), parity)
    save_json(str(output_dir / "chunked_evaluator_microbenchmark.json"), chunk_micro)
    if not parity["passed"] or not chunk_micro["passed"]:
        status = "P0_3B_FAIL_CHUNKED_EVALUATOR"
        reasons.append("Chunked evaluator parity or microbenchmark finite gate failed.")
        write_decision_memo(output_dir, status, reasons, "Fix chunked evaluator before any development closure.")
        return 1

    cells = load_dev_cells(d=6, T=180, lag=NOMINAL_LAG, data_seed=0)
    common_target_indices = eligible_target_indices(180, NOMINAL_LAG, 64)
    schedule = make_cyclic_schedule(
        common_target_indices,
        d=6,
        max_iter=500,
        windows_per_step=2,
        targets_per_step=2,
        seed=7101,
    )
    schedule_digest = schedule_hash(schedule)
    save_json(str(output_dir / "window_indices.json"), {"common_target_indices": [int(i) for i in common_target_indices]})
    save_json(str(output_dir / "jacobian_sampling_schedule.json"), {"schedule_hash": schedule_digest, "schedule": schedule})
    save_json(str(output_dir / "jacobian_estimator_config.json"), {
        "estimator": "deterministic sampled exact raw-chain Jacobian",
        "create_graph": True,
        "sampled_windows_per_step": 2,
        "sampled_output_targets_per_step": 2,
        "normalization_denominator": "W_sample * target_sample * d_source * nominal_lag",
        "nominal_lag": NOMINAL_LAG,
        "common_target_indices_start": int(common_target_indices[0]),
        "common_target_indices_stop_inclusive": int(common_target_indices[-1]),
    })

    runtime = training_runtime_estimate(cells, common_target_indices, schedule, args.micro_iters, train_seed=0)
    save_json(str(output_dir / "development_runtime_estimate.json"), runtime)
    runtime_failures = []
    for cell_name, by_method in runtime["records"].items():
        for method, rec in by_method.items():
            if rec["estimated_500_iter_hours"] > args.runtime_limit_hours:
                runtime_failures.append({
                    "cell": cell_name,
                    "method": method,
                    "estimated_500_iter_hours": rec["estimated_500_iter_hours"],
                })
    runtime["failures"] = runtime_failures
    runtime["passed"] = len(runtime_failures) == 0
    save_json(str(output_dir / "development_runtime_estimate.json"), runtime)
    if runtime_failures or args.stop_after_runtime:
        status = "P0_3B_STOPPED_AFTER_RUNTIME_ESTIMATE" if not runtime_failures else "P0_3B_FAIL_RUNTIME_BUDGET"
        reasons.append("Runtime estimate completed; full 500-iter closure not started." if not runtime_failures else "Runtime budget gate failed.")
        write_decision_memo(output_dir, status, reasons, "Review runtime report before launching any longer run.")
        return 0 if not runtime_failures else 1

    per_checkpoint, diagnostics, closure_gates = run_development_closure(
        output_dir,
        cells,
        common_target_indices,
        schedule,
        schedule_digest,
        commit_hash,
    )
    save_json(str(output_dir / "per_checkpoint_metrics.json"), per_checkpoint)
    save_json(str(output_dir / "diagnostics.json"), diagnostics)
    save_json(str(output_dir / "horizon_sensitivity.json"), diagnostics.get("horizon_sensitivity", {}))
    save_json(str(output_dir / "stage0_gates.json"), closure_gates)

    legacy_after = legacy_file_hashes(str(PROJECT_ROOT))
    legacy_report = {"before": legacy_before, "after": legacy_after, "unchanged": legacy_before == legacy_after}
    save_json(str(output_dir / "legacy_file_hashes.json"), legacy_report)
    if legacy_before != legacy_after:
        closure_gates["passed"] = False
        closure_gates.setdefault("failures", []).append("Legacy class file hash changed during P0.3b run")

    if closure_gates["passed"]:
        status = "P0_3B_CPU_CLOSURE_COMPLETE"
        reasons.append("Generator calibration, chunked evaluator parity, runtime estimate, and four-cell checkpoint-500 closure completed.")
        next_action = "Package P0.3b output for external advisor review; keep GPU closed until approval."
        rc = 0
    else:
        status = "P0_3B_FAIL_CPU_CLOSURE_GATES"
        reasons.extend(closure_gates.get("failures", []))
        next_action = "Send failure diagnostics to advisor; do not start Stage 1 GPU."
        rc = 1
    write_decision_memo(output_dir, status, reasons, next_action)
    print(str(output_dir))
    print(status)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
