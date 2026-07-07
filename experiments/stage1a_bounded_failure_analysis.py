"""P1 bounded failure analysis for the frozen Stage 1a negative result.

This script is analysis-only. It imports model/generator/metric code from the
approved Stage 1a release checkout and writes new outputs under an independent
analysis directory. It must not modify the frozen Stage 1a artifact.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch


EXPECTED_STAGE1_COMMIT = "65e6ae9afef552c84d8211a9d6e9aa70db48c276"
EXPECTED_SOURCE_MANIFEST_SHA = "be91dc2d3ee916690ebcd519d42811f3d3698ef4eddf053d96cdf96d0f4cab3d"
FORMAL_METHODS = ["baseline", "cp_depthwise", "fixed_fir3", "fixed_ema"]
CELLS = ["Stat+Linear", "Stat+Nonlinear", "NS+Linear", "NS+Nonlinear"]
DATA_SEEDS = [1, 2, 3]
TRAIN_SEEDS = [0, 1]
CHECKPOINT_ITER = 500
EPS = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-root", required=True, help="Approved Stage 1a release checkout.")
    parser.add_argument("--stage1-root", required=True, help="Frozen official Stage 1a result root.")
    parser.add_argument("--output-root", default=None, help="Analysis output root. Defaults under release results_kbs.")
    parser.add_argument("--mode", choices=["inventory-estimate", "full"], default="inventory-estimate")
    parser.add_argument("--device", default="cpu", choices=["cpu"], help="P1 defaults to CPU; GPU is intentionally unsupported.")
    parser.add_argument("--max-total-hours", type=float, default=24.0)
    parser.add_argument("--max-single-task-hours", type=float, default=6.0)
    return parser.parse_args()


def now_stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, default=str), encoding="utf-8")
    os.replace(tmp, path)


def save_csv(path: Path, rows: Sequence[Dict[str, object]], fieldnames: Optional[Sequence[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: List[str] = []
        seen = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    keys.append(key)
        fieldnames = keys
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    os.replace(tmp, path)


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_hash_json(payload) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def setup_release_imports(release_root: Path) -> Dict[str, object]:
    exp = str((release_root / "experiments").resolve())
    src = str((release_root / "src").resolve())
    for p in [exp, src]:
        if p not in sys.path:
            sys.path.insert(0, p)
    modules = {}
    for name in [
        "stage1a_gpu_benchmark",
        "aggregate_stage1a",
        "release_lock",
        "repaired_istf",
        "factorial_data",
        "knowledge_metrics",
    ]:
        modules[name] = importlib.import_module(name)
    return modules


def git_head(path: Path) -> str:
    proc = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(path), capture_output=True, text=True, timeout=30)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip())
    return proc.stdout.strip()


def git_status(path: Path) -> str:
    proc = subprocess.run(["git", "status", "--porcelain"], cwd=str(path), capture_output=True, text=True, timeout=30)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip())
    return proc.stdout


def verify_release_source(release_root: Path) -> Dict[str, object]:
    manifest_path = release_root / "configs" / "release_source_manifest.json"
    manifest_sha = file_sha256(manifest_path)
    failures: List[Dict[str, object]] = []
    if manifest_sha != EXPECTED_SOURCE_MANIFEST_SHA:
        failures.append({
            "type": "source_manifest_sha_mismatch",
            "actual": manifest_sha,
            "expected": EXPECTED_SOURCE_MANIFEST_SHA,
        })
    head = git_head(release_root) if (release_root / ".git").exists() else None
    if head != EXPECTED_STAGE1_COMMIT:
        failures.append({"type": "release_commit_mismatch", "actual": head, "expected": EXPECTED_STAGE1_COMMIT})
    status = git_status(release_root) if (release_root / ".git").exists() else ""
    if status:
        # The release checkout may contain ignored results, but tracked source must be clean.
        failures.append({"type": "release_git_status_not_clean", "status": status})
    manifest = load_json(manifest_path)
    key_failures = []
    for rel, expected_sha in sorted(manifest.get("files", {}).items()):
        p = release_root / rel
        if not p.exists():
            key_failures.append({"file": rel, "type": "missing"})
            continue
        actual_sha = file_sha256(p)
        if actual_sha != expected_sha:
            key_failures.append({"file": rel, "type": "sha_mismatch", "actual": actual_sha, "expected": expected_sha})
    if key_failures:
        failures.append({"type": "key_source_manifest_mismatch", "failures": key_failures})
    out = {
        "release_root": str(release_root),
        "expected_commit": EXPECTED_STAGE1_COMMIT,
        "actual_commit": head,
        "expected_source_manifest_sha256": EXPECTED_SOURCE_MANIFEST_SHA,
        "actual_source_manifest_sha256": manifest_sha,
        "git_status_porcelain": status,
        "key_source_file_count": len(manifest.get("files", {})),
        "passed": not failures,
        "failures": failures,
    }
    if failures:
        raise RuntimeError(f"Release source parity failed: {failures}")
    return out


def role_from_run_id(run_id: str) -> str:
    return "limited_ablation" if run_id.startswith("stage1a_limited__") else "formal"


def method_from_run_id(run_id: str) -> str:
    parts = run_id.split("__")
    return parts[1] if len(parts) > 1 else "unknown"


def build_expected_run_index(stage1_root: Path) -> Tuple[List[Dict[str, object]], Dict[Tuple[str, str, int, int], Dict[str, object]]]:
    manifest = load_json(stage1_root / "run_manifest.json")
    runs = manifest["runs"]
    formal = [r for r in runs if r["role"] == "formal"]
    index = {}
    for r in formal:
        index[(r["cell"], r["method"], int(r["data_seed"]), int(r["train_seed"]))] = r
    return runs, index


def resolve_run_path(stage1_root: Path, run: Dict[str, object]) -> Path:
    p = Path(str(run["output_path"]))
    if p.is_absolute():
        return p
    # Stage 1a manifests store paths relative to the release checkout root.
    # stage1_root = <release_root>/results_kbs/stage1a_...
    release_root = stage1_root.parents[1]
    return release_root / p


def required_run_files(run_dir: Path) -> List[Path]:
    return [
        run_dir / "status.json",
        run_dir / "metrics.json",
        run_dir / "diagnostics.json",
        run_dir / "runtime.json",
        run_dir / "loss_trace.json",
        run_dir / "schedule.json",
        run_dir / "config_snapshot.json",
        run_dir / "config_sha256.txt",
        run_dir / "commit_hash.txt",
        run_dir / "generator_metadata.json",
        run_dir / "checkpoints" / "iter_0500.pt",
        run_dir / "scores" / "raw_chain_j_bar.npy",
        run_dir / "scores" / "score_nominal.npy",
        run_dir / "scores" / "score_full_H.npy",
    ]


def artifact_manifest(stage1_root: Path) -> Dict[str, object]:
    files = []
    for p in sorted(stage1_root.rglob("*")):
        if p.is_file():
            rel = p.relative_to(stage1_root).as_posix()
            files.append({"path": rel, "size_bytes": p.stat().st_size, "sha256": file_sha256(p)})
    return {
        "stage1_root": str(stage1_root),
        "file_count": len(files),
        "total_size_bytes": int(sum(int(f["size_bytes"]) for f in files)),
        "files": files,
    }


def artifact_inventory(stage1_root: Path) -> Dict[str, object]:
    runs, _ = build_expected_run_index(stage1_root)
    failures: List[Dict[str, object]] = []
    seen = set()
    role_counts = {"formal": 0, "limited_ablation": 0}
    method_counts: Dict[str, int] = {}
    for run in runs:
        key = run["run_id"]
        if key in seen:
            failures.append({"type": "duplicate_run_id", "run_id": key})
        seen.add(key)
        role_counts[run["role"]] = role_counts.get(run["role"], 0) + 1
        method_counts[run["method"]] = method_counts.get(run["method"], 0) + 1
        run_dir = resolve_run_path(stage1_root, run)
        if not run_dir.exists():
            failures.append({"type": "missing_run_dir", "run_id": key, "path": str(run_dir)})
            continue
        status_path = run_dir / "status.json"
        status = load_json(status_path) if status_path.exists() else {}
        if status.get("status") != "complete":
            failures.append({"type": "status_not_complete", "run_id": key, "status": status.get("status")})
        for fp in required_run_files(run_dir):
            if not fp.exists() or fp.stat().st_size <= 0:
                failures.append({"type": "missing_or_empty_file", "run_id": key, "path": str(fp)})
    if len(runs) != 100:
        failures.append({"type": "run_count_mismatch", "observed": len(runs), "expected": 100})
    if role_counts.get("formal") != 96 or role_counts.get("limited_ablation") != 4:
        failures.append({"type": "role_count_mismatch", "role_counts": role_counts})
    failed_runs_path = stage1_root / "failed_runs.json"
    if failed_runs_path.exists():
        failed = load_json(failed_runs_path)
        if failed not in ([], {"failed_runs": []}):
            failures.append({"type": "failed_runs_not_empty", "payload": failed})
    return {
        "stage1_root": str(stage1_root),
        "run_count": len(runs),
        "role_counts": role_counts,
        "method_counts": method_counts,
        "missing_count": len([f for f in failures if f.get("type") == "missing_run_dir"]),
        "duplicate_count": len(runs) - len(seen),
        "failure_count": len(failures),
        "passed": not failures,
        "failures": failures[:200],
    }


def metric_track(method: str) -> str:
    return "full_H" if method == "fixed_ema" else "nominal"


def metric_payload(run_dir: Path, method: str) -> Dict[str, float]:
    payload = load_json(run_dir / "metrics.json")
    key = "metrics_full_H" if metric_track(method) == "full_H" else "metrics_nominal"
    return payload[key]


def independent_aggregation(stage1_root: Path, out_dir: Path) -> Dict[str, object]:
    _, index = build_expected_run_index(stage1_root)
    rows = []
    for (cell, method, data_seed, train_seed), run in sorted(index.items()):
        run_dir = resolve_run_path(stage1_root, run)
        metrics = metric_payload(run_dir, method)
        status = load_json(run_dir / "status.json")
        row = {
            "cell": cell,
            "method": method,
            "data_seed": data_seed,
            "train_seed": train_seed,
            "metric_track": metric_track(method),
            "status": status.get("status"),
            "semantic_audit_passed": status.get("semantic_audit_passed"),
            "auroc": float(metrics["auroc"]),
            "auprc": float(metrics["auprc"]),
            "mcc": float(metrics["mcc_exact_topk"]),
            "f1": float(metrics["f1_exact_topk"]),
            "shd": float(metrics["shd_exact_topk"]),
            "nshd": float(metrics["nshd_exact_topk"]),
        }
        rows.append(row)
    save_csv(out_dir / "independent_aggregation_rows.csv", rows)

    averaged: Dict[str, Dict[str, Dict[int, Dict[str, float]]]] = {cell: {} for cell in CELLS}
    for cell in CELLS:
        for method in FORMAL_METHODS:
            averaged[cell][method] = {}
            for data_seed in DATA_SEEDS:
                subset = [r for r in rows if r["cell"] == cell and r["method"] == method and r["data_seed"] == data_seed]
                if sorted(int(r["train_seed"]) for r in subset) != TRAIN_SEEDS:
                    raise RuntimeError(f"Missing train seeds for {cell}/{method}/data{data_seed}")
                averaged[cell][method][data_seed] = {
                    metric: float(np.mean([float(r[metric]) for r in subset]))
                    for metric in ["auroc", "auprc", "mcc", "f1", "shd", "nshd"]
                }

    effects: Dict[str, Dict[str, object]] = {}
    for cell in CELLS:
        effects[cell] = {}
        for label, left, right in [
            ("cp_vs_baseline", "cp_depthwise", "baseline"),
            ("cp_vs_fir3", "cp_depthwise", "fixed_fir3"),
            ("ema_vs_cp", "fixed_ema", "cp_depthwise"),
        ]:
            per_seed = {}
            for data_seed in DATA_SEEDS:
                per_seed[str(data_seed)] = {
                    f"delta_{metric}": averaged[cell][left][data_seed][metric] - averaged[cell][right][data_seed][metric]
                    for metric in ["auroc", "auprc", "mcc", "f1", "shd", "nshd"]
                }
            effects[cell][label] = {
                "left": left,
                "right": right,
                "n_data_seeds": len(DATA_SEEDS),
                "per_data_seed": per_seed,
                "positive_data_seed_count_delta_auroc": int(sum(1 for v in per_seed.values() if v["delta_auroc"] > 0)),
                "mean": {
                    f"delta_{metric}": float(np.mean([v[f"delta_{metric}"] for v in per_seed.values()]))
                    for metric in ["auroc", "auprc", "mcc", "f1", "shd", "nshd"]
                },
            }
    save_json(out_dir / "independent_paired_effects.json", effects)

    frozen = load_json(stage1_root / "stage1a_aggregate_go_no_go.json")
    frozen_effects = frozen["paired_effects_by_data_seed"]
    diffs = []
    for cell in CELLS:
        for label in ["cp_vs_baseline", "cp_vs_fir3", "ema_vs_cp"]:
            ours = effects[cell][label]
            ref = frozen_effects[cell][label]
            for metric in ["delta_auroc", "delta_auprc", "delta_mcc"]:
                diff = abs(float(ours["mean"][metric]) - float(ref["mean"][metric]))
                diffs.append({"cell": cell, "label": label, "field": f"mean.{metric}", "abs_diff": diff})
            if int(ours["positive_data_seed_count_delta_auroc"]) != int(ref["positive_data_seed_count_delta_auroc"]):
                diffs.append({
                    "cell": cell,
                    "label": label,
                    "field": "positive_data_seed_count_delta_auroc",
                    "abs_diff": math.inf,
                })
    max_diff = max(float(d["abs_diff"]) for d in diffs) if diffs else 0.0
    alignment = {
        "passed": bool(max_diff <= 1e-10),
        "tolerance": 1e-10,
        "max_abs_diff": max_diff,
        "diffs": diffs,
    }
    save_json(out_dir / "frozen_aggregator_alignment.json", alignment)
    return {"rows": rows, "averaged": averaged, "effects": effects, "alignment": alignment}


def offdiag_mask(d: int) -> np.ndarray:
    return ~np.eye(d, dtype=bool)


def average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    sorted_vals = values[order]
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_vals[end] == sorted_vals[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    return ranks


def corr_or_reason(a: np.ndarray, b: np.ndarray, rank: bool = False) -> Dict[str, object]:
    va = np.asarray(a, dtype=np.float64).ravel()
    vb = np.asarray(b, dtype=np.float64).ravel()
    if rank:
        va = average_ranks(va)
        vb = average_ranks(vb)
    if np.std(va) < EPS or np.std(vb) < EPS:
        return {"value": None, "undefined_reason": "constant_vector"}
    da = va - float(np.mean(va))
    db = vb - float(np.mean(vb))
    denom = math.sqrt(float(np.sum(da * da) * np.sum(db * db)))
    if denom < EPS:
        return {"value": None, "undefined_reason": "constant_vector"}
    return {"value": float(np.sum(da * db) / denom), "undefined_reason": None}


def topk_set(scores: np.ndarray, k: int, km) -> set:
    return km.topk_edges_exact(scores, k=k, exclude_diag=True)


def topk_edges_exact_np(scores_2d: np.ndarray, k: int) -> set[Tuple[int, int]]:
    scores = np.asarray(scores_2d, dtype=np.float64)
    candidates = []
    for target in range(scores.shape[0]):
        for source in range(scores.shape[1]):
            if target == source:
                continue
            candidates.append((-float(scores[target, source]), target, source))
    candidates.sort()
    return {(source, target) for _, target, source in candidates[: min(int(k), len(candidates))]}


def graph_2d(graph: np.ndarray) -> np.ndarray:
    arr = np.asarray(graph)
    if arr.ndim == 3:
        return (np.sum(arr, axis=2) > 0).astype(np.int32)
    return arr.astype(np.int32)


def edge_set_from_graph(graph: np.ndarray, km) -> set:
    return km.adjacency_to_edge_set(graph_2d(graph), exclude_diag=True)


def reconstruct_cell(config: Dict[str, object], cell: str, data_seed: int, stage1_mod):
    return stage1_mod.generate_cell(config, cell, data_seed)


def load_score(stage1_root: Path, run: Dict[str, object], name: str = "score_nominal.npy") -> np.ndarray:
    return np.load(resolve_run_path(stage1_root, run) / "scores" / name)


def score_pair_stats(a: np.ndarray, b: np.ndarray, graph: np.ndarray, km) -> Dict[str, object]:
    d = a.shape[0]
    mask = offdiag_mask(d)
    va = np.asarray(a, dtype=np.float64)[mask]
    vb = np.asarray(b, dtype=np.float64)[mask]
    k = len(edge_set_from_graph(graph, km))
    ea = topk_set(a, k, km)
    eb = topk_set(b, k, km)
    true_edges = edge_set_from_graph(graph, km)
    fp_a = ea - true_edges
    fp_b = eb - true_edges
    fn_a = true_edges - ea
    fn_b = true_edges - eb
    rank_a = average_ranks(-va)
    rank_b = average_ranks(-vb)
    return {
        "pearson": corr_or_reason(va, vb, rank=False),
        "spearman": corr_or_reason(va, vb, rank=True),
        "exact_topk_jaccard": len(ea & eb) / max(len(ea | eb), 1),
        "max_abs_score_diff": float(np.max(np.abs(va - vb))),
        "mean_abs_score_diff": float(np.mean(np.abs(va - vb))),
        "max_abs_rank_change": float(np.max(np.abs(rank_a - rank_b))),
        "mean_abs_rank_change": float(np.mean(np.abs(rank_a - rank_b))),
        "false_positive_edge_changes": {
            "removed": sorted(list(fp_a - fp_b)),
            "added": sorted(list(fp_b - fp_a)),
        },
        "false_negative_edge_changes": {
            "removed": sorted(list(fn_a - fn_b)),
            "added": sorted(list(fn_b - fn_a)),
        },
        "topk_edges_a": sorted(list(ea)),
        "topk_edges_b": sorted(list(eb)),
    }


def summarize_values(values: Sequence[float]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"median": math.nan, "iqr": math.nan, "min": math.nan, "max": math.nan, "mean": math.nan}
    return {
        "median": float(np.median(arr)),
        "iqr": float(np.percentile(arr, 75) - np.percentile(arr, 25)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
    }


def score_map_equivalence(stage1_root: Path, out_dir: Path, config: Dict[str, object], modules: Dict[str, object]):
    _, index = build_expected_run_index(stage1_root)
    stage1_mod = modules["stage1a_gpu_benchmark"]
    km = modules["knowledge_metrics"]
    rows = []
    details = []
    for cell in CELLS:
        for data_seed in DATA_SEEDS:
            x, graph, _ = reconstruct_cell(config, cell, data_seed, stage1_mod)
            del x
            for train_seed in TRAIN_SEEDS:
                base = index[(cell, "baseline", data_seed, train_seed)]
                cp = index[(cell, "cp_depthwise", data_seed, train_seed)]
                s_base = load_score(stage1_root, base)
                s_cp = load_score(stage1_root, cp)
                stats = score_pair_stats(s_base, s_cp, graph, km)
                pval = stats["pearson"]["value"]
                sval = stats["spearman"]["value"]
                row = {
                    "cell": cell,
                    "data_seed": data_seed,
                    "train_seed": train_seed,
                    "baseline_run_id": base["run_id"],
                    "cp_run_id": cp["run_id"],
                    "baseline_score_sha256": file_sha256(resolve_run_path(stage1_root, base) / "scores" / "score_nominal.npy"),
                    "cp_score_sha256": file_sha256(resolve_run_path(stage1_root, cp) / "scores" / "score_nominal.npy"),
                    "pearson": pval if pval is not None else "",
                    "pearson_undefined_reason": stats["pearson"]["undefined_reason"],
                    "spearman": sval if sval is not None else "",
                    "spearman_undefined_reason": stats["spearman"]["undefined_reason"],
                    "exact_topk_jaccard": stats["exact_topk_jaccard"],
                    "max_abs_score_diff": stats["max_abs_score_diff"],
                    "mean_abs_score_diff": stats["mean_abs_score_diff"],
                    "max_abs_rank_change": stats["max_abs_rank_change"],
                    "mean_abs_rank_change": stats["mean_abs_rank_change"],
                    "fp_removed_count": len(stats["false_positive_edge_changes"]["removed"]),
                    "fp_added_count": len(stats["false_positive_edge_changes"]["added"]),
                    "fn_removed_count": len(stats["false_negative_edge_changes"]["removed"]),
                    "fn_added_count": len(stats["false_negative_edge_changes"]["added"]),
                }
                rows.append(row)
                details.append({**row, "edge_changes": {
                    "false_positive": stats["false_positive_edge_changes"],
                    "false_negative": stats["false_negative_edge_changes"],
                }})
    save_csv(out_dir / "score_map_equivalence.csv", rows)
    save_json(out_dir / "score_map_equivalence.json", {"pairs": details})
    by_cell = {}
    for cell in CELLS:
        subset = [r for r in rows if r["cell"] == cell]
        by_cell[cell] = {
            "pearson": summarize_values([float(r["pearson"]) for r in subset if r["pearson"] != ""]),
            "spearman": summarize_values([float(r["spearman"]) for r in subset if r["spearman"] != ""]),
            "exact_topk_jaccard": summarize_values([float(r["exact_topk_jaccard"]) for r in subset]),
            "max_abs_score_diff": summarize_values([float(r["max_abs_score_diff"]) for r in subset]),
        }
    summary = {
        "overall": {
            "pearson": summarize_values([float(r["pearson"]) for r in rows if r["pearson"] != ""]),
            "spearman": summarize_values([float(r["spearman"]) for r in rows if r["spearman"] != ""]),
            "exact_topk_jaccard": summarize_values([float(r["exact_topk_jaccard"]) for r in rows]),
            "max_abs_score_diff": summarize_values([float(r["max_abs_score_diff"]) for r in rows]),
        },
        "by_cell": by_cell,
    }
    save_json(out_dir / "score_map_equivalence_summary.json", summary)
    return {"rows": rows, "summary": summary}


def instantiate_model_from_checkpoint(stage1_root: Path, method: str, run: Dict[str, object], config: Dict[str, object], modules: Dict[str, object], device: torch.device):
    stage1_mod = modules["stage1a_gpu_benchmark"]
    cfg = stage1_mod.method_cfg(config, method)
    model, _ = stage1_mod.instantiate_paired_method(method, cfg, int(run["data_seed"]), int(run["train_seed"]))
    ckpt = torch.load(resolve_run_path(stage1_root, run) / "checkpoints" / "iter_0500.pt", map_location=device)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.to(device)
    model.eval()
    return model, ckpt


def copy_predictor_state(src_model, dst_model) -> None:
    prefixes = ("inputgate.", "outputgate.", "encoders.")
    src_state = src_model.state_dict()
    dst_state = dst_model.state_dict()
    for key in list(dst_state):
        if key.startswith(prefixes):
            dst_state[key] = src_state[key].detach().clone()
    dst_model.load_state_dict(dst_state, strict=False)


def tensor_sha(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def predictions_for_model(model, x, target_indices: Sequence[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    with torch.no_grad():
        batch = model.make_histories(x, target_indices=target_indices, require_grad=False)
        pred = model(batch["filtered_history"])
        target = batch["raw_target"]
        loss_per_var = torch.mean((pred - target) ** 2, dim=0)
        raw_loss = float(torch.mean((pred - target) ** 2).detach().cpu())
    return (
        pred.detach().cpu().numpy(),
        target.detach().cpu().numpy(),
        loss_per_var.detach().cpu().numpy(),
        raw_loss,
    )


def prediction_pair_stats(pred_a: np.ndarray, pred_b: np.ndarray, target: np.ndarray) -> Dict[str, object]:
    flat_a = pred_a.ravel()
    flat_b = pred_b.ravel()
    corr = corr_or_reason(flat_a, flat_b)
    diff = pred_b - pred_a
    per_var = []
    for j in range(pred_a.shape[1]):
        per_var.append({
            "variable": j,
            "prediction_correlation": corr_or_reason(pred_a[:, j], pred_b[:, j]),
            "mae": float(np.mean(np.abs(diff[:, j]))),
            "rmse": float(math.sqrt(float(np.mean(diff[:, j] ** 2)))),
            "target_variance": float(np.var(target[:, j])),
            "pred_a_variance": float(np.var(pred_a[:, j])),
            "pred_b_variance": float(np.var(pred_b[:, j])),
        })
    return {
        "prediction_correlation": corr,
        "prediction_mae": float(np.mean(np.abs(diff))),
        "prediction_rmse": float(math.sqrt(float(np.mean(diff ** 2)))),
        "per_variable": per_var,
    }


def prediction_equivalence(stage1_root: Path, out_dir: Path, config: Dict[str, object], modules: Dict[str, object], device: torch.device):
    _, index = build_expected_run_index(stage1_root)
    stage1_mod = modules["stage1a_gpu_benchmark"]
    rows = []
    per_var_rows = []
    pred_dir = out_dir / "prediction_arrays"
    pred_dir.mkdir(parents=True, exist_ok=True)
    for cell in CELLS:
        for data_seed in DATA_SEEDS:
            x, graph, _ = reconstruct_cell(config, cell, data_seed, stage1_mod)
            del graph
            idx = stage1_mod.common_target_indices(config)
            for train_seed in TRAIN_SEEDS:
                base_run = index[(cell, "baseline", data_seed, train_seed)]
                cp_run = index[(cell, "cp_depthwise", data_seed, train_seed)]
                base_model, _ = instantiate_model_from_checkpoint(stage1_root, "baseline", base_run, config, modules, device)
                cp_model, _ = instantiate_model_from_checkpoint(stage1_root, "cp_depthwise", cp_run, config, modules, device)
                base_pred, target, base_loss_var, base_loss = predictions_for_model(base_model, x, idx)
                cp_pred, target_cp, cp_loss_var, cp_loss = predictions_for_model(cp_model, x, idx)
                if not np.allclose(target, target_cp):
                    raise RuntimeError(f"Target mismatch in prediction equivalence {cell} data{data_seed} train{train_seed}")
                base_metrics_loss = float(load_json(resolve_run_path(stage1_root, base_run) / "metrics.json")["eval_raw_prediction_loss"])
                cp_metrics_loss = float(load_json(resolve_run_path(stage1_root, cp_run) / "metrics.json")["eval_raw_prediction_loss"])
                base_loss_diff = abs(base_loss - base_metrics_loss)
                cp_loss_diff = abs(cp_loss - cp_metrics_loss)
                if base_loss_diff > 1e-8 or cp_loss_diff > 1e-8:
                    raise RuntimeError(
                        f"Prediction replay loss mismatch {cell}/data{data_seed}/train{train_seed}: "
                        f"baseline {base_loss_diff}, cp {cp_loss_diff}"
                    )
                base_path = pred_dir / f"{base_run['run_id']}_pred.npy"
                cp_path = pred_dir / f"{cp_run['run_id']}_pred.npy"
                np.save(base_path, base_pred)
                np.save(cp_path, cp_pred)
                stats = prediction_pair_stats(base_pred, cp_pred, target)
                row = {
                    "cell": cell,
                    "data_seed": data_seed,
                    "train_seed": train_seed,
                    "baseline_run_id": base_run["run_id"],
                    "cp_run_id": cp_run["run_id"],
                    "baseline_prediction_sha256": file_sha256(base_path),
                    "cp_prediction_sha256": file_sha256(cp_path),
                    "baseline_loss_replay": base_loss,
                    "cp_loss_replay": cp_loss,
                    "eval_raw_loss_diff_cp_minus_baseline": cp_loss - base_loss,
                    "baseline_loss_metrics_abs_diff": base_loss_diff,
                    "cp_loss_metrics_abs_diff": cp_loss_diff,
                    "prediction_correlation": stats["prediction_correlation"]["value"] if stats["prediction_correlation"]["value"] is not None else "",
                    "prediction_correlation_undefined_reason": stats["prediction_correlation"]["undefined_reason"],
                    "prediction_mae": stats["prediction_mae"],
                    "prediction_rmse": stats["prediction_rmse"],
                }
                rows.append(row)
                for item in stats["per_variable"]:
                    per_var_rows.append({
                        "cell": cell,
                        "data_seed": data_seed,
                        "train_seed": train_seed,
                        "variable": item["variable"],
                        "baseline_loss_per_variable": float(base_loss_var[int(item["variable"])]),
                        "cp_loss_per_variable": float(cp_loss_var[int(item["variable"])]),
                        "loss_diff_cp_minus_baseline": float(cp_loss_var[int(item["variable"])] - base_loss_var[int(item["variable"])]),
                        "prediction_correlation": item["prediction_correlation"]["value"] if item["prediction_correlation"]["value"] is not None else "",
                        "prediction_correlation_undefined_reason": item["prediction_correlation"]["undefined_reason"],
                        "prediction_mae": item["mae"],
                        "prediction_rmse": item["rmse"],
                        "target_variance": item["target_variance"],
                        "baseline_prediction_variance": item["pred_a_variance"],
                        "cp_prediction_variance": item["pred_b_variance"],
                    })
    save_csv(out_dir / "prediction_equivalence.csv", rows)
    save_csv(out_dir / "per_variable_prediction_loss.csv", per_var_rows)
    summary = {
        "median_prediction_correlation": float(np.median([float(r["prediction_correlation"]) for r in rows if r["prediction_correlation"] != ""])),
        "median_prediction_mae": float(np.median([float(r["prediction_mae"]) for r in rows])),
        "median_abs_eval_loss_diff": float(np.median([abs(float(r["eval_raw_loss_diff_cp_minus_baseline"])) for r in rows])),
    }
    save_json(out_dir / "prediction_equivalence.json", {"summary": summary, "pairs": rows})
    return {"rows": rows, "summary": summary}


def cp_kernel_lag_order(model) -> np.ndarray:
    w = model.filter.conv.weight.detach().cpu().numpy()[:, 0, :]
    # Conv1d storage order oldest -> current; convert to lag order current, lag1, lag2.
    return w[:, ::-1].astype(np.float64)


def cp_effective_h(model) -> np.ndarray:
    k = cp_kernel_lag_order(model)
    h = model.residual_gain * k
    h[:, 0] += 1.0
    return h


def fir3_effective_h(config: Dict[str, object], modules: Dict[str, object]) -> np.ndarray:
    gamma = float(config["model"]["fixed_fir3"]["gamma"])
    d = int(config["data"]["d"])
    h = np.zeros((d, 3), dtype=np.float64)
    h[:, 0] = 1.0 - 2.0 * gamma / 3.0
    h[:, 1] = gamma / 3.0
    h[:, 2] = gamma / 3.0
    return h


def residual_norm_from_h(h: np.ndarray) -> float:
    delta = np.zeros_like(h)
    delta[:, 0] = 1.0
    return float(np.linalg.norm(h - delta))


def frequency_features(h: np.ndarray, grid_n: int = 512) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    omegas = np.linspace(0.0, math.pi, grid_n)
    feature_rows = []
    response_rows = []
    for var in range(h.shape[0]):
        vals = []
        for omega in omegas:
            resp = sum(h[var, r] * np.exp(-1j * omega * r) for r in range(h.shape[1]))
            mag = abs(resp)
            vals.append(mag)
            response_rows.append({"variable": var, "omega": float(omega), "magnitude": float(mag)})
        mags = np.asarray(vals, dtype=np.float64)
        dc = float(mags[0])
        nyq = float(mags[-1])
        cutoff = ""
        if dc > EPS:
            below = np.where(mags <= dc / math.sqrt(2.0))[0]
            if below.size > 0:
                cutoff = float(omegas[int(below[0])])
        feature_rows.append({
            "variable": var,
            "dc_gain": dc,
            "nyquist_gain": nyq,
            "minus_3db_cutoff": cutoff,
            "total_variation": float(np.sum(np.abs(np.diff(mags)))),
        })
    return feature_rows, response_rows


def learned_filter_audit(stage1_root: Path, out_dir: Path, config: Dict[str, object], modules: Dict[str, object], device: torch.device):
    _, index = build_expected_run_index(stage1_root)
    rows = []
    coef_rows = []
    freq_rows = []
    freq_feature_rows = []
    h_fir = fir3_effective_h(config, modules)
    fir_resid_norm = residual_norm_from_h(h_fir)
    fir_features, fir_response = frequency_features(h_fir)
    for item in fir_features:
        freq_feature_rows.append({"method": "fixed_fir3", "cell": "", "data_seed": "", "train_seed": "", **item})
    for item in fir_response:
        freq_rows.append({"method": "fixed_fir3", "cell": "", "data_seed": "", "train_seed": "", **item})
    for cell in CELLS:
        for data_seed in DATA_SEEDS:
            for train_seed in TRAIN_SEEDS:
                run = index[(cell, "cp_depthwise", data_seed, train_seed)]
                model, _ = instantiate_model_from_checkpoint(stage1_root, "cp_depthwise", run, config, modules, device)
                h = cp_effective_h(model)
                k = cp_kernel_lag_order(model)
                cp_resid_norm = residual_norm_from_h(h)
                diag = load_json(resolve_run_path(stage1_root, run) / "diagnostics.json")["filter_diagnostics"]
                rows.append({
                    "cell": cell,
                    "data_seed": data_seed,
                    "train_seed": train_seed,
                    "run_id": run["run_id"],
                    "identity_deviation": float(diag["identity_deviation"]),
                    "filtered_raw_variance_ratio": float(diag["filtered_raw_variance_ratio"]),
                    "kernel_frobenius_norm": float(np.linalg.norm(k)),
                    "cp_residual_kernel_norm_h_minus_delta": cp_resid_norm,
                    "fixed_fir3_residual_kernel_norm_h_minus_delta": fir_resid_norm,
                    "cp_residual_norm_ratio_to_fir3": cp_resid_norm / max(fir_resid_norm, EPS),
                })
                for var in range(k.shape[0]):
                    for lag in range(k.shape[1]):
                        coef_rows.append({
                            "cell": cell,
                            "data_seed": data_seed,
                            "train_seed": train_seed,
                            "run_id": run["run_id"],
                            "variable": var,
                            "lag": lag,
                            "kernel_k": float(k[var, lag]),
                            "effective_h": float(h[var, lag]),
                            "residual_h_minus_delta": float(h[var, lag] - (1.0 if lag == 0 else 0.0)),
                        })
                features, response = frequency_features(h)
                for item in features:
                    freq_feature_rows.append({"method": "cp_depthwise", "cell": cell, "data_seed": data_seed, "train_seed": train_seed, **item})
                for item in response:
                    freq_rows.append({"method": "cp_depthwise", "cell": cell, "data_seed": data_seed, "train_seed": train_seed, **item})
    save_csv(out_dir / "learned_filter_audit.csv", rows)
    save_csv(out_dir / "learned_filter_coefficients.csv", coef_rows)
    save_csv(out_dir / "filter_frequency_features.csv", freq_feature_rows)
    save_csv(out_dir / "filter_frequency_response.csv", freq_rows)
    kernel_norms = [float(r["kernel_frobenius_norm"]) for r in rows]
    identity_devs = [float(r["identity_deviation"]) for r in rows]
    residual_ratios = [float(r["cp_residual_norm_ratio_to_fir3"]) for r in rows]
    movement = {
        "formal_cp_run_count": len(rows),
        "kernel_norm_gt_1e_minus_6_count": int(sum(v > 1e-6 for v in kernel_norms)),
        "kernel_norm_gt_1e_minus_6_required_count": 18,
        "median_identity_deviation": float(np.median(identity_devs)),
        "median_cp_residual_norm_ratio_to_fir3": float(np.median(residual_ratios)),
        "b_nontrivial_filter_movement_passed": bool(
            sum(v > 1e-6 for v in kernel_norms) >= 18
            and float(np.median(identity_devs)) >= 1e-4
            and float(np.median(residual_ratios)) >= 0.10
        ),
        "fixed_fir3_residual_kernel_norm_h_minus_delta": fir_resid_norm,
    }
    save_json(out_dir / "filter_movement_gate.json", movement)
    return {"rows": rows, "movement": movement}


def build_substitution_model(kind: str, cp_model, cp_run: Dict[str, object], config: Dict[str, object], modules: Dict[str, object], device: torch.device):
    stage1_mod = modules["stage1a_gpu_benchmark"]
    if kind == "learned":
        return cp_model
    cfg = stage1_mod.method_cfg(config, "cp_depthwise")
    if kind == "identity":
        model, _ = stage1_mod.instantiate_paired_method("baseline", cfg, int(cp_run["data_seed"]), int(cp_run["train_seed"]))
    elif kind == "fixed_fir3":
        model, _ = stage1_mod.instantiate_paired_method("fixed_fir3", cfg, int(cp_run["data_seed"]), int(cp_run["train_seed"]))
    else:
        raise ValueError(kind)
    model.to(device)
    copy_predictor_state(cp_model, model)
    model.eval()
    return model


def state_predictor_hash(model) -> str:
    h = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        if key.startswith(("inputgate.", "outputgate.", "encoders.")):
            h.update(key.encode("utf-8"))
            h.update(np.ascontiguousarray(value.detach().cpu().numpy()).tobytes())
    return h.hexdigest()


def evaluate_model_full(model, x, graph, idx, config: Dict[str, object], modules: Dict[str, object]):
    rep = modules["repaired_istf"]
    return rep.evaluate_repaired_model_chunked(
        model,
        x,
        graph,
        target_indices=idx,
        attribution_horizon=model.attribution_horizon,
        chunk_size=int(config["evaluation"]["chunk_size"]),
        include_filtered_coordinate=False,
        prediction_target_indices=idx,
        leakage_target_indices=idx[: min(32, len(idx))],
    )


def counterfactual_substitution(stage1_root: Path, out_dir: Path, config: Dict[str, object], modules: Dict[str, object], device: torch.device, limit: Optional[int] = None):
    _, index = build_expected_run_index(stage1_root)
    stage1_mod = modules["stage1a_gpu_benchmark"]
    rows = []
    score_dir = out_dir / "counterfactual_scores"
    pred_dir = out_dir / "counterfactual_predictions"
    score_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for cell in CELLS:
        for data_seed in DATA_SEEDS:
            x, graph, _ = reconstruct_cell(config, cell, data_seed, stage1_mod)
            true_edge_count = len(edge_set_from_graph(graph, modules["knowledge_metrics"]))
            idx = stage1_mod.common_target_indices(config)
            for train_seed in TRAIN_SEEDS:
                cp_run = index[(cell, "cp_depthwise", data_seed, train_seed)]
                cp_model, _ = instantiate_model_from_checkpoint(stage1_root, "cp_depthwise", cp_run, config, modules, device)
                predictor_hash_before = state_predictor_hash(cp_model)
                for kind in ["learned", "identity", "fixed_fir3"]:
                    if limit is not None and count >= limit:
                        save_csv(out_dir / "counterfactual_filter_substitution_partial.csv", rows)
                        return {"rows": rows, "partial": True}
                    model = build_substitution_model(kind, cp_model, cp_run, config, modules, device)
                    pred_hash_before = state_predictor_hash(model)
                    started = time.perf_counter()
                    out = evaluate_model_full(model, x, graph, idx, config, modules)
                    elapsed = time.perf_counter() - started
                    pred, target, loss_per_var, replay_loss = predictions_for_model(model, x, idx)
                    pred_hash_after = state_predictor_hash(model)
                    if pred_hash_before != pred_hash_after:
                        raise RuntimeError(f"Predictor hash changed during counterfactual eval {cp_run['run_id']} {kind}")
                    s_path = score_dir / f"{cp_run['run_id']}__{kind}__score_nominal.npy"
                    p_path = pred_dir / f"{cp_run['run_id']}__{kind}__prediction.npy"
                    np.save(s_path, out["score_nominal"])
                    np.save(p_path, pred)
                    metrics = out["metrics_nominal"]
                    rows.append({
                        "cell": cell,
                        "data_seed": data_seed,
                        "train_seed": train_seed,
                        "cp_run_id": cp_run["run_id"],
                        "substitution": kind,
                        "true_edge_count": true_edge_count,
                        "predictor_hash_before": pred_hash_before,
                        "predictor_hash_after": pred_hash_after,
                        "cp_predictor_hash": predictor_hash_before,
                        "score_nominal_sha256": file_sha256(s_path),
                        "prediction_sha256": file_sha256(p_path),
                        "eval_raw_prediction_loss": out["eval_raw_prediction_loss"],
                        "prediction_replay_raw_loss": replay_loss,
                        "prediction_replay_loss_abs_diff": abs(float(out["eval_raw_prediction_loss"]) - float(replay_loss)),
                        "auroc": metrics["auroc"],
                        "auprc": metrics["auprc"],
                        "mcc": metrics["mcc_exact_topk"],
                        "f1": metrics["f1_exact_topk"],
                        "semantic_cross_variable_leakage": out["cross_variable_leakage"]["cross_variable_leakage"],
                        "semantic_temporal_horizon_median": out["temporal_horizon_mass"]["median"],
                        "semantic_temporal_horizon_max": out["temporal_horizon_mass"]["max"],
                        "wall_time_seconds": elapsed,
                    })
                    count += 1
    save_csv(out_dir / "counterfactual_filter_substitution.csv", rows)
    save_json(out_dir / "counterfactual_filter_substitution.json", {"rows": rows})
    return {"rows": rows, "partial": False}


def ceiling_effect_audit(stage1_root: Path, out_dir: Path, config: Dict[str, object], modules: Dict[str, object], aggregation: Dict[str, object]):
    _, index = build_expected_run_index(stage1_root)
    stage1_mod = modules["stage1a_gpu_benchmark"]
    km = modules["knowledge_metrics"]
    rows = []
    edge_rows = []
    for cell in CELLS:
        for data_seed in DATA_SEEDS:
            x, graph, _ = reconstruct_cell(config, cell, data_seed, stage1_mod)
            del x
            true_edges = edge_set_from_graph(graph, km)
            k = len(true_edges)
            gt = graph_2d(graph)
            non_edge_count = int(gt.shape[0] * (gt.shape[0] - 1) - len(true_edges))
            for train_seed in TRAIN_SEEDS:
                scores = {
                    method: load_score(stage1_root, index[(cell, method, data_seed, train_seed)])
                    for method in ["baseline", "cp_depthwise", "fixed_fir3"]
                }
                edge_sets = {method: topk_set(score, k, km) for method, score in scores.items()}
                base = edge_sets["baseline"]
                base_fp = base - true_edges
                base_fn = true_edges - base
                for method in ["cp_depthwise", "fixed_fir3"]:
                    pred = edge_sets[method]
                    corrected = sorted(list(base_fp - (pred - true_edges)))
                    broken = sorted(list((pred - true_edges) - base_fp))
                    for e in corrected:
                        edge_rows.append({"cell": cell, "data_seed": data_seed, "train_seed": train_seed, "method": method, "change_type": "corrected_fp", "source": e[0], "target": e[1]})
                    for e in broken:
                        edge_rows.append({"cell": cell, "data_seed": data_seed, "train_seed": train_seed, "method": method, "change_type": "newly_broken_fp", "source": e[0], "target": e[1]})
                base_metrics = metric_payload(resolve_run_path(stage1_root, index[(cell, "baseline", data_seed, train_seed)]), "baseline")
                rows.append({
                    "cell": cell,
                    "data_seed": data_seed,
                    "train_seed": train_seed,
                    "baseline_auroc": float(base_metrics["auroc"]),
                    "baseline_auprc": float(base_metrics["auprc"]),
                    "baseline_mcc": float(base_metrics["mcc_exact_topk"]),
                    "auroc_theoretical_headroom_descriptive_only": float(1.0 - float(base_metrics["auroc"])),
                    "true_edge_count": len(true_edges),
                    "non_edge_count": non_edge_count,
                    "exact_topk_k": k,
                    "baseline_fp_count": len(base_fp),
                    "baseline_fn_count": len(base_fn),
                    "cp_corrected_edges_count": len(base_fp - (edge_sets["cp_depthwise"] - true_edges)),
                    "cp_newly_broken_edges_count": len((edge_sets["cp_depthwise"] - true_edges) - base_fp),
                    "fir3_corrected_edges_count": len(base_fp - (edge_sets["fixed_fir3"] - true_edges)),
                    "fir3_newly_broken_edges_count": len((edge_sets["fixed_fir3"] - true_edges) - base_fp),
                })
    save_csv(out_dir / "ceiling_effect_audit.csv", rows)
    save_csv(out_dir / "baseline_error_edge_tables.csv", edge_rows)
    save_json(out_dir / "ceiling_effect_audit.json", {
        "note": "AUROC headroom is descriptive only and cannot overturn the preregistered Stage 1a no-go.",
        "rows": rows,
    })
    return {"rows": rows}


def estimate_runtime(stage1_root: Path, out_dir: Path, config: Dict[str, object], modules: Dict[str, object], device: torch.device, max_total_hours: float, max_single_task_hours: float):
    _, index = build_expected_run_index(stage1_root)
    stage1_mod = modules["stage1a_gpu_benchmark"]
    cell = "Stat+Linear"
    data_seed = 1
    train_seed = 0
    cp_run = index[(cell, "cp_depthwise", data_seed, train_seed)]
    x, graph, _ = reconstruct_cell(config, cell, data_seed, stage1_mod)
    idx = stage1_mod.common_target_indices(config)
    cp_model, _ = instantiate_model_from_checkpoint(stage1_root, "cp_depthwise", cp_run, config, modules, device)

    # Prediction replay microbenchmark.
    t0 = time.perf_counter()
    predictions_for_model(cp_model, x, idx)
    pred_sec = time.perf_counter() - t0

    # Counterfactual full evaluator microbenchmark: identity substitution is representative.
    identity_model = build_substitution_model("identity", cp_model, cp_run, config, modules, device)
    t0 = time.perf_counter()
    evaluate_model_full(identity_model, x, graph, idx, config, modules)
    eval_sec = time.perf_counter() - t0

    # Gradient decomposition estimate: one checkpoint component set is cheaper than full replay.
    t0 = time.perf_counter()
    one_gradient_decomposition_sample(cp_model, x, idx, config, modules)
    grad_sec = time.perf_counter() - t0

    estimates = {
        "prediction_recompute": {"unit_seconds": pred_sec, "unit_count": 24, "estimated_hours": pred_sec * 24 / 3600.0},
        "counterfactual_evaluations": {"unit_seconds": eval_sec, "unit_count": 72, "estimated_hours": eval_sec * 72 / 3600.0},
        "gradient_decomposition": {"unit_seconds": grad_sec, "unit_count": 16, "estimated_hours": grad_sec * 16 / 3600.0},
        "score_audit": {"unit_seconds": 0.05, "unit_count": 24, "estimated_hours": 0.05 * 24 / 3600.0},
    }
    total = float(sum(v["estimated_hours"] for v in estimates.values()))
    max_single = float(max(v["estimated_hours"] for v in estimates.values()))
    stop = total > max_total_hours or max_single > max_single_task_hours
    out = {
        "cpu_only": True,
        "microbenchmark_run": cp_run["run_id"],
        "estimates": estimates,
        "estimated_total_hours": total,
        "estimated_max_single_task_hours": max_single,
        "max_total_hours": max_total_hours,
        "max_single_task_hours": max_single_task_hours,
        "stop_required": bool(stop),
        "stop_reason": (
            "runtime_estimate_exceeds_budget" if stop else None
        ),
    }
    save_json(out_dir / "runtime_estimate.json", out)
    return out


def one_gradient_decomposition_sample(model, x, idx, config: Dict[str, object], modules: Dict[str, object]) -> Dict[str, object]:
    stage1_mod = modules["stage1a_gpu_benchmark"]
    schedule = stage1_mod.make_cyclic_schedule(
        idx,
        d=int(config["data"]["d"]),
        max_iter=1,
        windows_per_step=int(config["training"]["jacobian_estimator"]["sampled_windows_per_step"]),
        targets_per_step=int(config["training"]["jacobian_estimator"]["sampled_output_targets_per_step"]),
        seed=7101,
    )
    return gradient_components_for_model(model, x, idx, schedule[0])


def flat_filter_grad(model) -> np.ndarray:
    parts = []
    for name, param in model.named_parameters():
        if name.startswith("filter."):
            if param.grad is None:
                parts.append(np.zeros(param.numel(), dtype=np.float64))
            else:
                parts.append(param.grad.detach().cpu().numpy().astype(np.float64).ravel())
    if not parts:
        return np.zeros(0, dtype=np.float64)
    return np.concatenate(parts)


def zero_grads(model) -> None:
    for p in model.parameters():
        p.grad = None


def gradient_components_for_model(model, x, idx, schedule_entry) -> Dict[str, object]:
    # Same fixed prediction batch and Jacobian schedule for all components.
    comp = model.compute_loss_components(x, schedule_entry=schedule_entry, target_indices=idx)
    vectors: Dict[str, np.ndarray] = {}
    for key in ["train_prediction_loss", "raw_chain_jacobian_penalty", "identity_penalty", "total_training_objective"]:
        zero_grads(model)
        comp = model.compute_loss_components(x, schedule_entry=schedule_entry, target_indices=idx)
        comp[key].backward()
        vectors[key] = flat_filter_grad(model)
    g_pred = vectors["train_prediction_loss"]
    g_j = vectors["raw_chain_jacobian_penalty"]
    g_i = vectors["identity_penalty"]
    g_total = vectors["total_training_objective"]
    summed = g_pred + g_j + g_i
    max_abs = float(np.max(np.abs(g_total - summed))) if g_total.size else 0.0
    rel = float(np.linalg.norm(g_total - summed) / max(np.linalg.norm(g_total), EPS)) if g_total.size else 0.0
    return {
        "norm_prediction": float(np.linalg.norm(g_pred)),
        "norm_jacobian": float(np.linalg.norm(g_j)),
        "norm_identity": float(np.linalg.norm(g_i)),
        "norm_total": float(np.linalg.norm(g_total)),
        "r_I_pred": float(np.linalg.norm(g_i) / (np.linalg.norm(g_pred) + EPS)),
        "r_I_J": float(np.linalg.norm(g_i) / (np.linalg.norm(g_j) + EPS)),
        "denominator_pred_near_zero": bool(np.linalg.norm(g_pred) < 1e-12),
        "denominator_j_near_zero": bool(np.linalg.norm(g_j) < 1e-12),
        "cos_identity_prediction": cosine_np(g_i, g_pred),
        "component_sum_max_abs_diff": max_abs,
        "component_sum_relative_l2_diff": rel,
        "component_sum_passed": bool(max_abs <= 1e-6 or rel <= 1e-5),
        "gradient_sha256": {
            key: hashlib.sha256(np.ascontiguousarray(val).tobytes()).hexdigest()
            for key, val in vectors.items()
        },
    }


def cosine_np(a: np.ndarray, b: np.ndarray):
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom < EPS:
        return None
    return float(np.dot(a, b) / denom)


def gradient_decomposition_replay(out_dir: Path, config: Dict[str, object], modules: Dict[str, object], device: torch.device):
    # Bounded replay for data_seed=0 only. It does not inspect seeds 4-8.
    stage1_mod = modules["stage1a_gpu_benchmark"]
    rows = []
    alignment = []
    checkpoint_set = {1, 20, 120, 500}
    for cell in CELLS:
        x, graph, _ = reconstruct_cell(config, cell, 0, stage1_mod)
        del graph
        idx = stage1_mod.common_target_indices(config)
        cfg = stage1_mod.method_cfg(config, "cp_depthwise")
        deterministic = stage1_mod.configure_torch_determinism(stage1_mod.predictor_seed(0, 0), device)
        model, _ = stage1_mod.instantiate_paired_method("cp_depthwise", cfg, 0, 0)
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=float(config["training"]["lr"]), weight_decay=float(config["training"]["weight_decay"]))
        schedule = stage1_mod.make_cyclic_schedule(
            idx,
            d=cfg.d,
            max_iter=500,
            windows_per_step=int(config["training"]["jacobian_estimator"]["sampled_windows_per_step"]),
            targets_per_step=int(config["training"]["jacobian_estimator"]["sampled_output_targets_per_step"]),
            seed=stage1_mod.schedule_seed(0, 0),
        )
        loss_trace = []
        for it in range(1, 501):
            loss_payload = stage1_mod.train_step(model, optimizer, x, schedule[it - 1], idx, grad_clip=float(config["training"]["grad_clip"]))
            loss_trace.append({"iter": it, **loss_payload})
            if it in checkpoint_set:
                components = gradient_components_for_model(model, x, idx, schedule[it - 1])
                row = {"cell": cell, "data_seed": 0, "train_seed": 0, "iteration": it, **components}
                rows.append(row)
        # Alignment to existing development trajectory if available is recorded as not_found unless a local path is supplied later.
        alignment.append({
            "cell": cell,
            "data_seed": 0,
            "train_seed": 0,
            "deterministic_settings": deterministic,
            "existing_development_checkpoint_found": False,
            "alignment_status": "NO_EXISTING_DEVELOPMENT_ARTIFACT_FOUND_FOR_BYTE_ALIGNMENT",
            "loss_trace_sha256": hashlib.sha256(json.dumps(loss_trace, sort_keys=True).encode("utf-8")).hexdigest(),
        })
    save_csv(out_dir / "gradient_decomposition_seed0.csv", rows)
    save_json(out_dir / "gradient_decomposition_seed0.json", {"rows": rows})
    save_json(out_dir / "gradient_replay_alignment.json", {"alignment": alignment})
    valid_ratio = 0
    near_zero = 0
    domination_hits_by_cell: Dict[str, int] = {cell: 0 for cell in CELLS}
    conflict_hits = 0
    for r in rows:
        pred_nz = bool(r["denominator_pred_near_zero"])
        j_nz = bool(r["denominator_j_near_zero"])
        if pred_nz or j_nz:
            near_zero += 1
        else:
            valid_ratio += 1
            if float(r["r_I_pred"]) >= 2.0 or float(r["r_I_J"]) >= 2.0:
                domination_hits_by_cell[str(r["cell"])] += 1
        cos = r["cos_identity_prediction"]
        if cos is not None and float(cos) <= -0.5:
            conflict_hits += 1
    domination_branch = sum(1 for c in CELLS if domination_hits_by_cell[c] >= 3) >= 3
    conflict_branch = conflict_hits >= 8
    replay_alignment_valid = all(
        item.get("alignment_status") == "ALIGNED_WITH_EXISTING_DEVELOPMENT_ARTIFACT"
        for item in alignment
    )
    raw_a3_passed = bool(domination_branch or conflict_branch)
    gate = {
        "valid_ratio_combinations": valid_ratio,
        "near_zero_combinations": near_zero,
        "domination_hits_by_cell": domination_hits_by_cell,
        "conflict_hits": conflict_hits,
        "domination_branch_passed": bool(domination_branch),
        "conflict_branch_passed": bool(conflict_branch),
        "raw_A3_condition_passed_before_replay_alignment": raw_a3_passed,
        "gradient_replay_alignment_valid": bool(replay_alignment_valid),
        "A3_passed": bool(raw_a3_passed and replay_alignment_valid),
        "A3_interpretation_disabled_reason": None if replay_alignment_valid else "existing_data_seed0_development_artifact_not_found_or_not_aligned",
        "component_sum_all_passed": all(bool(r["component_sum_passed"]) for r in rows),
    }
    save_json(out_dir / "gradient_decomposition_gate.json", gate)
    return {"rows": rows, "gate": gate}


def evaluate_a1_a2_b(out_dir: Path, aggregation: Dict[str, object], counterfactual: Optional[Dict[str, object]], filter_audit: Dict[str, object], gradient: Optional[Dict[str, object]]):
    # A1/A2 require counterfactual rows. If runtime stop prevented them, mark invalid.
    result = {
        "A1_cp_learned_vs_identity_equivalence": {"valid": False, "passed": False, "reason": "counterfactual_not_run"},
        "A2_fir3_substitution_substantial_change": {"valid": False, "passed": False, "reason": "counterfactual_not_run"},
        "A3_identity_gradient_domination_or_conflict": {"valid": gradient is not None, "passed": False},
        "B_filter_movement": filter_audit["movement"],
        "B_no_filtering_benefit": {"valid": False, "passed": False, "reason": "counterfactual_not_run"},
    }
    if gradient is not None:
        result["A3_identity_gradient_domination_or_conflict"] = gradient["gate"]
    if counterfactual is not None and counterfactual.get("rows"):
        cf_rows = counterfactual["rows"]
        # Build lookups for substitution metrics.
        by_key = {}
        for r in cf_rows:
            by_key[(r["cell"], int(r["data_seed"]), int(r["train_seed"]), r["substitution"])] = r
        # A1 pair-level learned vs identity.
        a1_hits = 0
        pred_corrs = []
        a1_pair_rows = []
        for cell in CELLS:
            for data_seed in DATA_SEEDS:
                for train_seed in TRAIN_SEEDS:
                    learned = by_key[(cell, data_seed, train_seed, "learned")]
                    ident = by_key[(cell, data_seed, train_seed, "identity")]
                    # Scores are compared from saved counterfactual arrays.
                    s_learned = np.load(out_dir / "counterfactual_scores" / f"{learned['cp_run_id']}__learned__score_nominal.npy")
                    s_ident = np.load(out_dir / "counterfactual_scores" / f"{ident['cp_run_id']}__identity__score_nominal.npy")
                    pearson = corr_or_reason(s_learned[offdiag_mask(s_learned.shape[0])], s_ident[offdiag_mask(s_ident.shape[0])])["value"]
                    spearman = corr_or_reason(s_learned[offdiag_mask(s_learned.shape[0])], s_ident[offdiag_mask(s_ident.shape[0])], rank=True)["value"]
                    k = int(learned["true_edge_count"])
                    e_learned = topk_edges_exact_np(s_learned, k)
                    e_ident = topk_edges_exact_np(s_ident, k)
                    jaccard = len(e_learned & e_ident) / max(len(e_learned | e_ident), 1)
                    p_learned = np.load(out_dir / "counterfactual_predictions" / f"{learned['cp_run_id']}__learned__prediction.npy")
                    p_ident = np.load(out_dir / "counterfactual_predictions" / f"{ident['cp_run_id']}__identity__prediction.npy")
                    pcorr_payload = corr_or_reason(p_learned.ravel(), p_ident.ravel())
                    pcorr = pcorr_payload["value"]
                    dauroc = float(ident["auroc"]) - float(learned["auroc"])
                    dauprc = float(ident["auprc"]) - float(learned["auprc"])
                    rel_loss = abs(float(ident["eval_raw_prediction_loss"]) - float(learned["eval_raw_prediction_loss"])) / max(abs(float(learned["eval_raw_prediction_loss"])), EPS)
                    if pcorr is not None:
                        pred_corrs.append(float(pcorr))
                    cond = (
                        pearson is not None and pearson >= 0.995
                        and spearman is not None and spearman >= 0.995
                        and jaccard >= 0.95
                        and abs(dauroc) <= 0.005
                        and abs(dauprc) <= 0.005
                        and rel_loss <= 0.01
                        and pcorr is not None and pcorr >= 0.995
                    )
                    if cond:
                        a1_hits += 1
                    a1_pair_rows.append({
                        "cell": cell,
                        "data_seed": data_seed,
                        "train_seed": train_seed,
                        "pearson": pearson,
                        "spearman": spearman,
                        "exact_topk_jaccard": jaccard,
                        "delta_auroc_identity_minus_learned": dauroc,
                        "delta_auprc_identity_minus_learned": dauprc,
                        "eval_raw_loss_relative_difference": rel_loss,
                        "prediction_correlation": pcorr,
                        "prediction_correlation_undefined_reason": pcorr_payload["undefined_reason"],
                        "passed": cond,
                    })
        median_pcorr = float(np.median(pred_corrs)) if pred_corrs else None
        result["A1_cp_learned_vs_identity_equivalence"] = {
            "valid": True,
            "passed": bool(a1_hits >= 20 and median_pcorr is not None and median_pcorr >= 0.995),
            "passing_pairs": a1_hits,
            "required_pairs": 20,
            "median_prediction_correlation": median_pcorr,
            "required_median_prediction_correlation": 0.995,
            "pair_rows": a1_pair_rows,
        }
        # A2 using data-seed-level effects.
        cell_hits_auroc = 0
        cell_hits_jaccard = 0
        for cell in CELLS:
            per_seed_dauroc = []
            per_seed_metric_change = []
            per_seed_jacc = []
            for data_seed in DATA_SEEDS:
                vals = []
                metric_changes = []
                jacc_vals = []
                for train_seed in TRAIN_SEEDS:
                    learned = by_key[(cell, data_seed, train_seed, "learned")]
                    fir = by_key[(cell, data_seed, train_seed, "fixed_fir3")]
                    vals.append(float(fir["auroc"]) - float(learned["auroc"]))
                    metric_changes.append(max(
                        abs(float(fir["auroc"]) - float(learned["auroc"])),
                        abs(float(fir["auprc"]) - float(learned["auprc"])),
                        abs(float(fir["mcc"]) - float(learned["mcc"])),
                        abs(float(fir["f1"]) - float(learned["f1"])),
                    ))
                    s_l = np.load(out_dir / "counterfactual_scores" / f"{learned['cp_run_id']}__learned__score_nominal.npy")
                    s_f = np.load(out_dir / "counterfactual_scores" / f"{fir['cp_run_id']}__fixed_fir3__score_nominal.npy")
                    k = int(learned["true_edge_count"])
                    e_l = topk_edges_exact_np(s_l, k)
                    e_f = topk_edges_exact_np(s_f, k)
                    jacc_vals.append(len(e_l & e_f) / max(len(e_l | e_f), 1))
                per_seed_dauroc.append(float(np.mean(vals)))
                per_seed_metric_change.append(float(np.mean(metric_changes)))
                per_seed_jacc.append(float(np.mean(jacc_vals)))
            if float(np.mean(per_seed_dauroc)) >= 0.01 and sum(v > 0 for v in per_seed_dauroc) >= 2:
                cell_hits_auroc += 1
            if float(np.median(per_seed_jacc)) <= 0.80 and max(per_seed_metric_change) >= 0.01:
                cell_hits_jaccard += 1
        result["A2_fir3_substitution_substantial_change"] = {
            "valid": True,
            "passed": bool(cell_hits_auroc >= 2 or cell_hits_jaccard >= 2),
            "auroc_branch_cells": cell_hits_auroc,
            "jaccard_metric_branch_cells": cell_hits_jaccard,
        }
        # B no filtering benefit requires no qualifying cells across four comparisons.
        averaged = aggregation["averaged"]

        def qualifying_from_per_seed(per_seed_values: List[float]) -> bool:
            return float(np.mean(per_seed_values)) >= 0.03 and sum(v > 0 for v in per_seed_values) >= 2

        formal_cp_qual = []
        formal_fir_qual = []
        identity_sub_qual = []
        fir_sub_qual = []
        for cell in CELLS:
            formal_cp_vals = [
                averaged[cell]["cp_depthwise"][data_seed]["auroc"] - averaged[cell]["baseline"][data_seed]["auroc"]
                for data_seed in DATA_SEEDS
            ]
            formal_fir_vals = [
                averaged[cell]["fixed_fir3"][data_seed]["auroc"] - averaged[cell]["baseline"][data_seed]["auroc"]
                for data_seed in DATA_SEEDS
            ]
            identity_vals = []
            fir_vals = []
            for data_seed in DATA_SEEDS:
                base = averaged[cell]["baseline"][data_seed]["auroc"]
                ident_train = [
                    float(by_key[(cell, data_seed, train_seed, "identity")]["auroc"])
                    for train_seed in TRAIN_SEEDS
                ]
                fir_train = [
                    float(by_key[(cell, data_seed, train_seed, "fixed_fir3")]["auroc"])
                    for train_seed in TRAIN_SEEDS
                ]
                identity_vals.append(float(np.mean(ident_train)) - base)
                fir_vals.append(float(np.mean(fir_train)) - base)
            formal_cp_qual.append({"cell": cell, "qualifying": qualifying_from_per_seed(formal_cp_vals), "per_seed_delta_auroc": formal_cp_vals, "mean_delta_auroc": float(np.mean(formal_cp_vals))})
            formal_fir_qual.append({"cell": cell, "qualifying": qualifying_from_per_seed(formal_fir_vals), "per_seed_delta_auroc": formal_fir_vals, "mean_delta_auroc": float(np.mean(formal_fir_vals))})
            identity_sub_qual.append({"cell": cell, "qualifying": qualifying_from_per_seed(identity_vals), "per_seed_delta_auroc": identity_vals, "mean_delta_auroc": float(np.mean(identity_vals))})
            fir_sub_qual.append({"cell": cell, "qualifying": qualifying_from_per_seed(fir_vals), "per_seed_delta_auroc": fir_vals, "mean_delta_auroc": float(np.mean(fir_vals))})
        any_qual = any(item["qualifying"] for group in [formal_cp_qual, formal_fir_qual, identity_sub_qual, fir_sub_qual] for item in group)
        result["B_no_filtering_benefit"] = {
            "valid": True,
            "passed": bool(not any_qual),
            "formal_cp_vs_baseline": formal_cp_qual,
            "formal_fir3_vs_baseline": formal_fir_qual,
            "cp_predictor_identity_substitution_vs_formal_baseline": identity_sub_qual,
            "cp_predictor_fir3_substitution_vs_formal_baseline": fir_sub_qual,
            "note": "Counterfactual substitutions use the CP predictor and are mechanism diagnostics, not independently trained methods.",
        }
    a = result["A1_cp_learned_vs_identity_equivalence"].get("passed") and result["A2_fir3_substitution_substantial_change"].get("passed") and result["A3_identity_gradient_domination_or_conflict"].get("A3_passed", result["A3_identity_gradient_domination_or_conflict"].get("passed", False))
    b = bool(result["B_filter_movement"].get("b_nontrivial_filter_movement_passed")) and bool(result["B_no_filtering_benefit"].get("passed"))
    if a:
        decision = "PARAMETERIZATION_OPTIMIZATION_FAILURE"
    elif b:
        decision = "SHORT_MEMORY_FILTERING_HYPOTHESIS_FAILURE_IN_D2"
    else:
        decision = "INCONCLUSIVE_BOUNDED_POSTMORTEM"
    result["final_decision"] = decision
    save_json(out_dir / "ab_decision_gates.json", result)
    return result


def write_decision_memo(out_dir: Path, payload: Dict[str, object]) -> None:
    lines = [
        "# P1 Bounded Failure Analysis Decision Memo",
        "",
        "## 1. Artifact Integrity",
        json.dumps(payload.get("artifact_inventory", {}), indent=2, sort_keys=True),
        "",
        "## 2. Aggregation Parity",
        json.dumps(payload.get("aggregation_alignment", {}), indent=2, sort_keys=True),
        "",
        "## 3. Score/Prediction Equivalence",
        json.dumps(payload.get("score_summary", {}), indent=2, sort_keys=True),
        json.dumps(payload.get("prediction_summary", {}), indent=2, sort_keys=True),
        "",
        "## 4. Filter Movement",
        json.dumps(payload.get("filter_movement", {}), indent=2, sort_keys=True),
        "",
        "## 5. Counterfactual Substitution",
        json.dumps(payload.get("counterfactual_summary", {}), indent=2, sort_keys=True),
        "",
        "## 6. Gradient Decomposition",
        json.dumps(payload.get("gradient_gate", {}), indent=2, sort_keys=True),
        "",
        "## 7. Ceiling Description",
        "AUROC headroom is descriptive only and does not overturn Stage 1a no-go.",
        "",
        "## 8. A/B Determination",
        json.dumps(payload.get("ab_decision", {}), indent=2, sort_keys=True),
        "",
    ]
    (out_dir / "decision_memo.md").write_text("\n".join(lines), encoding="utf-8")


def package_outputs(out_dir: Path, package_path: Path) -> Dict[str, object]:
    shutil.make_archive(str(package_path.with_suffix("")), "zip", root_dir=out_dir)
    return {
        "path": str(package_path),
        "size_bytes": package_path.stat().st_size,
        "sha256": file_sha256(package_path),
    }


def main() -> int:
    args = parse_args()
    release_root = Path(args.release_root).resolve()
    stage1_root = Path(args.stage1_root).resolve()
    if args.output_root is None:
        output_root = release_root / "results_kbs" / "stage1a_bounded_failure_analysis" / now_stamp()
    else:
        output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")
    modules = setup_release_imports(release_root)
    release_parity = verify_release_source(release_root)
    config = load_json(stage1_root / "config_snapshot.json")
    analysis_commit = None
    try:
        analysis_commit = git_head(Path(__file__).resolve().parents[1])
    except Exception:
        analysis_commit = "unknown"
    environment = {
        "analysis_commit": analysis_commit,
        "stage1_release_commit": EXPECTED_STAGE1_COMMIT,
        "stage1_source_manifest_sha256": EXPECTED_SOURCE_MANIFEST_SHA,
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "device": str(device),
        "mode": args.mode,
    }
    save_json(output_root / "environment.json", environment)
    save_json(output_root / "release_source_parity.json", release_parity)

    inv = artifact_inventory(stage1_root)
    save_json(output_root / "artifact_inventory.json", inv)
    manifest = artifact_manifest(stage1_root)
    save_json(output_root / "artifact_sha256_manifest.json", manifest)
    if not inv["passed"]:
        save_json(output_root / "p1_status.json", {"status": "failed_artifact_inventory", "output_root": str(output_root)})
        return 1

    runtime = estimate_runtime(stage1_root, output_root, config, modules, device, args.max_total_hours, args.max_single_task_hours)
    if runtime["stop_required"]:
        post = artifact_manifest(stage1_root)
        same = canonical_hash_json(manifest) == canonical_hash_json(post)
        save_json(output_root / "artifact_post_analysis_integrity.json", {
            "passed": bool(same),
            "pre_manifest_hash": canonical_hash_json(manifest),
            "post_manifest_hash": canonical_hash_json(post),
        })
        save_json(output_root / "p1_status.json", {
            "status": "stopped_by_runtime_rule",
            "output_root": str(output_root),
            "runtime_estimate": runtime,
        })
        return 2

    aggregation = independent_aggregation(stage1_root, output_root)
    if not aggregation["alignment"]["passed"]:
        save_json(output_root / "aggregation_discrepancy_report.json", aggregation["alignment"])
        save_json(output_root / "p1_status.json", {"status": "failed_aggregation_parity", "output_root": str(output_root)})
        return 1

    score = score_map_equivalence(stage1_root, output_root, config, modules)
    prediction = prediction_equivalence(stage1_root, output_root, config, modules, device)
    filter_audit = learned_filter_audit(stage1_root, output_root, config, modules, device)
    counterfactual = counterfactual_substitution(stage1_root, output_root, config, modules, device)
    ceiling = ceiling_effect_audit(stage1_root, output_root, config, modules, aggregation)
    gradient = gradient_decomposition_replay(output_root, config, modules, device)
    ab = evaluate_a1_a2_b(output_root, aggregation, counterfactual, filter_audit, gradient)
    post = artifact_manifest(stage1_root)
    post_same = canonical_hash_json(manifest) == canonical_hash_json(post)
    post_integrity = {
        "passed": bool(post_same),
        "pre_manifest_hash": canonical_hash_json(manifest),
        "post_manifest_hash": canonical_hash_json(post),
        "pre_file_count": manifest["file_count"],
        "post_file_count": post["file_count"],
    }
    save_json(output_root / "artifact_post_analysis_integrity.json", post_integrity)
    memo_payload = {
        "artifact_inventory": inv,
        "aggregation_alignment": aggregation["alignment"],
        "score_summary": score["summary"],
        "prediction_summary": prediction["summary"],
        "filter_movement": filter_audit["movement"],
        "counterfactual_summary": {"row_count": len(counterfactual["rows"])},
        "gradient_gate": gradient["gate"],
        "ceiling_summary": {"row_count": len(ceiling["rows"])},
        "ab_decision": ab,
        "post_integrity": post_integrity,
    }
    write_decision_memo(output_root, memo_payload)
    package = package_outputs(output_root, release_root / "results_kbs" / "stage1a_bounded_failure_analysis" / "phase7_stage1a_bounded_failure_analysis_v1.zip")
    save_json(output_root / "p1_status.json", {
        "status": "complete" if post_integrity["passed"] else "failed_post_artifact_integrity",
        "output_root": str(output_root),
        "package": package,
        "final_decision": ab["final_decision"],
    })
    print(json.dumps({"output_root": str(output_root), "final_decision": ab["final_decision"], "package": package}, indent=2))
    return 0 if post_integrity["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
