"""Aggregate the bounded Phase 9 RTX 4090 development round."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


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


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def finite_number(value) -> float | None:
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def grouped_means(
    rows: Sequence[Mapping[str, object]],
    *,
    group_fields: Sequence[str],
    metric_fields: Sequence[str],
) -> List[Dict[str, object]]:
    groups: Dict[tuple, List[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[field] for field in group_fields)].append(row)
    output = []
    for key, members in sorted(groups.items()):
        record = {field: value for field, value in zip(group_fields, key)}
        record["n"] = len(members)
        for metric in metric_fields:
            values = [
                finite_number(member.get(metric))
                for member in members
                if finite_number(member.get(metric)) is not None
            ]
            record[f"{metric}_mean"] = None if not values else float(np.mean(values))
            record[f"{metric}_min"] = None if not values else float(np.min(values))
            record[f"{metric}_max"] = None if not values else float(np.max(values))
        output.append(record)
    return output


def read_d2_root(root: Path) -> List[Dict[str, object]]:
    rows = []
    for run_dir in sorted((root / "d2_adaptive_fir").glob("*")):
        if not run_dir.is_dir():
            continue
        config = load_json(run_dir / "config.json")
        metrics = load_json(run_dir / "metrics.json")
        diagnostics = load_json(run_dir / "diagnostics.json")
        status = load_json(run_dir / "status.json")
        filt = diagnostics["filter"]
        rows.append({
            "root": root.name,
            "run_id": config["run_id"],
            "cell": config["cell"],
            "method": config["method"],
            "data_seed": int(config["data_seed"]),
            "train_seed": int(config["train_seed"]),
            "max_iter": int(config["max_iter"]),
            "auroc": metrics["nominal"]["auroc"],
            "auprc": metrics["nominal"]["auprc"],
            "mcc": metrics["nominal"]["mcc_exact_topk"],
            "f1": metrics["nominal"]["f1_exact_topk"],
            "mse": metrics["eval_raw_prediction_loss"],
            "identity_deviation": filt.get("identity_deviation"),
            "gate_mean": filt.get("contextual_gate_mean", filt.get("gate_mean")),
            "gate_std": filt.get("contextual_gate_std"),
            "context_kernel_norm": filt.get("context_kernel_frobenius_norm"),
            "temporal_mass_median": diagnostics["temporal_horizon_mass"]["median"],
            "cross_variable_leakage": diagnostics["cross_variable_leakage"][
                "cross_variable_leakage"
            ],
            "wall_seconds": status["wall_seconds"],
        })
    return rows


def read_phase8_root(root: Path) -> List[Dict[str, object]]:
    rows = []
    for run_dir in sorted((root / "phase8_gradient_guard").glob("*")):
        if not run_dir.is_dir():
            continue
        config = load_json(run_dir / "config.json")
        evaluation = load_json(run_dir / "evaluation.json")
        diagnostics = load_json(run_dir / "diagnostics.json")
        status = load_json(run_dir / "status.json")
        projection = diagnostics.get("projection") or {}
        rows.append({
            "root": root.name,
            "run_id": config["run_id"],
            "method": config["method"],
            "data_seed": int(config["data_seed"]),
            "train_seed": int(config["train_seed"]),
            "max_iter": int(config["max_iter"]),
            "auroc": evaluation["metrics_total_nominal"]["auroc"],
            "auprc": evaluation["metrics_total_nominal"]["auprc"],
            "mcc": evaluation["metrics_total_nominal"]["mcc_exact_topk"],
            "f1": evaluation["metrics_total_nominal"]["f1_exact_topk"],
            "mse": evaluation["fixed_target_prediction_mse"],
            "coefficient_r": evaluation["coefficient_r_total_lag1"],
            "missing_route": evaluation["m_missing"],
            "conflict_fraction": projection.get("conflict_fraction"),
            "norm_cap_fraction": projection.get("norm_cap_fraction"),
            "gradient_ratio_median": projection.get(
                "median_raw_coverage_to_prediction_gradient_ratio"
            ),
            "wall_seconds": status["wall_seconds"],
        })
    return rows


def read_audit_root(root: Path) -> List[Dict[str, object]]:
    rows = []
    for run_dir in sorted(root.glob("*")):
        if not run_dir.is_dir() or not (run_dir / "status.json").exists():
            continue
        status = load_json(run_dir / "status.json")
        audit = load_json(run_dir / "sampled_attribution_audit.json")
        interventions = load_json(run_dir / "fixed_target_interventions.json")
        mixing = load_json(run_dir / "coordinate_mixing_audit.json")
        config = load_json(run_dir / "config.json")
        fixed_delta = interventions["fixed_target_prediction_mse_delta"]
        rows.append({
            "root": root.name,
            "run_id": status["run_id"],
            "dataset": status["dataset"],
            "method": status["method"],
            "train_seed": int(status.get("train_seed", config.get("train_seed", 0))),
            "mse": status["fixed_target_prediction_mse"],
            "auroc_total": (
                None
                if audit["total_nominal_metrics"] is None
                else audit["total_nominal_metrics"]["auroc"]
            ),
            "auroc_partial": (
                None
                if audit["partial_nominal_metrics"] is None
                else audit["partial_nominal_metrics"]["auroc"]
            ),
            "missing_route": audit["missing_route_relative_magnitude"],
            "partial_total_r": audit["partial_total_nominal_pearson"],
            "partial_total_jaccard": audit["partial_total_nominal_topk_jaccard"],
            "tail_median": audit["temporal_tail_statistics"]["median"],
            "tail_max": audit["temporal_tail_statistics"]["maximum"],
            "mask_x_delta": fixed_delta.get("mask_x"),
            "mask_c_delta": fixed_delta.get("mask_c"),
            "shuffle_x_delta": fixed_delta.get(
                "shuffle_x_only",
                fixed_delta.get("shuffle_x"),
            ),
            "shuffle_c_delta": fixed_delta.get("shuffle_c_only"),
            "mix_max_source_share_median": (
                None if mixing is None else mixing["max_source_share"]["median"]
            ),
            "mix_entropy_median": (
                None if mixing is None else mixing["normalized_source_entropy"]["median"]
            ),
            "wall_seconds": status["wall_seconds"],
        })
    return rows


def paired_deltas(
    rows: Sequence[Mapping[str, object]],
    *,
    candidate: str,
    reference: str,
    metric: str,
) -> List[Dict[str, object]]:
    index = {
        (
            row["cell"],
            row["method"],
            row["data_seed"],
            row["train_seed"],
        ): row
        for row in rows
    }
    output = []
    cells = sorted({str(row["cell"]) for row in rows})
    for cell in cells:
        effects = []
        for row in rows:
            if row["cell"] != cell or row["method"] != candidate:
                continue
            reference_row = index.get((
                cell,
                reference,
                row["data_seed"],
                row["train_seed"],
            ))
            if reference_row is None:
                continue
            effects.append(
                float(row[metric]) - float(reference_row[metric])
            )
        output.append({
            "cell": cell,
            "candidate": candidate,
            "reference": reference,
            "metric": metric,
            "n": len(effects),
            "mean_delta": None if not effects else float(np.mean(effects)),
            "positive_count": int(np.sum(np.asarray(effects) > 0)),
            "effects": effects,
        })
    return output


def determinism_report(root_a: Path, root_b: Path) -> Dict[str, object]:
    run = "netsim48__concat__ts0__it1000__H64"
    a_dir = root_a / run
    b_dir = root_b / run
    a = np.load(a_dir / "sampled_attribution_objects.npz")
    b = np.load(b_dir / "sampled_attribution_objects.npz")
    arrays = {}
    for key in a.files:
        arrays[key] = {
            "exact": bool(np.array_equal(a[key], b[key])),
            "max_abs_difference": (
                float(np.max(np.abs(a[key] - b[key])))
                if np.issubdtype(a[key].dtype, np.number)
                else None
            ),
        }
    json_files = {}
    for name in (
        "sampled_attribution_audit.json",
        "fixed_target_interventions.json",
        "coordinate_mixing_audit.json",
    ):
        json_files[name] = {
            "exact": load_json(a_dir / name) == load_json(b_dir / name)
        }
    return {
        "same_current_commit_command": True,
        "arrays": arrays,
        "json_files": json_files,
        "passed": all(item["exact"] for item in arrays.values())
        and all(item["exact"] for item in json_files.values()),
    }


def fmt(value, digits: int = 4) -> str:
    return "N/A" if value is None else f"{float(value):.{digits}f}"


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def build_report(summary: Mapping[str, object]) -> str:
    d2_500 = summary["d2_500_means"]
    d2_delta = summary["d2_500_adaptive_deltas"]
    phase8 = summary["phase8_500_means"]
    audit = summary["audit_means"]
    lines = [
        "# Phase 9 RTX 4090 Bounded Enhancement Round 1",
        "",
        "**Status:** development-only; no manuscript or frozen artifact was modified.",
        "",
        "## Executive decision",
        "",
        "- **Filtering repair line: STOP.** Static adaptive FIR showed a large "
        "stationary-nonlinear gain at 2000 iterations but no stable non-stationary "
        "benefit. The contextual gate also failed both non-stationary cells.",
        "- **Gradient-projection repair line: STOP.** Protecting prediction reduced "
        "MSE but degraded graph recovery; protecting only historical coverage was "
        "nearly equivalent to the standard repair.",
        "- **Audit-generalization line: CONTINUE TO FORMAL PLANNING.** Missing-route, "
        "partial-total discrepancy, fixed-target route use, coordinate mixing, and "
        "bounded horizon diagnostics were stable across NetSim, MoCap, Mamba, and "
        "an active causal TCN auxiliary route.",
        "- These are development results, not submission evidence. A frozen "
        "advisor-approved protocol is still required before manuscript use.",
        "",
        "## D2 at 500 iterations",
        "",
    ]
    d2_rows = []
    for row in d2_500:
        d2_rows.append([
            row["cell"],
            row["method"],
            row["n"],
            fmt(row["auroc_mean"]),
            fmt(row["auprc_mean"]),
            fmt(row["mcc_mean"]),
            fmt(row["mse_mean"], 5),
        ])
    lines.append(markdown_table(
        ["Cell", "Method", "n", "AUROC", "AUPRC", "MCC", "raw MSE"],
        d2_rows,
    ))
    lines.extend(["", "Paired AUROC effects for the adaptive candidates:", ""])
    lines.append(markdown_table(
        ["Cell", "Candidate", "Reference", "Mean delta", "Positive seeds"],
        [
            [
                row["cell"],
                row["candidate"],
                row["reference"],
                fmt(row["mean_delta"]),
                f"{row['positive_count']}/{row['n']}",
            ]
            for row in d2_delta
        ],
    ))
    lines.extend(["", "## Phase 8 gradient strategies at 500 iterations", ""])
    lines.append(markdown_table(
        ["Method", "n", "AUROC", "AUPRC", "MCC", "MSE", "Coeff. r"],
        [
            [
                row["method"],
                row["n"],
                fmt(row["auroc_mean"]),
                fmt(row["auprc_mean"]),
                fmt(row["mcc_mean"]),
                fmt(row["mse_mean"], 5),
                fmt(row["coefficient_r_mean"]),
            ]
            for row in phase8
        ],
    ))
    lines.extend(["", "## Cross-domain audit stability", ""])
    lines.append(markdown_table(
        [
            "Architecture",
            "Dataset",
            "n",
            "Missing route",
            "Partial-total r",
            "Tail median",
            "Mix entropy",
            "mask-c delta",
        ],
        [
            [
                row["method"],
                row["dataset"],
                row["n"],
                fmt(row["missing_route_mean"]),
                fmt(row["partial_total_r_mean"]),
                fmt(row["tail_median_mean"]),
                fmt(row["mix_entropy_median_mean"]),
                fmt(row["mask_c_delta_mean"]),
            ]
            for row in audit
            if row["method"] in {"concat", "tcn_concat"}
        ],
    ))
    lines.extend([
        "",
        "## Horizon sensitivity",
        "",
        "Using the same target windows, nominal-lag scores were identical at "
        "H=32/64/128. H=64 captured about 100% of H=128 off-diagonal mass on "
        "NetSim, 99.93% on MoCap-run, and approximately 100% on MoCap-salsa. "
        "This supports H=64 for the bounded audit, but does not establish mass "
        "beyond H=128.",
        "",
        "## Determinism",
        "",
        f"Same-current-commit NetSim48 concat rerun exact match: "
        f"**{summary['determinism']['passed']}**.",
        "",
        "## Evidence boundary",
        "",
        "- NetSim baseline graph recovery is only around random-to-moderate; these "
        "runs support route/score diagnostics, not a performance benchmark claim.",
        "- MoCap has no accepted direct graph ground truth here; only prediction "
        "interventions and attribution semantics are reported.",
        "- No Phase 7 seeds 4-8, Stage 1b outputs, AutoDL GPU, or KBS manuscript "
        "files were accessed or modified.",
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    root = args.results_root
    d2_500 = (
        read_d2_root(root / "gpu_d2_it500_round1")
        + read_d2_root(root / "gpu_d2_contextual_it500")
    )
    d2_2000 = (
        read_d2_root(root / "gpu_d2_it2000_bounded")
        + read_d2_root(root / "gpu_d2_contextual_it2000_bounded")
    )
    phase8 = (
        read_phase8_root(root / "gpu_phase8_it500_round1")
        + read_phase8_root(root / "gpu_phase8_history_guard_it500")
    )
    audit = (
        read_audit_root(root / "audit_current_commit_seed0_baseline")
        + read_audit_root(root / "audit_current_commit_seed0_concat")
        + read_audit_root(root / "audit_cross_domain_it1000_stability")
        + read_audit_root(root / "audit_active_tcn_architecture_seed0")
        + read_audit_root(root / "audit_active_tcn_architecture_stability")
    )
    # Keep only current canonical Mamba seed-0 plus stability seeds 1/2.
    audit = [
        row
        for row in audit
        if not (
            row["root"] == "audit_cross_domain_it1000_stability"
            and row["train_seed"] == 0
        )
    ]
    d2_metrics = ["auroc", "auprc", "mcc", "f1", "mse"]
    audit_metrics = [
        "mse",
        "auroc_total",
        "auroc_partial",
        "missing_route",
        "partial_total_r",
        "partial_total_jaccard",
        "tail_median",
        "tail_max",
        "mask_x_delta",
        "mask_c_delta",
        "shuffle_x_delta",
        "shuffle_c_delta",
        "mix_max_source_share_median",
        "mix_entropy_median",
    ]
    summary = {
        "development_only": True,
        "formal_result": False,
        "d2_500_rows": d2_500,
        "d2_500_means": grouped_means(
            d2_500,
            group_fields=["cell", "method"],
            metric_fields=d2_metrics,
        ),
        "d2_500_adaptive_deltas": (
            paired_deltas(
                d2_500,
                candidate="adaptive_fir",
                reference="baseline",
                metric="auroc",
            )
            + paired_deltas(
                d2_500,
                candidate="contextual_fir",
                reference="baseline",
                metric="auroc",
            )
            + paired_deltas(
                d2_500,
                candidate="adaptive_fir",
                reference="fixed_fir3",
                metric="auroc",
            )
            + paired_deltas(
                d2_500,
                candidate="contextual_fir",
                reference="fixed_fir3",
                metric="auroc",
            )
        ),
        "d2_2000_rows": d2_2000,
        "d2_2000_means": grouped_means(
            d2_2000,
            group_fields=["cell", "method", "root"],
            metric_fields=d2_metrics,
        ),
        "phase8_500_rows": phase8,
        "phase8_500_means": grouped_means(
            phase8,
            group_fields=["method"],
            metric_fields=[
                "auroc",
                "auprc",
                "mcc",
                "f1",
                "mse",
                "coefficient_r",
                "missing_route",
            ],
        ),
        "audit_rows": audit,
        "audit_means": grouped_means(
            audit,
            group_fields=["method", "dataset"],
            metric_fields=audit_metrics,
        ),
        "horizon_sensitivity": load_json(
            root
            / "audit_horizon_sensitivity_seed0"
            / "horizon_sensitivity_summary.json"
        ),
        "determinism": determinism_report(
            root / "audit_determinism_repeat",
            root / "audit_determinism_repeat_b",
        ),
        "decisions": {
            "filtering_method_line": "STOP",
            "gradient_projection_method_line": "STOP",
            "audit_generalization": "CONTINUE_TO_FROZEN_PROTOCOL_PLANNING",
            "autodl_formal_execution": "NOT_AUTHORIZED_BY_THIS_DEVELOPMENT_ROUND",
        },
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_root / "d2_rows.csv", d2_500 + d2_2000)
    write_csv(args.output_root / "phase8_rows.csv", phase8)
    write_csv(args.output_root / "audit_rows.csv", audit)
    atomic_json(args.output_root / "phase9_4090_round1_summary.json", summary)
    atomic_text(
        args.output_root / "PHASE9_4090_ROUND1_REPORT.md",
        build_report(summary),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
