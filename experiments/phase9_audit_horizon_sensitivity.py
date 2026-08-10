"""Read-only horizon sensitivity for Phase 9 cross-domain audit checkpoints."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import fields
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _path in (PROJECT_ROOT, PROJECT_ROOT / "src", PROJECT_ROOT / "experiments"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from phase8_coverage import Phase8ModelConfig, exact_topk_jaccard, make_legacy_concat  # noqa: E402
from phase9_audit_generalization import (  # noqa: E402
    deterministic_audit_targets,
    sampled_raw_chain_audit,
)
from phase9_run_audit_generalization import DATASET_SPECS, load_dataset  # noqa: E402


def parse_list(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_ints(value: str) -> List[int]:
    return [int(item) for item in parse_list(value)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--datasets", default=",".join(DATASET_SPECS))
    parser.add_argument("--train-seed", type=int, default=0)
    parser.add_argument("--horizons", default="32,64,128")
    parser.add_argument("--window-count", type=int, default=32)
    parser.add_argument("--device", default="cuda")
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
    raise TypeError(type(value).__name__)


def offdiag_lag_mass(j_bar: np.ndarray) -> np.ndarray:
    mask = ~np.eye(j_bar.shape[0], dtype=bool)
    return np.sum(j_bar[mask, :], axis=0)


def vector_corr(a: np.ndarray, b: np.ndarray) -> float | None:
    mask = ~np.eye(a.shape[0], dtype=bool)
    av = a[mask]
    bv = b[mask]
    if np.std(av) <= 1e-12 or np.std(bv) <= 1e-12:
        return None
    return float(np.corrcoef(av, bv)[0, 1])


def load_adapter(
    *,
    checkpoint_dir: Path,
    device: torch.device,
):
    config = json.loads((checkpoint_dir / "config.json").read_text(encoding="utf-8"))
    allowed = {field.name for field in fields(Phase8ModelConfig)}
    model_values = {
        key: value
        for key, value in config["model"].items()
        if key in allowed
    }
    cfg = Phase8ModelConfig(**model_values)
    adapter = make_legacy_concat(cfg)
    checkpoint = torch.load(checkpoint_dir / "checkpoint.pt", map_location="cpu")
    adapter.model.load_state_dict(checkpoint["model_state"])
    adapter.model.to(device).eval()
    return adapter, config


def main() -> int:
    args = parse_args()
    device = torch.device(args.device)
    horizons = sorted(set(parse_ints(args.horizons)))
    if not horizons or horizons[0] < 1:
        raise ValueError("At least one positive horizon is required")
    max_horizon = horizons[-1]
    summaries: Dict[str, object] = {}
    args.output_root.mkdir(parents=True, exist_ok=True)
    for dataset in parse_list(args.datasets):
        spec = DATASET_SPECS[dataset]
        checkpoint_dir = (
            args.checkpoint_root
            / f"{dataset}__concat__ts{args.train_seed}__it1000__H64"
        )
        adapter, checkpoint_config = load_adapter(
            checkpoint_dir=checkpoint_dir,
            device=device,
        )
        x, graph, metadata = load_dataset(dataset, data_root=args.data_root)
        targets = deterministic_audit_targets(
            T=x.shape[1],
            lag=adapter.lag,
            attribution_horizon=max_horizon,
            count=args.window_count,
            seed=int(spec["perturbation_seed"]) + 202,
        )
        audits = {}
        arrays = {}
        for horizon in horizons:
            audit = sampled_raw_chain_audit(
                adapter,
                x,
                target_indices=targets,
                attribution_horizon=horizon,
                graph=graph,
            )
            audits[horizon] = audit
            arrays[f"j_bar_total_H{horizon}"] = audit.j_bar_total
            arrays[f"s_total_nominal_H{horizon}"] = audit.s_total_nominal
        reference = audits[max_horizon]
        lag_mass = offdiag_lag_mass(reference.j_bar_total)
        total_mass = float(np.sum(lag_mass))
        cumulative = {}
        comparisons = {}
        edge_count = None
        if graph is not None:
            graph_2d = np.asarray(graph) != 0
            graph_2d = graph_2d.copy()
            np.fill_diagonal(graph_2d, 0)
            edge_count = int(np.sum(graph_2d))
        for horizon in horizons:
            audit = audits[horizon]
            cumulative[horizon] = float(
                np.sum(lag_mass[:horizon]) / max(total_mass, 1e-12)
            )
            comparisons[horizon] = {
                "nominal_score_max_abs_difference_vs_max_horizon": float(
                    np.max(np.abs(
                        audit.s_total_nominal - reference.s_total_nominal
                    ))
                ),
                "nominal_score_pearson_vs_max_horizon": vector_corr(
                    audit.s_total_nominal,
                    reference.s_total_nominal,
                ),
                "nominal_score_topk_jaccard_vs_max_horizon": (
                    None
                    if edge_count is None
                    else exact_topk_jaccard(
                        audit.s_total_nominal,
                        reference.s_total_nominal,
                        edge_count,
                    )
                ),
                "missing_route_relative_magnitude": (
                    audit.missing_route_relative_magnitude
                ),
                "temporal_tail_statistics": audit.temporal_tail_statistics,
            }
        shell_mass = {}
        previous = 0
        for horizon in horizons:
            shell_mass[f"h{previous + 1}_to_h{horizon}"] = float(
                np.sum(lag_mass[previous:horizon]) / max(total_mass, 1e-12)
            )
            previous = horizon
        dataset_summary = {
            "dataset": dataset,
            "checkpoint_dir": str(checkpoint_dir),
            "checkpoint_config": checkpoint_config,
            "source_sha256": metadata["source_sha256"],
            "target_indices": targets.tolist(),
            "horizons": horizons,
            "max_horizon": max_horizon,
            "horizon_status": "bounded sensitivity; full prefix not claimed",
            "comparisons": comparisons,
            "offdiagonal_attribution_mass_cumulative_fraction": cumulative,
            "offdiagonal_attribution_mass_shell_fraction": shell_mass,
            "mass_beyond_max_horizon_assessed": False,
            "development_only": True,
            "formal_result": False,
        }
        summaries[dataset] = dataset_summary
        atomic_json(args.output_root / dataset / "horizon_sensitivity.json", dataset_summary)
        atomic_npz(args.output_root / dataset / "horizon_objects.npz", **arrays)
    atomic_json(args.output_root / "horizon_sensitivity_summary.json", {
        "development_only": True,
        "formal_result": False,
        "datasets": summaries,
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
