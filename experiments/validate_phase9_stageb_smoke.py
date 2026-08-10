"""Validate two non-evidentiary Phase 9 Stage B AutoDL smoke roots."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Mapping

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
for path in (PROJECT_ROOT, SRC_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from knowledge_metrics import topk_edges_exact  # noqa: E402


METHODS = ("baseline", "mamba_concat", "tcn_concat")
MATRIX_RUN_IDS = {
    method: f"b_autodl_confirmation__netsim16__{method}__rep1"
    for method in METHODS
}
TOLERANCE = 1e-6


def release_lock_fingerprint(release_lock: Mapping[str, object]) -> str:
    payload = json.dumps(
        release_lock,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-a", type=Path, required=True)
    parser.add_argument("--root-b", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def checkpoint_max_abs(left: Path, right: Path) -> float:
    left_state = torch.load(left, map_location="cpu")["model_state"]
    right_state = torch.load(right, map_location="cpu")["model_state"]
    if set(left_state) != set(right_state):
        return float("inf")
    return max(
        (
            float(
                torch.max(torch.abs(left_state[name] - right_state[name]))
            )
            for name in left_state
        ),
        default=0.0,
    )


def score_max_abs(left: Path, right: Path) -> float:
    with np.load(left, allow_pickle=False) as a, np.load(
        right, allow_pickle=False
    ) as b:
        if set(a.files) != set(b.files):
            return float("inf")
        return max(
            (
                float(
                    np.max(
                        np.abs(
                            np.asarray(a[name], dtype=np.float64)
                            - np.asarray(b[name], dtype=np.float64)
                        )
                    )
                )
                for name in a.files
            ),
            default=0.0,
        )


def max_abs_nested(left: object, right: object) -> float:
    if left is None and right is None:
        return 0.0
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        if set(left) != set(right):
            return float("inf")
        return max(
            (max_abs_nested(left[key], right[key]) for key in left),
            default=0.0,
        )
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            return float("inf")
        return max(
            (max_abs_nested(a, b) for a, b in zip(left, right)),
            default=0.0,
        )
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return abs(float(left) - float(right))
    return 0.0 if left == right else float("inf")


def inspect_root(root: Path) -> dict:
    summary = load_json(root / "execution_summary.json")
    release_lock = load_json(root / "release_lock.json")
    statuses = summary.get("statuses", [])
    by_method = {status["method"]: status for status in statuses}
    checks = {
        "requested_three": summary.get("requested_records") == 3,
        "completed_three": summary.get("completed_records") == 3,
        "smoke": summary.get("smoke") is True,
        "nonformal": (
            summary.get("formal_result") is False
            and summary.get("confirmation_candidate") is False
        ),
        "method_set": set(by_method) == set(METHODS),
        "all_complete": all(
            status.get("status") == "complete" for status in statuses
        ),
        "all_cuda": all(
            str(status.get("device", "")).startswith("cuda")
            for status in statuses
        ),
        "all_deterministic": all(
            status.get("deterministic_algorithms") is True
            for status in statuses
        ),
        "all_finite": all(
            status.get("no_nan_inf") is True for status in statuses
        ),
        "all_20_iterations": all(
            status.get("effective_iterations") == 20
            for status in statuses
        ),
        "all_nonformal_status": all(
            status.get("formal_result") is False
            and status.get("confirmation_candidate") is False
            and status.get("manuscript_evidence") is False
            for status in statuses
        ),
        "positive_vram": all(
            status.get("cuda_max_memory_allocated_mb") is not None
            and math.isfinite(float(status["cuda_max_memory_allocated_mb"]))
            and float(status["cuda_max_memory_allocated_mb"]) > 0
            for status in statuses
        ),
        "matrix_run_ids": all(
            by_method.get(method, {}).get("matrix_run_id")
            == MATRIX_RUN_IDS[method]
            for method in METHODS
        ),
        "release_lock_stagea_bound": (
            isinstance(
                release_lock.get("stage_a_decision_canonical_sha256"),
                str,
            )
            and len(release_lock["stage_a_decision_canonical_sha256"]) == 64
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "release_lock": release_lock,
        "statuses": by_method,
    }


def run_dir(root: Path, method: str) -> Path:
    return (
        root
        / "runs"
        / f"smoke_it20__{MATRIX_RUN_IDS[method]}"
    )


def compare_roots(root_a: Path, root_b: Path) -> dict:
    comparisons = []
    for method in METHODS:
        left = run_dir(root_a, method)
        right = run_dir(root_b, method)
        audit_left = load_json(left / "sampled_attribution_audit.json")
        audit_right = load_json(right / "sampled_attribution_audit.json")
        checkpoint_diff = checkpoint_max_abs(
            left / "checkpoint.pt",
            right / "checkpoint.pt",
        )
        score_diff = score_max_abs(
            left / "sampled_attribution_objects.npz",
            right / "sampled_attribution_objects.npz",
        )
        metric_diff = max_abs_nested(audit_left, audit_right)
        edge_count = int(
            (audit_left.get("total_nominal_metrics") or {}).get(
                "n_true_edges", 0
            )
        )
        with np.load(
            left / "sampled_attribution_objects.npz",
            allow_pickle=False,
        ) as a, np.load(
            right / "sampled_attribution_objects.npz",
            allow_pickle=False,
        ) as b:
            score_a = np.asarray(a["s_total_nominal"])
            score_b = np.asarray(b["s_total_nominal"])
        topk_equal = (
            True
            if edge_count < 1
            else topk_edges_exact(
                score_a, k=edge_count, exclude_diag=True
            )
            == topk_edges_exact(
                score_b, k=edge_count, exclude_diag=True
            )
        )
        passed = (
            checkpoint_diff <= TOLERANCE
            and score_diff <= TOLERANCE
            and metric_diff <= TOLERANCE
            and topk_equal
        )
        comparisons.append(
            {
                "method": method,
                "checkpoint_max_abs": checkpoint_diff,
                "score_max_abs": score_diff,
                "metric_max_abs": metric_diff,
                "topk_equal": topk_equal,
                "passed": passed,
            }
        )
    return {
        "passed": all(item["passed"] for item in comparisons),
        "comparisons": comparisons,
    }


def main() -> int:
    args = parse_args()
    root_a = args.root_a.resolve()
    root_b = args.root_b.resolve()
    first = inspect_root(root_a)
    second = inspect_root(root_b)
    same_release = first["release_lock"] == second["release_lock"]
    comparison = compare_roots(root_a, root_b)
    payload = {
        "passed": (
            first["passed"]
            and second["passed"]
            and same_release
            and comparison["passed"]
        ),
        "root_a": {
            "path": str(root_a),
            "passed": first["passed"],
            "checks": first["checks"],
        },
        "root_b": {
            "path": str(root_b),
            "passed": second["passed"],
            "checks": second["checks"],
        },
        "same_release_lock": same_release,
        "release_lock_fingerprint": release_lock_fingerprint(
            first["release_lock"]
        ),
        "approved_commit": first["release_lock"].get("approved_commit"),
        "release_token_sha256": first["release_lock"].get(
            "release_token_sha256"
        ),
        "stage_a_decision_canonical_sha256": first["release_lock"].get(
            "stage_a_decision_canonical_sha256"
        ),
        "determinism": comparison,
        "formal_scientific_evidence": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0 if payload["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
