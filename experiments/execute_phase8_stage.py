"""Execute one authorized Phase 8 stage with a durable record ledger."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
for path in [PROJECT_ROOT, SRC_DIR]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from phase8_protocol import (  # noqa: E402
    file_sha256,
    load_json,
    load_run_matrix,
    validate_run_matrix,
    verify_release_lock,
)


STAGE_BLOCKS = {
    "preflight": {"infrastructure_smoke", "repair_scale_benchmark"},
    "replication": {"capacity_replication", "fixed_target_interventions", "coefficient_replication"},
    "pilot": {"repair_pilot"},
}
EXPECTED_COUNTS = {"preflight": 5, "replication": 50, "pilot": 30}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=sorted(STAGE_BLOCKS), required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-matrix", type=Path, required=True)
    parser.add_argument("--release-lock-dir", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--independent-replication-report", type=Path)
    return parser.parse_args()


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    temporary.replace(path)


def select_rows(stage: str, matrix_path: Path) -> List[Dict[str, str]]:
    blocks = STAGE_BLOCKS[stage]
    rows = [row for row in load_run_matrix(matrix_path) if row["block"] in blocks]
    if len(rows) != EXPECTED_COUNTS[stage]:
        raise RuntimeError(f"Stage {stage} expected {EXPECTED_COUNTS[stage]} records, got {len(rows)}")
    if any(row["phase"] == "gated_confirmation" for row in rows):
        raise PermissionError("Confirmation records are sealed")
    return rows


def prerequisite_check(stage: str, root: Path, independent_replication_report: Path | None = None) -> None:
    if stage in {"replication", "pilot"}:
        preflight_path = root / "gpu_preflight_validation.json"
        if not preflight_path.is_file() or load_json(preflight_path).get("passed") is not True:
            raise PermissionError("Formal execution requires a passing GPU preflight report")
    if stage == "pilot":
        replication_path = independent_replication_report or (root / "replication_aggregate_and_gates.json")
        if not replication_path.is_file() or load_json(replication_path).get("passed_execution_completeness") is not True:
            raise PermissionError("Pilot execution requires complete replication aggregation")


def initialize_root(args: argparse.Namespace, release: Dict[str, object]) -> None:
    args.output_root.mkdir(parents=True, exist_ok=True)
    snapshots = args.output_root / "execution_lock"
    snapshots.mkdir(parents=True, exist_ok=True)
    targets = {
        "config_snapshot.json": args.config,
        "run_matrix_snapshot.csv": args.run_matrix,
        "authorization_snapshot.json": args.authorization,
        "release_source_manifest.json": args.release_lock_dir / "release_source_manifest.json",
        "approved_phase8_code_commit.txt": args.release_lock_dir / "approved_phase8_code_commit.txt",
    }
    for name, source in targets.items():
        target = snapshots / name
        if target.is_file() and file_sha256(target) != file_sha256(source):
            raise RuntimeError(f"Execution-root snapshot changed: {name}")
        if not target.exists():
            shutil.copy2(source, target)
    atomic_json(snapshots / "release_lock_verification.json", release)


def main() -> int:
    args = parse_args()
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("CUBLAS_WORKSPACE_CONFIG must be :4096:8")
    matrix_report = validate_run_matrix(args.config, args.run_matrix)
    if not matrix_report["passed"]:
        raise RuntimeError(f"Frozen matrix/config validation failed: {matrix_report['failures']}")
    release = verify_release_lock(PROJECT_ROOT, args.release_lock_dir, require_clean=True)
    authorization = load_json(args.authorization)
    if authorization.get("authorization") != "GPT_APPROVED_PHASE8_REPLICATION_AND_REPAIR_RECOVERY":
        raise PermissionError("Wrong Phase 8 execution authorization")
    if authorization.get("release_commit") != (release["actual_commit"] or release["approved_commit"]):
        raise PermissionError("Authorization release commit mismatch")
    prerequisite_check(args.stage, args.output_root, args.independent_replication_report)
    initialize_root(args, release)
    if args.stage == "pilot" and args.independent_replication_report is not None:
        snapshot = args.output_root / "execution_lock" / "independent_track_a_replication_report.json"
        if snapshot.is_file() and file_sha256(snapshot) != file_sha256(args.independent_replication_report):
            raise RuntimeError("Independent Track A replication report snapshot changed")
        if not snapshot.exists():
            shutil.copy2(args.independent_replication_report, snapshot)
    rows = select_rows(args.stage, args.run_matrix)
    ledger_path = args.output_root / "execution_ledger.json"
    ledger = load_json(ledger_path) if ledger_path.is_file() else {
        "release": release,
        "config_sha256": file_sha256(args.config),
        "run_matrix_sha256": file_sha256(args.run_matrix),
        "confirmation_records_executed": 0,
        "stages": {},
    }
    stage_ledger = ledger["stages"].setdefault(args.stage, {
        "expected_record_ids": [row["record_id"] for row in rows],
        "records": {},
        "started_at_unix": time.time(),
    })
    runner = PROJECT_ROOT / "experiments" / "phase8_gpu_runner.py"
    logs = args.output_root / "orchestration_logs"
    logs.mkdir(parents=True, exist_ok=True)

    for row in rows:
        record_id = row["record_id"]
        status_path = args.output_root / "runs" / record_id / "status.json"
        if args.resume and status_path.is_file() and load_json(status_path).get("status") == "complete":
            stage_ledger["records"][record_id] = {"status": "complete_existing", "checked_at_unix": time.time()}
            atomic_json(ledger_path, ledger)
            continue
        command = [
            sys.executable,
            str(runner),
            "--config", str(args.config),
            "--run-matrix", str(args.run_matrix),
            "--release-lock-dir", str(args.release_lock_dir),
            "--authorization", str(args.authorization),
            "--output-root", str(args.output_root),
            "--record-id", record_id,
            "--device", "cuda",
        ]
        if args.resume:
            command.append("--resume")
        started = time.time()
        stage_ledger["records"][record_id] = {"status": "running", "started_at_unix": started}
        atomic_json(ledger_path, ledger)
        with (logs / f"{record_id}.stdout.log").open("w", encoding="utf-8") as stdout, (
            logs / f"{record_id}.stderr.log"
        ).open("w", encoding="utf-8") as stderr:
            result = subprocess.run(command, cwd=PROJECT_ROOT, stdout=stdout, stderr=stderr, check=False)
        stage_ledger["records"][record_id] = {
            "status": "complete" if result.returncode == 0 else "failed",
            "returncode": result.returncode,
            "started_at_unix": started,
            "completed_at_unix": time.time(),
        }
        atomic_json(ledger_path, ledger)
        if result.returncode != 0:
            stage_ledger["status"] = "failed"
            atomic_json(ledger_path, ledger)
            return result.returncode

    stage_ledger["status"] = "complete"
    stage_ledger["completed_at_unix"] = time.time()
    stage_ledger["completed_record_count"] = len(rows)
    atomic_json(ledger_path, ledger)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
