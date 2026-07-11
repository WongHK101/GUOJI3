"""Execute a release-locked final Phase 8 stage with a durable ledger."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
for path in (PROJECT_ROOT, SRC_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from phase8_final_protocol import validate_final_authorization, validate_final_run_matrix  # noqa: E402
from phase8_protocol import file_sha256, load_json, load_run_matrix, verify_release_lock  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-matrix", type=Path, required=True)
    parser.add_argument("--release-lock-dir", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--confirmation-token", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    temporary.replace(path)


def initialize_root(args: argparse.Namespace, release: dict, stage: str) -> None:
    args.output_root.mkdir(parents=True, exist_ok=True)
    lock = args.output_root / "execution_lock"
    lock.mkdir(parents=True, exist_ok=True)
    sources = {
        "config_snapshot.json": args.config,
        "run_matrix_snapshot.csv": args.run_matrix,
        "authorization_snapshot.json": args.authorization,
        "release_source_manifest.json": args.release_lock_dir / "release_source_manifest.json",
        "approved_phase8_code_commit.txt": args.release_lock_dir / "approved_phase8_code_commit.txt",
    }
    if args.confirmation_token is not None:
        sources["confirmation_release_token.json"] = args.confirmation_token
    for name, source in sources.items():
        if not source.is_file():
            raise FileNotFoundError(f"Execution-lock source is missing: {source}")
        destination = lock / name
        if destination.is_file() and file_sha256(destination) != file_sha256(source):
            raise RuntimeError(f"Execution-lock snapshot changed: {name}")
        if not destination.exists():
            shutil.copy2(source, destination)
    atomic_json(lock / "release_lock_verification.json", {"stage": stage, **release})


def main() -> int:
    args = parse_args()
    matrix_report = validate_final_run_matrix(args.config, args.run_matrix)
    if not matrix_report["passed"]:
        raise RuntimeError(f"Final Phase 8 matrix validation failed: {matrix_report['failures']}")
    config = load_json(args.config)
    stage = str(config["execution_stage"])
    expected = 18 if stage == "lambda_tradeoff" else 40
    rows = load_run_matrix(args.run_matrix)
    if len(rows) != expected:
        raise RuntimeError(f"Stage {stage} expected {expected} records, found {len(rows)}")
    release = verify_release_lock(PROJECT_ROOT, args.release_lock_dir, require_clean=True)
    release_commit = str(release["actual_commit"] or release["approved_commit"])
    first = matrix_report["resolved_records"][0]
    validate_final_authorization(
        args.authorization,
        release_commit=release_commit,
        config_sha256=file_sha256(args.config),
        matrix_sha256=file_sha256(args.run_matrix),
        phase=str(first["phase"]),
        block=str(first["block"]),
        confirmation_token_path=args.confirmation_token,
    )
    initialize_root(args, release, stage)

    ledger_path = args.output_root / "execution_ledger.json"
    ledger = load_json(ledger_path) if ledger_path.is_file() else {
        "stage": stage,
        "release": release,
        "config_sha256": file_sha256(args.config),
        "run_matrix_sha256": file_sha256(args.run_matrix),
        "expected_record_ids": [row["record_id"] for row in rows],
        "records": {},
        "started_at_unix": time.time(),
    }
    runner = PROJECT_ROOT / "experiments" / "phase8_gpu_runner.py"
    logs = args.output_root / "orchestration_logs"
    logs.mkdir(parents=True, exist_ok=True)
    for row in rows:
        record_id = row["record_id"]
        status_path = args.output_root / "runs" / record_id / "status.json"
        if args.resume and status_path.is_file() and load_json(status_path).get("status") == "complete":
            ledger["records"][record_id] = {"status": "complete_existing", "checked_at_unix": time.time()}
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
        if args.confirmation_token is not None:
            command.extend(["--confirmation-token", str(args.confirmation_token)])
        if args.resume:
            command.append("--resume")
        started = time.time()
        ledger["records"][record_id] = {"status": "running", "started_at_unix": started}
        atomic_json(ledger_path, ledger)
        with (logs / f"{record_id}.stdout.log").open("w", encoding="utf-8") as stdout, (
            logs / f"{record_id}.stderr.log"
        ).open("w", encoding="utf-8") as stderr:
            result = subprocess.run(command, cwd=PROJECT_ROOT, stdout=stdout, stderr=stderr, check=False)
        ledger["records"][record_id] = {
            "status": "complete" if result.returncode == 0 else "failed",
            "returncode": result.returncode,
            "started_at_unix": started,
            "completed_at_unix": time.time(),
        }
        atomic_json(ledger_path, ledger)
        if result.returncode != 0:
            ledger["status"] = "failed"
            atomic_json(ledger_path, ledger)
            return result.returncode
    ledger["status"] = "complete"
    ledger["completed_record_count"] = len(rows)
    ledger["completed_at_unix"] = time.time()
    atomic_json(ledger_path, ledger)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
