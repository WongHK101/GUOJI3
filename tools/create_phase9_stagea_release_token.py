"""Create an external release token for the clean Phase 9 Stage A commit."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CRITICAL_FILES = (
    "experiments/phase9_audit_stagea.py",
    "experiments/aggregate_phase9_audit_stagea.py",
    "src/phase9_audit_generalization.py",
    "src/phase8_coverage.py",
    "src/mamba_jrngc_pilot.py",
    "src/minimal_mamba.py",
    "src/repaired_istf.py",
    "configs/phase9_audit_stagea_v1.json",
    "paper-data/docs/phase9_audit_validation_v1/PHASE9_AUDIT_RUN_MATRIX.csv",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_text_sha256(path: Path) -> str:
    content = path.read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(content).hexdigest()


def git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def main() -> int:
    args = parse_args()
    if git("status", "--porcelain"):
        raise RuntimeError("Release token requires a clean worktree")
    config = args.config.resolve()
    matrix = args.matrix.resolve()
    files = {
        path: canonical_text_sha256(ROOT / path)
        for path in CRITICAL_FILES
    }
    payload = {
        "protocol_name": "phase9_audit_generality_stagea_v1",
        "authorized_stage": "A_4090_VALIDATION",
        "execution_authorized": True,
        "approved_commit": git("rev-parse", "HEAD"),
        "config_sha256": canonical_text_sha256(config),
        "matrix_sha256": canonical_text_sha256(matrix),
        "critical_files": files,
        "authorization_basis": (
            "user-authorized local self-review and prospective RTX4090 validation"
        ),
        "stage_b_authorized": False,
        "autodl_authorized": False,
        "manuscript_evidence_authorized": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
