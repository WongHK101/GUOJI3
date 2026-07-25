"""Create the external release token for Phase 9 Stage B confirmation."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PARENT_MATRIX = (
    ROOT
    / "paper-data"
    / "docs"
    / "phase9_audit_validation_v1"
    / "PHASE9_AUDIT_RUN_MATRIX.csv"
)
CRITICAL_FILES = (
    "experiments/phase9_audit_stageb.py",
    "experiments/aggregate_phase9_audit_stageb.py",
    "experiments/validate_phase9_stageb_smoke.py",
    "src/phase9_audit_generalization.py",
    "src/phase8_coverage.py",
    "src/mamba_jrngc_pilot.py",
    "src/minimal_mamba.py",
    "src/repaired_istf.py",
    "configs/phase9_audit_stageb_v1.json",
    "paper-data/docs/phase9_audit_validation_v1/"
    "PHASE9_AUDIT_STAGEB_AUTHORIZED_MATRIX.csv",
    "paper-data/docs/phase9_audit_validation_v1/"
    "PHASE9_AUDIT_RUN_MATRIX.csv",
    "paper-data/docs/phase9_audit_validation_v1/"
    "STAGEA_GATE_DECISION_LOCK.json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--stage-a-decision", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
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
    stage_a_decision_path = args.stage_a_decision.resolve()
    stage_a_decision = json.loads(
        stage_a_decision_path.read_text(encoding="utf-8-sig")
    )
    if (
        stage_a_decision.get("decision") != "UNLOCK_STAGE_B"
        or stage_a_decision.get("unlock_stage_b") is not True
        or stage_a_decision.get("semantic_integrity_passed") is not True
        or stage_a_decision.get("determinism", {}).get("passed") is not True
    ):
        raise RuntimeError("Stage A decision does not authorize Stage B")
    config_payload = json.loads(config.read_text(encoding="utf-8-sig"))
    decision_sha = canonical_text_sha256(stage_a_decision_path)
    if (
        config_payload.get("required_stage_a_decision_canonical_sha256")
        != decision_sha
    ):
        raise RuntimeError("Config is not bound to the supplied Stage A decision")
    parent_matrix_sha = canonical_text_sha256(PARENT_MATRIX)
    if (
        config_payload.get("parent_preregistered_matrix_sha256")
        != parent_matrix_sha
    ):
        raise RuntimeError("Parent preregistered matrix SHA256 mismatch")

    files = {
        path: canonical_text_sha256(ROOT / path)
        for path in CRITICAL_FILES
    }
    payload = {
        "protocol_name": "phase9_audit_generality_stageb_v1",
        "authorized_stage": "B_AUTODL_CONFIRMATION",
        "execution_authorized": True,
        "approved_commit": git("rev-parse", "HEAD"),
        "config_sha256": canonical_text_sha256(config),
        "matrix_sha256": canonical_text_sha256(matrix),
        "parent_preregistered_matrix_sha256": parent_matrix_sha,
        "stage_a_decision_canonical_sha256": decision_sha,
        "critical_files": files,
        "authorization_basis": (
            "user-authorized autonomous progression after independently "
            "verified Stage A UNLOCK_STAGE_B"
        ),
        "stage_b_authorized": True,
        "autodl_authorized": True,
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
