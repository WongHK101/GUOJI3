"""Build the external release-lock manifest for the Lorenz confirmation."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CRITICAL_FILES = (
    "configs/phase9_lorenz_901_confirmation_v1.json",
    "configs/phase9_lorenz_901_confirmation_run_matrix_v1.csv",
    "experiments/phase9_lorenz_901_confirmation.py",
    "experiments/aggregate_phase9_lorenz_901_confirmation.py",
    "experiments/phase9_lorenz_strong_operating_preflight.py",
    "experiments/phase9_audit_stageb.py",
    "experiments/mamba_jrngc_pilot.py",
    "src/lorenz96_frozen.py",
    "src/phase8_coverage.py",
    "src/phase9_audit_generalization.py",
    "src/minimal_mamba.py",
    "src/knowledge_metrics.py",
    "src/repaired_istf.py",
    "tests/test_phase9_lorenz_901_confirmation.py",
    "paper-data/docs/phase9_lorenz_preflight/FORMAL_901_PROTOCOL.md",
    "scripts/run_phase9_lorenz_901_confirmation.sh",
    "tools/build_phase9_lorenz_release_manifest.py",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha256(path: Path) -> str:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def git_output(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def main() -> int:
    args = parse_args()
    status = git_output("status", "--porcelain")
    if status:
        raise RuntimeError("Release manifest requires a clean worktree")
    commit = git_output("rev-parse", "HEAD")
    critical = {}
    for relative in CRITICAL_FILES:
        path = PROJECT_ROOT / relative
        if not path.is_file():
            raise RuntimeError(f"Missing critical file: {relative}")
        critical[relative] = file_sha256(path)
    source_lines = [
        f"{critical[relative]}  {relative}"
        for relative in sorted(critical)
    ]
    source_manifest_sha256 = hashlib.sha256(
        ("\n".join(source_lines) + "\n").encode("utf-8")
    ).hexdigest()
    config_path = (
        PROJECT_ROOT / "configs" / "phase9_lorenz_901_confirmation_v1.json"
    )
    matrix_path = (
        PROJECT_ROOT
        / "configs"
        / "phase9_lorenz_901_confirmation_run_matrix_v1.csv"
    )
    payload = {
        "protocol_name": "phase9_lorenz_901_confirmation_v1",
        "approved_commit": commit,
        "config_sha256": file_sha256(config_path),
        "canonical_config_sha256": canonical_json_sha256(config_path),
        "run_matrix_sha256": file_sha256(matrix_path),
        "critical_files": critical,
        "source_manifest_lines": source_lines,
        "source_manifest_sha256": source_manifest_sha256,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
