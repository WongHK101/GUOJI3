"""Release-lock helpers for frozen Stage 1a execution.

The approved code commit is intentionally not tracked in git: a tracked file
cannot contain the hash of the commit that contains it. The release package
ships `configs/approved_stage1a_code_commit.txt` as an artifact, and this file
is ignored by git for clean-worktree checks.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Dict, List


PROJECT_ROOT = Path(__file__).resolve().parents[1]
APPROVED_CODE_COMMIT_PATH = PROJECT_ROOT / "configs" / "approved_stage1a_code_commit.txt"
RELEASE_SOURCE_MANIFEST_PATH = PROJECT_ROOT / "configs" / "release_source_manifest.json"


KEY_SOURCE_FILES = [
    ".gitattributes",
    ".gitignore",
    "experiments/stage1a_gpu_benchmark.py",
    "experiments/aggregate_stage1a.py",
    "experiments/compare_stage1a_determinism.py",
    "experiments/validate_gpu_infrastructure_smoke.py",
    "experiments/release_lock.py",
    "src/repaired_istf.py",
    "src/factorial_data.py",
    "src/minimal_mamba.py",
    "src/knowledge_metrics.py",
    "configs/stage1a_frozen_config.json",
    "configs/stage1a_smoke_config.json",
    "configs/stage1a_gpu_infrastructure_smoke_config.json",
    "configs/approved_stage1a_frozen_config_sha256.txt",
    "configs/approved_stage1a_smoke_config_sha256.txt",
    "configs/approved_stage1a_gpu_infrastructure_smoke_config_sha256.txt",
]


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_available() -> bool:
    return (PROJECT_ROOT / ".git").exists()


def git_commit_hash() -> str:
    proc = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        timeout=30,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or "git rev-parse HEAD failed")
    return proc.stdout.strip()


def git_status_porcelain() -> str:
    proc = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        timeout=30,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or "git status --porcelain failed")
    return proc.stdout


def read_approved_code_commit() -> str:
    if not APPROVED_CODE_COMMIT_PATH.exists():
        raise RuntimeError(f"Approved code commit file is missing: {APPROVED_CODE_COMMIT_PATH}")
    value = APPROVED_CODE_COMMIT_PATH.read_text(encoding="utf-8").strip().split()[0]
    if not value:
        raise RuntimeError(f"Approved code commit file is empty: {APPROVED_CODE_COMMIT_PATH}")
    return value


def load_release_source_manifest() -> Dict[str, object]:
    if not RELEASE_SOURCE_MANIFEST_PATH.exists():
        raise RuntimeError(f"Release source manifest is missing: {RELEASE_SOURCE_MANIFEST_PATH}")
    with open(RELEASE_SOURCE_MANIFEST_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def current_key_file_sha256() -> Dict[str, str]:
    out: Dict[str, str] = {}
    for rel in KEY_SOURCE_FILES:
        path = PROJECT_ROOT / rel
        if not path.exists():
            raise RuntimeError(f"Release key file is missing: {path}")
        out[rel] = file_sha256(path)
    return out


def release_source_manifest_sha256() -> str:
    return file_sha256(RELEASE_SOURCE_MANIFEST_PATH)


def verify_source_manifest() -> Dict[str, object]:
    manifest = load_release_source_manifest()
    expected = manifest.get("files", {})
    actual = current_key_file_sha256()
    failures: List[Dict[str, object]] = []
    for rel, sha in expected.items():
        if actual.get(rel) != sha:
            failures.append({"file": rel, "expected": sha, "actual": actual.get(rel)})
    extra = sorted(set(actual) - set(expected))
    missing = sorted(set(expected) - set(actual))
    if extra:
        failures.append({"type": "extra_key_files_not_in_manifest", "files": extra})
    if missing:
        failures.append({"type": "manifest_files_missing_from_key_list", "files": missing})
    if failures:
        raise RuntimeError(f"Release source manifest mismatch: {failures}")
    return {
        "source_manifest_sha256": release_source_manifest_sha256(),
        "key_file_sha256": actual,
        "manifest_generated_for_commit": manifest.get("release_commit"),
    }


def verify_release_lock(require_clean_worktree: bool = True) -> Dict[str, object]:
    manifest_payload = verify_source_manifest()
    approved_commit = None
    actual_commit = None
    clean_worktree = None
    mode = "source_manifest_only"
    if git_available():
        mode = "git_commit_and_source_manifest"
        approved_commit = read_approved_code_commit()
        actual_commit = git_commit_hash()
        if actual_commit != approved_commit:
            raise RuntimeError(f"Git HEAD {actual_commit} does not match approved commit {approved_commit}")
        status = git_status_porcelain()
        clean_worktree = status == ""
        if require_clean_worktree and not clean_worktree:
            raise RuntimeError(f"Git worktree is not clean:\n{status}")
    return {
        "release_lock_mode": mode,
        "approved_commit": approved_commit,
        "actual_commit": actual_commit,
        "clean_worktree": clean_worktree if clean_worktree is not None else True,
        **manifest_payload,
    }
