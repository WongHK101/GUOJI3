"""Assemble the Route-B v2.4 submission-readiness review package.

The builder copies frozen evidence and already-generated manuscript assets. It
does not train, evaluate, or import model code. The staging directory must not
exist, which prevents accidental replacement of prior review artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import zipfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
ELS_ROOT = WORKSPACE_ROOT / "elsarticle"
PACKAGE_ROOT = WORKSPACE_ROOT / "kbs_review_packages"
V23_STAGE = PACKAGE_ROOT / "_phase7_kbs_jacobian_coverage_full_draft_v2_3_stage_20260710_1140"
MANUSCRIPT_STEM = "istf_kbs_jacobian_coverage_submission_candidate_v2_4"
EXPECTED_CANONICAL_SHA = "e47ecd460a8f62bb85c4f79e745c8ddb769e55f34586d6f593d4cc9247cccda2"

DOCS = [
    "BIBLIOGRAPHY_AUDIT_V2_4.md",
    "CANONICAL_MANUSCRIPT_GUARD_V2_4.md",
    "CONTROLLED_CONCAT_ARCHITECTURE_V2_4.md",
    "DIAGNOSTIC_REPLICATION_AUDIT_V2_4.md",
    "SUBMISSION_V2_4_COMPILE_REPORT.md",
    "SUBMISSION_V2_4_EVIDENCE_EXTRACTS_INDEX.md",
    "SUBMISSION_V2_4_EVIDENCE_TRACEABILITY.md",
    "SUBMISSION_V2_4_PACKAGE_README.md",
    "SUBMISSION_V2_4_REPRODUCE.md",
    "SUBMISSION_V2_4_VALIDATION.md",
    "SUBMISSION_V2_4_VISUAL_INSPECTION_REPORT.md",
    "V2.4_BIBLIOGRAPHY_AND_CITATION_CHANGELOG.md",
    "V2.4_CRITICAL_SCIENTIFIC_CORRECTIONS.md",
    "V2.4_JOURNAL_STRATEGY.md",
    "V2.4_REMAINING_OPTIONAL_WORK.md",
    "V2.4_SUBMISSION_EDITORIAL_CHANGELOG.md",
    "V2.4_UNRESOLVED_EVIDENCE_PROVENANCE.md",
    "kbs_jacobian_coverage_claim_evidence_matrix_v2_4_2026-07-10.md",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def copy_file(source: Path, target: Path, inventory: list[dict]) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    package_relative = target.resolve().relative_to(PACKAGE_ROOT.resolve())
    package_path = Path(*package_relative.parts[1:]).as_posix()
    inventory.append(
        {
            "package_path": package_path,
            "original_path": str(source),
            "sha256": sha256(target),
            "size_bytes": target.stat().st_size,
        }
    )


def git_commit(repo: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staging-dir", type=Path, required=True)
    parser.add_argument("--zip-path", type=Path, required=True)
    parser.add_argument("--clean-log", type=Path)
    args = parser.parse_args()

    staging = args.staging_dir.resolve()
    package_root = PACKAGE_ROOT.resolve()
    if package_root not in staging.parents:
        raise RuntimeError(f"Staging must be under {package_root}: {staging}")
    if staging.exists():
        raise FileExistsError(f"Refusing to replace existing staging directory: {staging}")
    if args.zip_path.exists():
        raise FileExistsError(f"Refusing to replace existing ZIP: {args.zip_path}")
    staging.mkdir(parents=True)

    canonical_sha = sha256(ELS_ROOT / "istf_kbs.tex")
    if canonical_sha != EXPECTED_CANONICAL_SHA:
        raise RuntimeError(f"Canonical manuscript changed: {canonical_sha}")

    inventory: list[dict] = []
    for ext in ("tex", "pdf", "log"):
        copy_file(
            ELS_ROOT / f"{MANUSCRIPT_STEM}.{ext}",
            staging / f"{MANUSCRIPT_STEM}.{ext}",
            inventory,
        )
    for name in ("elsarticle.cls", "elsarticle-num-names.bst"):
        copy_file(ELS_ROOT / name, staging / name, inventory)

    figure_dir = "coverage_audit_submission_v2_4"
    for source in sorted((ELS_ROOT / "figures" / figure_dir).glob("*")):
        if source.is_file():
            copy_file(source, staging / "figures" / figure_dir / source.name, inventory)
    table_dir = "coverage_audit_submission_v2_4"
    for source in sorted((ELS_ROOT / "tables" / table_dir).glob("*")):
        if source.is_file():
            copy_file(source, staging / "tables" / table_dir / source.name, inventory)

    docs_root = PROJECT_ROOT / "paper-data" / "docs"
    for name in DOCS:
        copy_file(docs_root / name, staging / name, inventory)

    tools = {
        PROJECT_ROOT / "tools" / "generate_coverage_audit_submission_v2_4_figures.py": staging / "tools" / "generate_coverage_audit_submission_v2_4_figures.py",
        PROJECT_ROOT / "tools" / "export_full_aux_penalty_submission_v2_4.py": staging / "tools" / "export_full_aux_penalty_submission_v2_4.py",
        PROJECT_ROOT / "src" / "mamba_jrngc_pilot.py": staging / "tools" / "frozen_provenance" / "mamba_jrngc_pilot.py",
        PROJECT_ROOT / "src" / "minimal_mamba.py": staging / "tools" / "frozen_provenance" / "minimal_mamba.py",
        PROJECT_ROOT / "experiments" / "risk_mitigation_20260515" / "run_full_aux_penalty.py": staging / "tools" / "frozen_provenance" / "run_full_aux_penalty.py",
        PROJECT_ROOT / "experiments" / "risk_mitigation_20260515" / "run_concat_posthoc_jacobian.py": staging / "tools" / "frozen_provenance" / "run_concat_posthoc_jacobian.py",
        PROJECT_ROOT / "experiments" / "test_shortcut_diagnostics.py": staging / "tools" / "frozen_provenance" / "test_shortcut_diagnostics.py",
        PROJECT_ROOT / "experiments" / "test_mask_supplement.py": staging / "tools" / "frozen_provenance" / "test_mask_supplement.py",
        PROJECT_ROOT / "experiments" / "p0_jacobian_semantics_audit.py": staging / "tools" / "frozen_provenance" / "p0_jacobian_semantics_audit.py",
        Path(__file__).resolve(): staging / "tools" / Path(__file__).name,
    }
    for source, target in tools.items():
        copy_file(source, target, inventory)

    frozen = {
        PROJECT_ROOT / "diagnostic_results" / "exp2_dcond_sweep.json": staging / "frozen_evidence" / "dcond" / "exp2_dcond_sweep.json",
        PROJECT_ROOT / "results" / "raw" / "mask_supplement_results.json": staging / "frozen_evidence" / "mask" / "mask_supplement_results.json",
        PROJECT_ROOT / "diagnostic_results" / "exp4_coefficient_recovery.json": staging / "frozen_evidence" / "coefficient" / "exp4_coefficient_recovery.json",
        PROJECT_ROOT / "risk_mitigation_results" / "full_aux_jacobian_penalty.json": staging / "frozen_evidence" / "full_aux_penalty" / "full_aux_jacobian_penalty.json",
        PROJECT_ROOT / "risk_mitigation_results" / "full_aux_jacobian_penalty.csv": staging / "frozen_evidence" / "full_aux_penalty" / "full_aux_jacobian_penalty.csv",
        PROJECT_ROOT / "risk_mitigation_results" / "concat_posthoc_jacobian.json": staging / "frozen_evidence" / "concat_posthoc" / "concat_posthoc_jacobian.json",
        V23_STAGE / "frozen_evidence" / "stage1a" / "stage1a_aggregate_go_no_go.json": staging / "frozen_evidence" / "stage1a" / "stage1a_aggregate_go_no_go.json",
        V23_STAGE / "frozen_evidence" / "p1" / "p1_status.json": staging / "frozen_evidence" / "p1" / "p1_status.json",
        V23_STAGE / "frozen_evidence" / "p1" / "ab_decision_gates.json": staging / "frozen_evidence" / "p1" / "ab_decision_gates.json",
    }
    for seed in range(5):
        name = f"p0_jacobian_semantics_d6_iter120_refactor_seed{seed}.json"
        frozen[PROJECT_ROOT / "results" / "p0_audit" / name] = staging / "frozen_evidence" / "p0" / name
    for source, target in frozen.items():
        copy_file(source, target, inventory)

    if args.clean_log is not None:
        copy_file(args.clean_log, staging / "clean_extraction_compile.log", inventory)

    commit_info = {
        "kbs_repository_commit": git_commit(ELS_ROOT),
        "method_repository_commit": git_commit(PROJECT_ROOT),
        "canonical_manuscript_sha256": canonical_sha,
        "manuscript_sha256": sha256(ELS_ROOT / f"{MANUSCRIPT_STEM}.tex"),
        "no_new_experiment": True,
    }
    (staging / "COMMIT_INFO.json").write_text(
        json.dumps(commit_info, indent=2) + "\n", encoding="utf-8"
    )

    source_manifest = {
        "purpose": "Source and evidence manifest for the v2.4 submission candidate.",
        "canonical_guard_sha256": EXPECTED_CANONICAL_SHA,
        "files": inventory,
    }
    (staging / "SOURCE_MANIFEST_V2_4.json").write_text(
        json.dumps(source_manifest, indent=2) + "\n", encoding="utf-8"
    )

    sum_lines = []
    for path in sorted(p for p in staging.rglob("*") if p.is_file()):
        relative = path.relative_to(staging).as_posix()
        if relative == "PACKAGE_SHA256SUMS.txt":
            continue
        sum_lines.append(f"{sha256(path)}  {relative}")
    (staging / "PACKAGE_SHA256SUMS.txt").write_text(
        "\n".join(sum_lines) + "\n", encoding="ascii"
    )

    args.zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(p for p in staging.rglob("*") if p.is_file()):
            archive.write(path, path.relative_to(staging).as_posix())
    print(json.dumps({"staging": str(staging), "zip": str(args.zip_path), "files": len(sum_lines) + 1}, indent=2))


if __name__ == "__main__":
    main()
