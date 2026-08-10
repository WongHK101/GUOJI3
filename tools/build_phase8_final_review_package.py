"""Build the terminal Phase 8 experiment-and-manuscript review package."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import zipfile


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def copy_file(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def copy_tree(source: Path, destination: Path) -> None:
    if not source.is_dir():
        raise FileNotFoundError(source)
    if destination.exists():
        raise FileExistsError(destination)
    shutil.copytree(source, destination)


def git_output(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=root, text=True).strip()


def copy_tracked_repository(root: Path, destination: Path) -> str:
    if git_output(root, "status", "--porcelain"):
        raise RuntimeError(f"Repository is not clean: {root}")
    commit = git_output(root, "rev-parse", "HEAD")
    tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=root).decode().split("\0")
    for relative in filter(None, tracked):
        source = root / relative
        if source.is_file():
            copy_file(source, destination / relative)
    return commit


def copy_final_manuscript(clean_root: Path, destination: Path) -> None:
    names = [
        "istf_kbs_jacobian_coverage_phase8_final.tex",
        "istf_kbs_jacobian_coverage_phase8_final.pdf",
        "istf_kbs_jacobian_coverage_phase8_final.log",
        "clean_compile_report.json",
        "clean_compile_pass1.stdout.log",
        "clean_compile_pass2.stdout.log",
        "clean_compile_pass3.stdout.log",
        "elsarticle.cls",
        "elsarticle-num-names.bst",
    ]
    for name in names:
        copy_file(clean_root / name, destination / name)
    for relative in (
        Path("figures/coverage_audit_phase8_final"),
        Path("source_data/coverage_audit_phase8_final"),
        Path("tables/coverage_audit_phase8_final"),
        Path("docs"),
    ):
        copy_tree(clean_root / relative, destination / relative)


def write_package_readme(path: Path, *, code_commit: str, manuscript_commit: str) -> None:
    text = f"""# Phase 8 Final Repair and Manuscript Revision

This is the terminal Phase 8 review artifact. It contains the complete bounded
lambda pilot, frozen comparator and Track A evidence, source code, release locks,
an independently compiled KBS manuscript, editable figures, source-data tables,
and decision/traceability documents.

## Locked decision

`STOP_METHOD_DEVELOPMENT_AUDIT_BOUNDARY_MANUSCRIPT`

No lambda passed the complete pilot-go rule. Held-out confirmation was not
eligible and was not executed. No further repair development is authorized.

## Commits

- Scientific GPU release: `78a85acd513fddde1744283c68f17e731692ba2e`
- Final code/documentation snapshot: `{code_commit}`
- Independent manuscript snapshot: `{manuscript_commit}`

## Directory map

- `artifacts/phase8_final_lambda_tradeoff_78a85ac/`: all 18 new runs and aggregate.
- `artifacts/phase8_recovery_execution_6f489b1/`: 30 frozen pilot/comparator runs.
- `artifacts/phase8_trackA_replication_dee0d30/`: five-pair replication evidence.
- `artifacts/phase8_final_cpu_preflight_78a85ac/`: accepted CPU semantic and estimator preflight.
- `frozen_evidence/`: P0, full-penalty, Stage 1a, and P1 boundary evidence.
- `source/code_repository/`: clean tracked code snapshot.
- `source/code_repository/tgc/`: frozen JRNGC dependency required by the adapters.
- `source/release_lock/`: exact GPU release lock and source manifest.
- `manuscript/`: clean-compiled TeX/PDF/log, figures, source data, and tables.
- `docs/`: decision, claim matrix, traceability, limitations, and reproduction guide.
- `local_reaggregate/`: independent local aggregate and logs.
- `SHA256_MANIFEST.txt`: package-relative hashes for every other file.

The package contains no confirmation output because confirmation was not eligible.
"""
    path.write_text(text, encoding="utf-8")


def write_manifest(root: Path) -> tuple[Path, int]:
    manifest = root / "SHA256_MANIFEST.txt"
    files = sorted(path for path in root.rglob("*") if path.is_file() and path != manifest)
    rows = [f"{sha256(path)}  {path.relative_to(root).as_posix()}" for path in files]
    manifest.write_text("\n".join(rows) + "\n", encoding="utf-8", newline="\n")
    return manifest, len(files)


def zip_tree(root: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(p for p in root.rglob("*") if p.is_file()):
            archive.write(path, path.relative_to(root).as_posix())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-zip", type=Path, required=True)
    parser.add_argument("--staging-dir", type=Path, required=True)
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--jrngc-root", type=Path, required=True)
    parser.add_argument("--manuscript-clean-root", type=Path, required=True)
    parser.add_argument("--manuscript-commit", required=True)
    parser.add_argument("--final-results-root", type=Path, required=True)
    parser.add_argument("--frozen-comparator-root", type=Path, required=True)
    parser.add_argument("--track-a-root", type=Path, required=True)
    parser.add_argument("--cpu-preflight-root", type=Path, required=True)
    parser.add_argument("--p0-audit-dir", type=Path, required=True)
    parser.add_argument("--full-aux-json", type=Path, required=True)
    parser.add_argument("--stage1a-json", type=Path, required=True)
    parser.add_argument("--p1-zip", type=Path, required=True)
    parser.add_argument("--local-reaggregate-root", type=Path, required=True)
    parser.add_argument("--release-lock-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.staging_dir.exists():
        raise FileExistsError(args.staging_dir)
    args.staging_dir.mkdir(parents=True)

    code_commit = copy_tracked_repository(args.code_root, args.staging_dir / "source/code_repository")
    copy_tree(args.jrngc_root / "tgc", args.staging_dir / "source/code_repository/tgc")
    copy_tree(args.release_lock_dir, args.staging_dir / "source/release_lock")
    copy_final_manuscript(args.manuscript_clean_root, args.staging_dir / "manuscript")
    copy_tree(args.final_results_root, args.staging_dir / "artifacts/phase8_final_lambda_tradeoff_78a85ac")
    copy_tree(args.frozen_comparator_root, args.staging_dir / "artifacts/phase8_recovery_execution_6f489b1")
    copy_tree(args.track_a_root, args.staging_dir / "artifacts/phase8_trackA_replication_dee0d30")
    copy_tree(args.cpu_preflight_root, args.staging_dir / "artifacts/phase8_final_cpu_preflight_78a85ac")
    copy_tree(args.local_reaggregate_root, args.staging_dir / "local_reaggregate")

    p0_destination = args.staging_dir / "frozen_evidence/p0"
    p0_paths = sorted(args.p0_audit_dir.glob("p0_jacobian_semantics_d6_iter120_refactor_seed*.json"))
    if len(p0_paths) != 5:
        raise RuntimeError(f"Expected five P0 audit files, found {len(p0_paths)}")
    for path in p0_paths:
        copy_file(path, p0_destination / path.name)
    copy_file(args.full_aux_json, args.staging_dir / "frozen_evidence/full_aux_jacobian_penalty.json")
    copy_file(args.stage1a_json, args.staging_dir / "frozen_evidence/stage1a/stage1a_aggregate_go_no_go.json")
    copy_file(args.p1_zip, args.staging_dir / "frozen_evidence/phase7_stage1a_bounded_failure_analysis_v1.zip")

    docs_source = args.code_root / "paper-data/docs/phase8_final"
    copy_tree(docs_source, args.staging_dir / "docs")
    write_package_readme(
        args.staging_dir / "README.md",
        code_commit=code_commit,
        manuscript_commit=args.manuscript_commit,
    )
    metadata = {
        "scientific_release_commit": "78a85acd513fddde1744283c68f17e731692ba2e",
        "code_documentation_commit": code_commit,
        "manuscript_commit": args.manuscript_commit,
        "method_decision": "STOP_METHOD_DEVELOPMENT_AUDIT_BOUNDARY_MANUSCRIPT",
        "confirmation_eligible": False,
        "confirmation_executed": False,
    }
    (args.staging_dir / "PACKAGE_METADATA.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    manifest, count = write_manifest(args.staging_dir)
    zip_tree(args.staging_dir, args.output_zip)
    print(
        json.dumps(
            {
                "output_zip": str(args.output_zip),
                "zip_sha256": sha256(args.output_zip),
                "zip_bytes": args.output_zip.stat().st_size,
                "manifest": str(manifest),
                "manifested_files": count,
                "code_commit": code_commit,
                "manuscript_commit": args.manuscript_commit,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
