"""Validate the five non-evidentiary Phase 8 GPU preflight records."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
for path in [PROJECT_ROOT, SRC_DIR]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from phase8_protocol import load_json  # noqa: E402
from phase8_results import validate_gpu_preflight  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-matrix", type=Path, required=True)
    parser.add_argument("--cpu-preflight-summary", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = args.root / "gpu_preflight_validation.json"
    try:
        cpu = load_json(args.cpu_preflight_summary) if args.cpu_preflight_summary else None
        report = validate_gpu_preflight(
            args.root,
            config_path=args.config,
            matrix_path=args.run_matrix,
            cpu_preflight_summary=cpu,
        )
    except Exception as exc:
        report = {"passed": False, "failures": [{"gate": "validator_exception", "error": str(exc)}]}
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report.get("passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
