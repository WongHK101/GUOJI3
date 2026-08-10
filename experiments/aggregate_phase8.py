"""Aggregate frozen Phase 8 replication or repair-pilot records."""

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
from phase8_results import aggregate_pilot, aggregate_replication  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["replication", "pilot"], required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-matrix", type=Path, required=True)
    parser.add_argument("--preflight-report", type=Path)
    parser.add_argument("--replication-report", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.stage == "replication":
        report = aggregate_replication(args.root, config_path=args.config, matrix_path=args.run_matrix)
        output = args.root / "replication_aggregate_and_gates.json"
    else:
        if args.preflight_report is None or args.replication_report is None:
            raise ValueError("Pilot aggregation requires --preflight-report and --replication-report")
        report = aggregate_pilot(
            args.root,
            config_path=args.config,
            matrix_path=args.run_matrix,
            preflight_report=load_json(args.preflight_report),
            replication_report=load_json(args.replication_report),
        )
        output = args.root / "repair_pilot_aggregate_and_gates.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
