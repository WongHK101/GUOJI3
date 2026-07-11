"""Aggregate the final Phase 8 lambda trade-off or held-out confirmation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
for path in (PROJECT_ROOT, SRC_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from phase8_final_results import aggregate_confirmation, aggregate_lambda_tradeoff  # noqa: E402
from phase8_protocol import load_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("lambda_tradeoff", "confirmation"), required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-matrix", type=Path, required=True)
    parser.add_argument("--frozen-reference-root", type=Path)
    parser.add_argument("--cpu-preflight-report", type=Path)
    parser.add_argument("--pilot-report", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.stage == "lambda_tradeoff":
        if args.frozen_reference_root is None or args.cpu_preflight_report is None:
            raise ValueError("Lambda trade-off aggregation requires frozen reference and CPU preflight")
        report = aggregate_lambda_tradeoff(
            args.root,
            args.frozen_reference_root,
            config_path=args.config,
            matrix_path=args.run_matrix,
            cpu_preflight_report=load_json(args.cpu_preflight_report),
        )
        default_output = args.root / "lambda_tradeoff_aggregate_and_decision.json"
    else:
        if args.pilot_report is None:
            raise ValueError("Confirmation aggregation requires --pilot-report")
        report = aggregate_confirmation(
            args.root,
            config_path=args.config,
            matrix_path=args.run_matrix,
            pilot_report=load_json(args.pilot_report),
        )
        default_output = args.root / "confirmation_aggregate_and_classification.json"
    output = args.output or default_output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
