"""Materialize the pre-registered Stage B rows after a passed Stage A gate."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


STAGE = "B_AUTODL_CONFIRMATION"
EXPECTED_COUNT = 36
EXPECTED_ROLE = "conditional_external_confirmation"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--stage-a-decision", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    decision = json.loads(
        args.stage_a_decision.read_text(encoding="utf-8-sig")
    )
    if (
        decision.get("decision") != "UNLOCK_STAGE_B"
        or decision.get("unlock_stage_b") is not True
        or decision.get("semantic_integrity_passed") is not True
        or decision.get("determinism", {}).get("passed") is not True
    ):
        raise RuntimeError("Stage A decision does not authorize Stage B")

    with args.source.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = [row for row in reader if row["stage"] == STAGE]
    if len(rows) != EXPECTED_COUNT:
        raise RuntimeError(
            f"Expected {EXPECTED_COUNT} sealed Stage B rows, got {len(rows)}"
        )
    if any(row["execution_authorized"] != "false" for row in rows):
        raise RuntimeError("Source Stage B rows must still be sealed")
    if any(row["evidence_role"] != EXPECTED_ROLE for row in rows):
        raise RuntimeError("Unexpected Stage B evidence role")
    if len({row["run_id"] for row in rows}) != EXPECTED_COUNT:
        raise RuntimeError("Stage B run IDs are not unique")

    authorized = []
    for row in rows:
        materialized = dict(row)
        materialized["execution_authorized"] = "true"
        authorized.append(materialized)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(authorized)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
