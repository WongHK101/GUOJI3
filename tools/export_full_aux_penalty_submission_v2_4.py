"""Export the frozen full-auxiliary-penalty sensitivity table for v2.4.

This is a read-only transformation of the saved five-seed JSON. It does not
train, evaluate, or alter a model. Lambda values are taken from the frozen
generating script and are recorded explicitly because its legacy CSV labels all
rows approximately and is unsuitable as the submission table source.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
DEFAULT_INPUT = PROJECT_ROOT / "risk_mitigation_results" / "full_aux_jacobian_penalty.json"
DEFAULT_OUTPUT = (
    WORKSPACE_ROOT
    / "elsarticle"
    / "tables"
    / "coverage_audit_submission_v2_4"
)

VARIANTS = [
    ("baseline", "Baseline JRNGC", "0.01", "N/A"),
    ("concat_x_only", "Concat x-only", "0.01", "0"),
    ("full_same_lambda", "Full, equal penalties", "0.01", "0.01"),
    ("full_budget_norm", "Full, budget normalized", "0.00667", "0.00667"),
    ("full_lc_01", r"Full, $\lambda_c/\lambda_x=0.1$", "0.01", "0.001"),
    ("full_lc_10", r"Full, $\lambda_c/\lambda_x=10$", "0.01", "0.1"),
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def value(summary: dict, metric: str) -> tuple[float | None, float | None]:
    mean = summary.get("mean", {}).get(metric)
    std = summary.get("std", {}).get(metric)
    return mean, std


def cell(mean: float | None, std: float | None, digits: int) -> str:
    if mean is None or std is None:
        return "N/A"
    return f"{mean:.{digits}f} $\\pm$ {std:.{digits}f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    payload = json.loads(args.input.read_text(encoding="utf-8"))
    summaries = payload["summary"]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for key, label, lam_x, lam_c in VARIANTS:
        summary = summaries[key]
        auroc = value(summary, "auroc")
        corr = value(summary, "coefficient_correlation")
        loss = value(summary, "pred_loss")
        aux = value(summary, "jc_norm")
        rows.append(
            {
                "key": key,
                "label": label,
                "lambda_x": lam_x,
                "lambda_c": lam_c,
                "auroc_mean": auroc[0],
                "auroc_population_sd": auroc[1],
                "coefficient_correlation_mean": corr[0],
                "coefficient_correlation_population_sd": corr[1],
                "prediction_loss_mean": loss[0],
                "prediction_loss_population_sd": loss[1],
                "auxiliary_jacobian_mean_absolute_magnitude_mean": aux[0],
                "auxiliary_jacobian_mean_absolute_magnitude_population_sd": aux[1],
            }
        )

    csv_path = args.output_dir / "full_aux_penalty_all_variants_v2_4.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    tex_lines = [
        r"\begin{tabularx}{\textwidth}{p{0.19\textwidth}ccYYYY}",
        r"\toprule",
        r"Variant & $\lambda_x$ & $\lambda_c$ & AUROC & Coefficient $r$ & Prediction loss & Mean $|J_c|$ \\",
        r"\midrule",
    ]
    for row in rows:
        auroc = cell(row["auroc_mean"], row["auroc_population_sd"], 3)
        corr = cell(
            row["coefficient_correlation_mean"],
            row["coefficient_correlation_population_sd"],
            3,
        )
        loss = cell(row["prediction_loss_mean"], row["prediction_loss_population_sd"], 5)
        aux = cell(
            row["auxiliary_jacobian_mean_absolute_magnitude_mean"],
            row["auxiliary_jacobian_mean_absolute_magnitude_population_sd"],
            4,
        )
        tex_lines.append(
            f'{row["label"]} & {row["lambda_x"]} & {row["lambda_c"]} & '
            f"{auroc} & {corr} & {loss} & {aux} \\\\"
        )
    tex_lines.extend([r"\bottomrule", r"\end{tabularx}"])
    tex_path = args.output_dir / "full_aux_penalty_all_variants_v2_4.tex"
    tex_path.write_text("\n".join(tex_lines) + "\n", encoding="ascii")

    manifest = {
        "purpose": "Frozen-data disclosure of every controlled full-penalty variant used in v2.4.",
        "input_path": str(args.input),
        "input_sha256": sha256(args.input),
        "generating_script": str(Path(__file__).resolve()),
        "aggregation": "Five saved seeds; mean and population SD copied from the frozen JSON summary.",
        "lambda_provenance": "experiments/risk_mitigation_20260515/run_full_aux_penalty.py lines 224-263",
        "interpretation": "Exploratory controlled penalty-strength diagnostic; full_lc_10 was tested, not preregistered as optimal.",
        "outputs": [csv_path.name, tex_path.name],
    }
    manifest_path = args.output_dir / "full_aux_penalty_table_manifest_v2_4.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {csv_path}")
    print(f"Wrote {tex_path}")
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
