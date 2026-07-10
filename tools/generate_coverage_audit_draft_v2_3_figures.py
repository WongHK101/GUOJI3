"""Render draft v2.3 figures for the Jacobian coverage audit manuscript.

The script reads only frozen diagnostic artifacts and writes vector and raster
exports for a separate manuscript draft. It does not train models or modify
any experimental result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
import numpy as np


# Editable SVG text is required for the manuscript bundle.
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams.update(
    {
        "font.size": 8,
        "axes.linewidth": 0.8,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "legend.frameon": False,
    }
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
DEFAULT_OUTPUT_DIR = (
    WORKSPACE_ROOT / "elsarticle" / "figures" / "coverage_audit_draft_v2_3"
)

PALETTE = {
    "ink": "#272727",
    "blue": "#0F4D92",
    "blue_soft": "#B4C0E4",
    "red": "#B64342",
    "red_soft": "#F6CFCB",
    "green": "#2E7D32",
    "green_soft": "#DDF3DE",
    "teal": "#42949E",
    "teal_soft": "#DDEEEF",
    "gold": "#E0A019",
    "gold_soft": "#F8E9B2",
    "gray": "#767676",
    "gray_soft": "#E5E5E5",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def add_panel_label(ax, label: str) -> None:
    ax.text(
        -0.05,
        1.03,
        label,
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        va="bottom",
        ha="left",
        color=PALETTE["ink"],
    )


def orthogonal_arrow(ax, start, end, color, lw=1.5, linestyle="-") -> None:
    """Draw a compact, horizontal-or-vertical connector with one arrowhead."""
    sx, sy = start
    ex, ey = end
    if abs(sx - ex) > 1e-8 and abs(sy - ey) > 1e-8:
        midx = (sx + ex) / 2
        ax.plot([sx, midx, midx], [sy, sy, ey], color=color, lw=lw, ls=linestyle)
        start = (midx, ey)
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=10,
            linewidth=lw,
            linestyle=linestyle,
            color=color,
        )
    )


def rounded_box(ax, x, y, w, h, text, *, face, edge, fontsize=7.5, weight=None):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.04",
        facecolor=face,
        edgecolor=edge,
        linewidth=1.0,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight=weight,
        color=PALETTE["ink"],
        wrap=True,
    )
    return patch


def save_figure(fig, output_dir: Path, stem: str) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for extension, dpi in (("svg", None), ("pdf", None), ("png", 600)):
        path = output_dir / f"{stem}.{extension}"
        kwargs = {"bbox_inches": "tight", "facecolor": "white"}
        if dpi is not None:
            kwargs["dpi"] = dpi
        fig.savefig(path, **kwargs)
        if extension == "svg":
            # Matplotlib emits trailing spaces in path records; remove them so
            # generated vector assets pass repository whitespace checks.
            path.write_text(
                "\n".join(line.rstrip() for line in path.read_text(encoding="utf-8").splitlines())
                + "\n",
                encoding="utf-8",
            )
        outputs.append(path)
    plt.close(fig)
    return outputs


def draw_coverage_mismatch(output_dir: Path) -> list[Path]:
    """Figure 1: route-ledger coverage audit argument."""
    fig = plt.figure(figsize=(7.25, 3.05))
    grid = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.08, 0.9], wspace=0.22)

    ax = fig.add_subplot(grid[0, 0])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.add_patch(
        Rectangle((0.15, 0.2), 9.7, 9.5, facecolor="none", edgecolor=PALETTE["gray"],
                  linewidth=0.9, linestyle=(0, (3, 2)))
    )
    add_panel_label(ax, "a")
    ax.text(5, 9.25, "Controlled concat diagnostic", ha="center", va="center", fontsize=8.5,
            fontweight="bold")
    rounded_box(ax, 0.75, 6.4, 2.15, 1.15, "history\n" + r"$X_{t-K:t-1}$", face=PALETTE["blue_soft"], edge=PALETTE["blue"], fontsize=6.2, weight="bold")
    rounded_box(ax, 0.75, 2.5, 2.15, 1.15, "auxiliary\n" + r"$c_t$", face=PALETTE["red_soft"], edge=PALETTE["red"], fontsize=6.5, weight="bold")
    rounded_box(ax, 4.0, 4.4, 2.1, 1.25, "concat\npredictor", face="#F7F7F7", edge=PALETTE["ink"], weight="bold")
    rounded_box(ax, 7.15, 4.4, 1.9, 1.25, "prediction\n" + r"$\hat{x}_{t+1}$", face=PALETTE["teal_soft"], edge=PALETTE["teal"], fontsize=6.7, weight="bold")
    orthogonal_arrow(ax, (2.8, 6.98), (4.0, 5.15), PALETTE["blue"])
    orthogonal_arrow(ax, (2.8, 3.07), (4.0, 4.9), PALETTE["red"], linestyle=(0, (3, 2)))
    orthogonal_arrow(ax, (6.1, 5.0), (7.15, 5.0), PALETTE["ink"])
    ax.add_patch(
        Rectangle((3.7, 5.95), 2.7, 2.45, facecolor="none", edgecolor=PALETTE["blue"],
                  linewidth=1.0, linestyle=(0, (3, 2))))
    ax.text(5.05, 8.05, r"x-only score and penalty", ha="center", va="center", fontsize=6.6,
            color=PALETTE["blue"], fontweight="bold")
    ax.text(5.05, 7.58, r"$S_x, R_x$", ha="center", va="center", fontsize=8.2,
            color=PALETTE["blue"])
    ax.text(5.2, 1.25, "Predictor can use c while\ngraph score reports x only.", ha="center", va="center",
            fontsize=7.1, color=PALETTE["red"], fontweight="bold")

    ax = fig.add_subplot(grid[0, 1])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.add_patch(
        Rectangle((0.15, 0.2), 9.7, 9.5, facecolor="none", edgecolor=PALETTE["gray"],
                  linewidth=0.9, linestyle=(0, (3, 2)))
    )
    add_panel_label(ax, "b")
    ax.text(5, 9.25, "Architecture route ledger", ha="center", va="center", fontsize=8.5,
            fontweight="bold")
    rounded_box(ax, 0.7, 7.45, 8.6, 0.95, r"$C=(V_{\rm score},V_{\rm penalty},P_{\rm pred},M_{\rm coord},H_{\rm attr})$",
                face="#F7F7F7", edge=PALETTE["ink"], fontsize=5.2, weight="bold")
    ledger = [
        (r"$r_x$: direct history", "scored", "penalized", "not-exempt", PALETTE["blue_soft"], PALETTE["blue"]),
        (r"$r_c$: auxiliary route", "not-scored", "unpenalized", "not-exempt", PALETTE["red_soft"], PALETTE["red"]),
    ]
    ax.text(0.75, 6.45, "route class", fontsize=5.2, fontweight="bold", ha="left")
    ax.text(4.62, 6.45, "score", fontsize=5.2, fontweight="bold", ha="center")
    ax.text(6.47, 6.45, "penalty", fontsize=5.2, fontweight="bold", ha="center")
    ax.text(8.60, 6.45, "exemption", fontsize=5.2, fontweight="bold", ha="center")
    for idx, (route, score, penalty, exemption, face, edge) in enumerate(ledger):
        y = 5.35 - idx * 1.65
        rounded_box(ax, 0.35, y, 3.35, 0.95, route, face=face, edge=edge, fontsize=4.7, weight="bold")
        rounded_box(ax, 3.85, y, 1.55, 0.95, score, face="#F7F7F7", edge=edge, fontsize=3.9)
        rounded_box(ax, 5.55, y, 1.85, 0.95, penalty, face="#F7F7F7", edge=edge, fontsize=3.9)
        rounded_box(ax, 7.55, y, 2.10, 0.95, exemption, face="#F7F7F7", edge=edge, fontsize=3.8)
    rounded_box(
        ax,
        0.9,
        1.05,
        8.2,
        1.25,
        "Record all three statuses independently.\nScored and penalized can coexist.",
        face=PALETTE["gray_soft"],
        edge=PALETTE["gray"],
        fontsize=5.0,
    )

    ax = fig.add_subplot(grid[0, 2])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.add_patch(
        Rectangle((0.15, 0.2), 9.7, 9.5, facecolor="none", edgecolor=PALETTE["gray"],
                  linewidth=0.9, linestyle=(0, (3, 2)))
    )
    add_panel_label(ax, "c")
    ax.text(5, 9.25, "Claim-specific audit profile", ha="center", va="center", fontsize=7.2, fontweight="bold")
    ax.text(5, 8.55, "Report every applicable flag", ha="center", va="center", fontsize=5.8, color=PALETTE["gray"])
    rounded_box(
        ax,
        1.05,
        6.95,
        7.90,
        1.05,
        "CLAIM-COVERED\nall 5 pass; no unknowns",
        face=PALETTE["green_soft"],
        edge=PALETTE["green"],
        fontsize=5.6,
        weight="bold",
    )
    profile_flags = [
        (0.65, 4.85, "PARTIALLY\nCOVERED", PALETTE["gold_soft"], PALETTE["gold"]),
        (5.15, 4.85, "COORDINATE-\nAMBIGUOUS", PALETTE["red_soft"], PALETTE["red"]),
        (0.65, 2.85, "HORIZON-\nTRUNCATED", PALETTE["gray_soft"], PALETTE["gray"]),
        (5.15, 2.85, "UNASSESSED", "#F7F7F7", PALETTE["gray"]),
    ]
    for x, y, label, face, edge in profile_flags:
        rounded_box(ax, x, y, 4.20, 1.25, label, face=face, edge=edge, fontsize=5.3, weight="bold")
    rounded_box(
        ax,
        1.70,
        0.85,
        6.60,
        0.95,
        "Failure and unassessed\nflags may coexist.",
        face=PALETTE["gray_soft"],
        edge=PALETTE["gray"],
        fontsize=4.9,
        weight="bold",
    )
    return save_figure(fig, output_dir, "fig1_coverage_mismatch_draft_v2_3")


def draw_controlled_diagnostics(data: dict, output_dir: Path) -> list[Path]:
    """Figure 2: controlled concat diagnostics, no legacy performance comparison."""
    sweep = data["dcond"]
    masks = data["masks"]
    coeff = data["coeff"]
    penalty = data["penalty"]["summary"]

    fig = plt.figure(figsize=(7.25, 5.6))
    grid = fig.add_gridspec(2, 2, wspace=0.56, hspace=0.42)

    ax = fig.add_subplot(grid[0, 0])
    ordered = sorted(sweep.values(), key=lambda item: item["d_cond"])
    x = [row["d_cond"] for row in ordered]
    auroc = [row["auroc"] for row in ordered]
    loss = [row["train_loss"] for row in ordered]
    first = ax.plot(x, auroc, marker="o", color=PALETTE["blue"], lw=1.8, label="AUROC")
    ax.set_xlabel(r"auxiliary dimension $d_{\rm cond}$")
    ax.set_ylabel("AUROC", color=PALETTE["blue"])
    ax.tick_params(axis="y", colors=PALETTE["blue"])
    ax.set_ylim(0.2, 1.0)
    ax.set_xticks(x)
    ax.grid(axis="y", alpha=0.18)
    twin = ax.twinx()
    second = twin.plot(x, loss, marker="s", color=PALETTE["red"], lw=1.6, label="training loss")
    twin.set_ylabel("training loss", color=PALETTE["red"])
    twin.tick_params(axis="y", colors=PALETTE["red"])
    twin.set_ylim(0, 0.010)
    ax.legend(first + second, [line.get_label() for line in first + second], loc="upper right", fontsize=6.8)
    ax.set_title("Auxiliary capacity decouples loss and score", loc="left", x=0.10, fontsize=8.5, fontweight="bold")
    add_panel_label(ax, "a")

    ax = fig.add_subplot(grid[0, 1])
    intervention_labels = [
        "Concat\nmask x",
        "Concat\nmask c",
        "Concat\nmask both",
        "Concat\nshuffle x",
        "No-aux\nmask x",
    ]
    intervention_values = [
        masks["concat_mask_x_only_delta"],
        masks["concat_mask_c_only_delta"],
        masks["concat_mask_both_delta"],
        masks["concat_shuffle_delta"],
        masks["istf_mask_delta"],
    ]
    intervention_colors = [PALETTE["red_soft"]] * 4 + [PALETTE["blue_soft"]]
    bars = ax.bar(range(len(intervention_values)), intervention_values, color=intervention_colors,
                  edgecolor=[PALETTE["red"]] * 4 + [PALETTE["blue"]], linewidth=0.9)
    for bar, value in zip(bars, intervention_values):
        ax.text(bar.get_x() + bar.get_width() / 2, value * 1.12, f"{value:.2f}", ha="center", va="bottom", fontsize=6.4)
    ax.set_yscale("log")
    ax.set_ylim(0.2, 60)
    ax.set_ylabel(r"$\Delta$ prediction loss (log)")
    ax.set_xticks(range(len(intervention_labels)))
    ax.set_xticklabels(intervention_labels, fontsize=6.7)
    ax.set_title("Interventions expose an auxiliary route", loc="left", x=0.10, fontsize=8.5, fontweight="bold")
    ax.grid(axis="y", alpha=0.18)
    add_panel_label(ax, "b")

    ax = fig.add_subplot(grid[1, 0])
    methods = ["Baseline\nJRNGC", "Concat\nx-only"]
    correlation = [coeff["Baseline"]["coefficient_correlation"], coeff["Concat"]["coefficient_correlation"]]
    ratio = [coeff["Baseline"]["coefficient_shrinkage"], coeff["Concat"]["coefficient_shrinkage"]]
    xpos = np.arange(len(methods))
    width = 0.33
    ax.bar(xpos - width / 2, correlation, width, label="Pearson r", color=PALETTE["blue_soft"], edgecolor=PALETTE["blue"])
    ax.bar(xpos + width / 2, ratio, width, label="norm ratio", color=PALETTE["red_soft"], edgecolor=PALETTE["red"])
    ax.axhline(1.0, color=PALETTE["gray"], lw=0.9, ls="--")
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("coefficient metric value")
    ax.set_xticks(xpos)
    ax.set_xticklabels(methods)
    ax.legend(loc="lower left", fontsize=6.8)
    ax.set_title("Concat reduces coefficient fidelity", loc="left", x=0.10, fontsize=8.5, fontweight="bold")
    ax.grid(axis="y", alpha=0.18)
    add_panel_label(ax, "c")

    ax = fig.add_subplot(grid[1, 1])
    selected = ["baseline", "concat_x_only", "full_lc_10"]
    labels = ["Baseline", "Concat\nx-only", "Full auxiliary\npenalty"]
    auroc_values = [penalty[key]["mean"]["auroc"] for key in selected]
    auroc_std = [penalty[key]["std"]["auroc"] for key in selected]
    corr_values = [penalty[key]["mean"]["coefficient_correlation"] for key in selected]
    corr_std = [penalty[key]["std"]["coefficient_correlation"] for key in selected]
    xpos = np.arange(len(labels))
    bars = ax.bar(xpos - width / 2, auroc_values, width, yerr=auroc_std, capsize=2.0,
                  label="AUROC", color=PALETTE["blue_soft"], edgecolor=PALETTE["blue"],
                  error_kw={"elinewidth": 0.8, "ecolor": PALETTE["blue"]})
    ax.bar(xpos + width / 2, corr_values, width, yerr=corr_std, capsize=2.0,
           label="coefficient r", color=PALETTE["green_soft"], edgecolor=PALETTE["green"],
           error_kw={"elinewidth": 0.8, "ecolor": PALETTE["green"]})
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("recovery metric")
    ax.set_xticks(xpos)
    ax.set_xticklabels(labels, fontsize=6.7)
    ax.legend(loc="lower left", fontsize=6.8)
    ax.set_title("Expanded penalty coverage mitigates\nx-only degradation", loc="left", x=0.10, fontsize=8.1, fontweight="bold")
    ax.grid(axis="y", alpha=0.18)
    add_panel_label(ax, "d")

    return save_figure(fig, output_dir, "fig2_controlled_concat_diagnostics_draft_v2_3")


def summarize_p0_audits(audits: list[dict]) -> dict[str, dict[str, float]]:
    """Aggregate the fixed five-seed P0 audit without changing source values."""
    metric_paths = {
        "concat_corr": ("concat_partial_vs_total_derivative", "offdiag_score_correlation"),
        "legacy_corr": ("istf_mamba_filtered_vs_raw_chain", "offdiag_score_correlation"),
        "legacy_jaccard": ("istf_mamba_filtered_vs_raw_chain", "topk_jaccard"),
    }
    summary: dict[str, dict[str, float]] = {}
    for name, (section, metric) in metric_paths.items():
        values = np.asarray([audit[section][metric] for audit in audits], dtype=float)
        summary[name] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=0)),
            "min": float(values.min()),
            "max": float(values.max()),
        }
    return summary


def draw_score_semantics(data: dict, output_dir: Path) -> list[Path]:
    """Figure 3: five-seed score-coordinate and horizon semantic audit."""
    summary = summarize_p0_audits(data["p0_audits"])

    fig = plt.figure(figsize=(7.25, 3.0))
    grid = fig.add_gridspec(1, 3, width_ratios=[1.08, 0.9, 1.12], wspace=0.62)

    ax = fig.add_subplot(grid[0, 0])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    add_panel_label(ax, "a")
    ax.text(5, 9.2, "Score semantics are distinct", ha="center", va="center", fontsize=7.5, fontweight="bold")
    entries = [
        (r"x-only: $\partial \hat{x}/\partial x$", PALETTE["blue_soft"], PALETTE["blue"]),
        (r"auxiliary: $\partial \hat{x}/\partial c$", PALETTE["red_soft"], PALETTE["red"]),
        (r"filtered: $\partial \hat{x}/\partial x'$", PALETTE["gold_soft"], PALETTE["gold"]),
        (r"raw-chain: $d\hat{x}/dx$", PALETTE["green_soft"], PALETTE["green"]),
    ]
    for idx, (label, face, edge) in enumerate(entries):
        rounded_box(ax, 0.55, 7.55 - 1.75 * idx, 8.9, 0.95, label, face=face, edge=edge, fontsize=6.6, weight="bold")
    ax.text(5, 0.85, "A graph interpretation must state\nwhich derivative is aggregated.", ha="center", va="center", fontsize=5.8, color=PALETTE["gray"])

    ax = fig.add_subplot(grid[0, 1])
    values = [summary["concat_corr"]["mean"], summary["legacy_corr"]["mean"], summary["legacy_jaccard"]["mean"]]
    errors = [summary["concat_corr"]["std"], summary["legacy_corr"]["std"], summary["legacy_jaccard"]["std"]]
    labels = ["Concat\nP/T r", "Legacy\nF/R r", "Legacy\ntop-k J"]
    colors = [PALETTE["red_soft"], PALETTE["red_soft"], PALETTE["gold_soft"]]
    edges = [PALETTE["red"], PALETTE["red"], PALETTE["gold"]]
    bars = ax.bar(np.arange(3), values, yerr=errors, capsize=2.1, color=colors, edgecolor=edges,
                  linewidth=0.9, error_kw={"elinewidth": 0.8, "ecolor": PALETTE["ink"]})
    for bar, value, error in zip(bars, values, errors):
        ax.text(bar.get_x() + bar.get_width() / 2, value + error + 0.04, f"{value:.3f}", ha="center", va="bottom", fontsize=6.3)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("agreement")
    ax.set_xticks(np.arange(3))
    ax.set_xticklabels(labels, fontsize=5.6)
    ax.set_title("Five-seed semantic audit", loc="left", x=0.10, fontsize=7.5, fontweight="bold")
    ax.grid(axis="y", alpha=0.18)
    ax.text(-0.29, 1.03, "b", transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom", ha="left", color=PALETTE["ink"])

    ax = fig.add_subplot(grid[0, 2])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    add_panel_label(ax, "c")
    ax.text(5, 9.2, "Coordinate validity / horizon validity", ha="center", va="center", fontsize=6.8, fontweight="bold")
    rounded_box(ax, 0.75, 6.85, 3.0, 1.1, "filtered\nspace", face=PALETTE["gold_soft"], edge=PALETTE["gold"], fontsize=6.1, weight="bold")
    rounded_box(ax, 6.25, 6.85, 3.0, 1.1, "source\nspace", face=PALETTE["blue_soft"], edge=PALETTE["blue"], fontsize=6.1, weight="bold")
    ax.plot([3.75, 6.25], [7.4, 7.4], color=PALETTE["red"], lw=1.5, ls=(0, (3, 2)))
    ax.text(5, 7.78, "cross-channel map", ha="center", va="bottom", fontsize=5.5, color=PALETTE["red"])
    rounded_box(ax, 0.65, 3.05, 8.7, 1.9, "Raw-chain attribution supports\ncoordinate validity.\nAudit horizon support separately.", face=PALETTE["green_soft"], edge=PALETTE["green"], fontsize=5.0, weight="bold")
    ax.text(5, 1.4, "Legacy filtered-coordinate Mamba: COORDINATE-AMBIGUOUS\nScore-semantics diagnostic, not a performance comparison.",
            ha="center", va="center", fontsize=5.3, color=PALETTE["gray"])
    return save_figure(fig, output_dir, "fig3_score_semantics_audit_draft_v2_3")


def collect_data() -> tuple[dict, dict]:
    project_sources = {
        "dcond": PROJECT_ROOT / "diagnostic_results" / "exp2_dcond_sweep.json",
        "masks": PROJECT_ROOT / "results" / "raw" / "mask_supplement_results.json",
        "coeff": PROJECT_ROOT / "diagnostic_results" / "exp4_coefficient_recovery.json",
        "penalty": PROJECT_ROOT / "risk_mitigation_results" / "full_aux_jacobian_penalty.json",
        "p0_audits": [
            PROJECT_ROOT / "results" / "p0_audit" / f"p0_jacobian_semantics_d6_iter120_refactor_seed{seed}.json"
            for seed in range(5)
        ],
    }
    frozen_root = PROJECT_ROOT / "frozen_evidence"
    package_sources = {
        "dcond": frozen_root / "dcond" / "exp2_dcond_sweep.json",
        "masks": frozen_root / "mask" / "mask_supplement_results.json",
        "coeff": frozen_root / "coefficient" / "exp4_coefficient_recovery.json",
        "penalty": frozen_root / "full_aux_penalty" / "full_aux_jacobian_penalty.json",
        "p0_audits": [
            frozen_root / "p0" / f"p0_jacobian_semantics_d6_iter120_refactor_seed{seed}.json"
            for seed in range(5)
        ],
    }
    sources = project_sources if project_sources["dcond"].is_file() else package_sources
    data = {
        name: ([load_json(path) for path in path_or_paths] if isinstance(path_or_paths, list) else load_json(path_or_paths))
        for name, path_or_paths in sources.items()
    }
    source_rows = []
    for name, path_or_paths in sources.items():
        paths = path_or_paths if isinstance(path_or_paths, list) else [path_or_paths]
        source_rows.extend({"name": name, "path": str(path), "sha256": sha256(path)} for path in paths)
    manifest = {
        "purpose": "Frozen source inputs used by the draft v2.3 science-lock figures.",
        "sources": source_rows,
        "notes": {
            "fig1": "Conceptual controlled-concat schematic. It represents an x-only score and corresponding x-only penalty, not every auxiliary-conditioning architecture.",
            "fig2": "Frozen controlled concat diagnostics. Panels a-c are explicitly single-run diagnostics; panel d is the five-seed result. The no-aux intervention comparator is used only to contextualize route sensitivity, not to claim method performance.",
            "fig3": "Five fixed P0 CPU semantic-audit seeds. Legacy cross-channel Mamba is shown only as a coordinate-semantics failure diagnostic.",
        },
    }
    return data, manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    data, manifest = collect_data()
    generated = []
    generated.extend(draw_coverage_mismatch(args.output_dir))
    generated.extend(draw_controlled_diagnostics(data, args.output_dir))
    generated.extend(draw_score_semantics(data, args.output_dir))
    manifest["generated_files"] = [path.name for path in generated]
    with (args.output_dir / "figure_data_manifest_draft_v2_3.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    print(f"Generated {len(generated)} figure exports in {args.output_dir}")


if __name__ == "__main__":
    main()
