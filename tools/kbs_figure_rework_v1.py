"""Top-journal figure rework for the KBS manuscript.

This script redraws root-cause and checkpoint-dynamics figures from existing
archived outputs only; it does not rerun training or alter experimental data.
"""

import csv
import json
import os
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedFormatter, FixedLocator, NullFormatter, NullLocator


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "paper-data", "figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "Liberation Sans"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 7,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.8,
        "axes.labelsize": 7.5,
        "axes.titlesize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.frameon": False,
    }
)

METHODS = ["JRNGC", "Concat-JRNGC", "ISTF-Mamba", "EMA-JRNGC"]
DATASETS = ["Linear", "Nonstationary"]
DATASET_KEYS = ["linear", "nonstationary"]

METHOD_STYLE = {
    "JRNGC": {"color": "#5A6F9F", "marker": "o", "label": "JRNGC"},
    "Concat-JRNGC": {"color": "#A7B4CF", "marker": "s", "label": "Concat-JRNGC"},
    "ISTF-Mamba": {"color": "#D99AAA", "marker": "^", "label": "ISTF-Mamba"},
    "EMA-JRNGC": {"color": "#39375F", "marker": "D", "label": "EMA-JRNGC"},
    "ISTF-TCN": {"color": "#B98EA5", "marker": "s", "label": "ISTF-TCN"},
    "Random": {"color": "#8C8C8C", "marker": "_", "label": "Random top-k"},
}


def save_figure(fig, name):
    base = os.path.join(OUTPUT_DIR, name)
    fig.savefig(base + ".svg", bbox_inches="tight")
    fig.savefig(base + ".pdf", bbox_inches="tight")
    fig.savefig(base + ".png", dpi=600, bbox_inches="tight")
    print(f"saved {base}.{{svg,pdf,png}}")


def panel_label(ax, label):
    ax.text(
        -0.18,
        1.08,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9.5,
        fontweight="bold",
        color="#222222",
    )


def summarize_ct_medical():
    metrics_dir = os.path.join(PROJECT_ROOT, "results_kbs", "metrics")
    summary = {}
    for method in ["JRNGC", "ISTF-Mamba"]:
        auroc = []
        f1 = []
        for seed in range(10):
            path = os.path.join(metrics_dir, f"CT_medical_{method}_seed{seed}_metrics.json")
            with open(path, encoding="utf-8") as f:
                payload = json.load(f)
            auroc.append(float(payload["standard_metrics"]["auroc"]))
            f1.append(float(payload["knowledge_metrics"]["f1_exact"]))
        summary[method] = {
            "auroc": float(np.mean(auroc)),
            "auroc_se": float(np.std(auroc, ddof=1) / np.sqrt(len(auroc))),
            "f1": float(np.mean(f1)),
            "f1_se": float(np.std(f1, ddof=1) / np.sqrt(len(f1))),
        }
    return summary


def load_root_cause_stats():
    base = os.path.join(PROJECT_ROOT, "results_kbs", "root_cause_v2", "summary")
    raw = {
        ds_key: {m: {"auroc": [], "f1": [], "path_f1": [], "pred_loss": []} for m in METHODS}
        for ds_key in DATASET_KEYS
    }

    for ds_key, stem in [
        ("linear", "root_cause_linear"),
        ("nonstationary", "root_cause_nonstationary"),
    ]:
        for seed in range(5):
            path = os.path.join(base, f"{stem}_data{seed}_summary.json")
            with open(path, encoding="utf-8") as f:
                summary = json.load(f)
            for method in METHODS:
                method_summary = summary.get(method, {})
                runs = method_summary.get("runs", [])
                if not runs:
                    continue
                raw[ds_key][method]["auroc"].append(np.mean([r["auroc"] for r in runs]))
                raw[ds_key][method]["f1"].append(np.mean([r["f1_exact"] for r in runs]))
                raw[ds_key][method]["pred_loss"].append(
                    np.mean([r.get("pkd_prediction_loss", np.nan) for r in runs])
                )
                raw[ds_key][method]["path_f1"].append(
                    method_summary.get("path_f1_mean", np.nan)
                )

    stats = {}
    for ds_key in DATASET_KEYS:
        stats[ds_key] = {}
        for method in METHODS:
            stats[ds_key][method] = {}
            for metric, values in raw[ds_key][method].items():
                arr = np.asarray(values, dtype=float)
                stats[ds_key][method][metric + "_mean"] = float(np.nanmean(arr))
                stats[ds_key][method][metric + "_std"] = float(np.nanstd(arr))
    return stats


def plot_metric_panel(ax, stats, metric, ylabel, ylim=None, log_scale=False):
    x = np.arange(len(DATASET_KEYS))
    offsets = np.linspace(-0.24, 0.24, len(METHODS))

    for offset, method in zip(offsets, METHODS):
        style = METHOD_STYLE[method]
        vals = np.array([stats[ds][method][metric + "_mean"] for ds in DATASET_KEYS])
        errs = np.array([stats[ds][method][metric + "_std"] for ds in DATASET_KEYS])
        xpos = x + offset
        ax.plot(xpos, vals, color=style["color"], lw=0.8, alpha=0.55, zorder=2)
        ax.errorbar(
            xpos,
            vals,
            yerr=errs,
            fmt=style["marker"],
            markersize=4.2,
            color=style["color"],
            markerfacecolor=style["color"],
            markeredgecolor="white",
            markeredgewidth=0.5,
            elinewidth=0.8,
            capsize=2,
            linestyle="none",
            zorder=3,
        )

    ax.set_xlim(-0.55, 1.55)
    ax.set_xticks(x)
    ax.set_xticklabels(DATASETS)
    ax.set_ylabel(ylabel)
    if ylim:
        ax.set_ylim(*ylim)
    if log_scale:
        ax.set_yscale("log")
        ax.set_ylim(0.018, 1.35)
        ax.set_yticks([0.03, 0.1, 0.3, 1.0])
        ax.set_yticklabels(["0.03", "0.1", "0.3", "1.0"])
    ax.grid(axis="y", color="#D4D4D4", lw=0.45, alpha=0.65)
    ax.tick_params(axis="both", length=3, width=0.8)


def figure_root_cause_rework(stats):
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 3.9))
    plt.subplots_adjust(left=0.075, right=0.99, bottom=0.12, top=0.84, wspace=0.34, hspace=0.47)

    panel_specs = [
        ("auroc", "AUROC", (0, 1.08), False, "a", "Graph ranking"),
        ("f1", "F1 (exact top-k)", (0, 1.08), False, "b", "Exact edge recovery"),
        ("path_f1", "Path F1", (0, 0.76), False, "c", "Propagation-path recovery"),
        ("pred_loss", "Prediction loss", None, True, "d", "Prediction objective"),
    ]

    for ax, (metric, ylabel, ylim, log_scale, label, title) in zip(axes.ravel(), panel_specs):
        plot_metric_panel(ax, stats, metric, ylabel, ylim=ylim, log_scale=log_scale)
        ax.set_title(title, fontweight="bold", pad=4)
        panel_label(ax, label)
        if metric == "auroc":
            ax.axhline(0.5, color="#8A8A8A", lw=0.8, ls=":", zorder=1)
            ax.text(1.42, 0.515, "chance", ha="right", va="bottom", fontsize=6, color="#777777")
        if metric == "f1":
            ax.axhline(0.074, color="#8A8A8A", lw=0.8, ls=":", zorder=1)
            ax.text(
                0.98,
                0.08,
                "random top-k\napprox. 0.074",
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=6,
                color="#777777",
            )

    handles = [
        Line2D(
            [0],
            [0],
            marker=METHOD_STYLE[m]["marker"],
            color=METHOD_STYLE[m]["color"],
            markerfacecolor=METHOD_STYLE[m]["color"],
            markeredgecolor="white",
            markeredgewidth=0.5,
            lw=0.9,
            markersize=4.5,
            label=METHOD_STYLE[m]["label"],
        )
        for m in METHODS
    ]
    fig.legend(handles=handles, loc="upper center", ncol=4, bbox_to_anchor=(0.53, 0.985))
    save_figure(fig, "fig3_root_cause_main_v2")
    plt.close(fig)


def load_dynamics():
    base = os.path.join(
        PROJECT_ROOT, "results_kbs", "root_cause_v2", "phase5_1c", "checkpoints"
    )
    files = {
        "linear": "root_cause_linear_data0_JRNGC_train0_dynamics.csv",
        "nonstationary": "root_cause_nonstationary_data0_JRNGC_train0_dynamics.csv",
    }
    dynamics = {}
    for key, filename in files.items():
        rows = []
        with open(os.path.join(base, filename), newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                rows.append({k: float(v) for k, v in row.items()})
        dynamics[key] = rows
    with open(os.path.join(base, "dynamics_summary.json"), encoding="utf-8") as f:
        summary = json.load(f)
    return dynamics, summary


def add_checkpoint_marker(ax, x, y, marker, color, label):
    ax.scatter(
        [x],
        [y],
        marker=marker,
        s=58,
        color=color,
        edgecolor="white",
        linewidth=0.7,
        zorder=5,
        label=label,
    )


def figure_checkpoint_rework(dynamics, summary):
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.85), sharey=True)
    plt.subplots_adjust(left=0.075, right=0.90, bottom=0.19, top=0.80, wspace=0.18)

    random_f1 = 0.074
    cmap = plt.get_cmap("viridis")
    norm = Normalize(vmin=0, vmax=2000)
    line_color = "#7A7A7A"

    entries = [
        ("linear", "root_cause_linear_data0_train0", "Linear", "a"),
        ("nonstationary", "root_cause_nonstationary_data0_train0", "Nonstationary", "b"),
    ]

    scatter_for_colorbar = None
    for ax, (key, summary_key, title, label) in zip(axes, entries):
        rows = dynamics[key]
        iters = np.array([r["iter"] for r in rows])
        pred_loss = np.array([r["pred_loss"] for r in rows])
        f1 = np.array([r["f1_exact"] for r in rows])
        auroc = np.array([r["auroc"] for r in rows])
        sm = summary[summary_key]

        ax.plot(pred_loss, f1, color=line_color, lw=0.9, zorder=1)
        scatter_for_colorbar = ax.scatter(
            pred_loss,
            f1,
            c=iters,
            cmap=cmap,
            norm=norm,
            s=22,
            edgecolor="white",
            linewidth=0.35,
            zorder=3,
        )
        ax.set_xscale("log")
        ax.invert_xaxis()
        ax.set_xlim(1.55, 0.18)
        ax.set_ylim(-0.05, 1.08)
        ax.xaxis.set_major_locator(FixedLocator([1.0, 0.5, 0.2]))
        ax.xaxis.set_major_formatter(FixedFormatter(["1.0", "0.5", "0.2"]))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel("Prediction loss (decreases rightward)")
        ax.grid(axis="y", color="#D4D4D4", lw=0.45, alpha=0.65)
        ax.axhline(random_f1, color="#8A8A8A", lw=0.8, ls=":", zorder=0)

        best_f1 = sm["oracle_best_f1_checkpoint"]
        best_loss = sm["best_pred_loss_checkpoint"]
        add_checkpoint_marker(
            ax,
            best_f1["pred_loss"],
            best_f1["f1"],
            "*",
            "#C84343",
            "best-F1 checkpoint",
        )
        add_checkpoint_marker(
            ax,
            best_loss["pred_loss"],
            best_loss["f1"],
            "X",
            "#222222",
            "lowest-loss checkpoint",
        )

        ax.set_title(
            f"{title}\n" + r"$\rho$(loss,F1) = "
            + f"{sm['spearman_corr']['pred_loss_vs_f1']:+.3f}",
            fontweight="bold",
            pad=4,
        )
        panel_label(ax, label)

    axes[0].set_ylabel("F1 (exact top-k)")
    cbar = fig.colorbar(
        scatter_for_colorbar,
        ax=axes,
        orientation="vertical",
        fraction=0.035,
        pad=0.025,
        aspect=25,
    )
    cbar.set_label("Training iteration", fontsize=7)
    cbar.set_ticks([0, 500, 1000, 1500, 2000])
    cbar.ax.tick_params(labelsize=6.5, length=2)

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="*",
            color="none",
            markerfacecolor="#C84343",
            markeredgecolor="white",
            markersize=8,
            label="best-F1 checkpoint",
        ),
        Line2D(
            [0],
            [0],
            marker="X",
            color="none",
            markerfacecolor="#222222",
            markeredgecolor="white",
            markersize=6.5,
            label="lowest-loss checkpoint",
        ),
        Line2D([0], [0], color="#8A8A8A", lw=0.8, ls=":", label="random top-k F1"),
    ]
    fig.legend(legend_handles, [h.get_label() for h in legend_handles], loc="upper center", ncol=3)
    save_figure(fig, "fig4_checkpoint_dynamics_v2")
    plt.close(fig)


def figure_causaltime_rework():
    ct_medical = summarize_ct_medical()
    datasets = ["CT-medical", "CT-pm25", "CT-traffic"]
    data = {
        "CT-medical": {
            "random_f1": 0.0853,
            "JRNGC": ct_medical["JRNGC"],
            "ISTF-Mamba": ct_medical["ISTF-Mamba"],
            "ISTF-TCN": None,
        },
        "CT-pm25": {
            "random_f1": 0.0622,
            "JRNGC": {"auroc": 0.4469, "f1": 0.0173},
            "ISTF-Mamba": {"auroc": 0.3734, "f1": 0.0472},
            "ISTF-TCN": {"auroc": 0.4256, "f1": 0.0142},
        },
        "CT-traffic": {
            "random_f1": 0.0397,
            "JRNGC": {"auroc": 0.4089, "f1": 0.0000},
            "ISTF-Mamba": {"auroc": 0.4244, "f1": 0.0484},
            "ISTF-TCN": {"auroc": 0.4033, "f1": 0.0000},
        },
    }
    methods = ["JRNGC", "ISTF-Mamba", "ISTF-TCN"]
    offsets = {"JRNGC": -0.18, "ISTF-Mamba": 0.0, "ISTF-TCN": 0.18}
    x = np.arange(len(datasets))

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.65))
    plt.subplots_adjust(left=0.075, right=0.985, bottom=0.22, top=0.78, wspace=0.22)

    for ax, metric, ylabel, label, title in [
        (axes[0], "auroc", "AUROC", "a", "Ranking boundary"),
        (axes[1], "f1", "F1 (exact top-k)", "b", "Top-k edge recovery"),
    ]:
        for method in methods:
            for i, ds in enumerate(datasets):
                entry = data[ds].get(method)
                if entry is None:
                    if ds == "CT-medical" and method == "ISTF-TCN":
                        na_y = 0.333 if metric == "auroc" else 0.006
                        ax.text(
                            i + offsets[method],
                            na_y,
                            "N/A",
                            ha="center",
                            va="bottom",
                            fontsize=6,
                            color=METHOD_STYLE[method]["color"],
                            fontweight="bold",
                        )
                    continue
                style = METHOD_STYLE[method]
                xpos = i + offsets[method]
                yerr = entry.get(f"{metric}_se")
                if yerr is None:
                    ax.scatter(
                        xpos,
                        entry[metric],
                        s=34,
                        marker=style["marker"],
                        color=style["color"],
                        edgecolor="white",
                        linewidth=0.6,
                        zorder=3,
                    )
                else:
                    ax.errorbar(
                        xpos,
                        entry[metric],
                        yerr=yerr,
                        fmt=style["marker"],
                        markersize=5.0,
                        color=style["color"],
                        markerfacecolor=style["color"],
                        markeredgecolor="white",
                        markeredgewidth=0.6,
                        ecolor=style["color"],
                        elinewidth=0.8,
                        capsize=2.0,
                        zorder=3,
                    )

        if metric == "auroc":
            ax.axhline(0.5, color="#8A8A8A", lw=0.8, ls=":", zorder=1)
            ax.text(2.40, 0.506, "chance", ha="right", va="bottom", fontsize=6, color="#777777")
            ax.set_ylim(0.32, 0.54)
            ax.set_yticks([0.35, 0.40, 0.45, 0.50])
        else:
            random_vals = [data[ds]["random_f1"] for ds in datasets]
            ax.scatter(
                x + 0.30,
                random_vals,
                marker="_",
                s=170,
                color=METHOD_STYLE["Random"]["color"],
                linewidth=1.4,
                zorder=3,
            )
            ax.set_ylim(-0.005, 0.15)
            ax.set_yticks([0.00, 0.05, 0.10, 0.15])
            for i, ds in enumerate(datasets):
                val = data[ds]["random_f1"]
                ax.text(i + 0.30, val + 0.004, f"{val:.3f}", ha="center", va="bottom", fontsize=5.8, color="#777777")

        ax.set_xlim(-0.55, 2.55)
        ax.set_xticks(x)
        ax.set_xticklabels(datasets)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontweight="bold", pad=4)
        ax.grid(axis="y", color="#D4D4D4", lw=0.45, alpha=0.65)
        panel_label(ax, label)

    for ax in axes:
        ax.text(
            0.00,
            -0.34,
            "inferential",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=6,
            color="#555555",
        )
        ax.text(
            1.5,
            -0.34,
            "descriptive boundary",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=6,
            color="#555555",
        )

    handles = [
        Line2D(
            [0],
            [0],
            marker=METHOD_STYLE[m]["marker"],
            color=METHOD_STYLE[m]["color"],
            markerfacecolor=METHOD_STYLE[m]["color"],
            markeredgecolor="white",
            markeredgewidth=0.6,
            lw=0.8,
            markersize=4.8,
            label=METHOD_STYLE[m]["label"],
        )
        for m in methods
    ]
    handles.append(
        Line2D(
            [0],
            [0],
            marker="_",
            color=METHOD_STYLE["Random"]["color"],
            lw=0,
            markersize=8,
            markeredgewidth=1.4,
            label=METHOD_STYLE["Random"]["label"],
        )
    )
    handles.append(Line2D([0], [0], color="#8A8A8A", lw=0.8, ls=":", label="AUROC chance"))
    fig.legend(handles=handles, loc="upper center", ncol=5, bbox_to_anchor=(0.54, 0.99))

    save_figure(fig, "fig6_causaltime_boundary_v2")
    plt.close(fig)


def load_negative_controls():
    base = os.path.join(
        PROJECT_ROOT, "results_kbs", "root_cause_v2", "phase5_1c", "negative_controls"
    )
    with open(os.path.join(base, "permuted_gt_null.json"), encoding="utf-8") as f:
        permuted = json.load(f)
    with open(os.path.join(base, "shuffled_data_results.json"), encoding="utf-8") as f:
        shuffled = json.load(f)
    return permuted, shuffled


def figure_negative_controls_rework(permuted, shuffled):
    datasets = [
        ("root_cause_linear", "Linear"),
        ("root_cause_nonstationary", "Nonstationary"),
    ]
    x = np.arange(len(datasets))
    ema_color = METHOD_STYLE["EMA-JRNGC"]["color"]
    shuffle_color = "#B8B8B8"
    null_a = "#777777"
    null_b = "#AAAAAA"

    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.55))
    plt.subplots_adjust(left=0.075, right=0.99, bottom=0.23, top=0.80, wspace=0.34)

    # Panel a: actual EMA F1 compared with p99 null thresholds.
    ax = axes[0]
    actual = []
    p99_random = []
    p99_perm = []
    for key, _ in datasets:
        entry = permuted[key]["0"]["EMA-JRNGC"]["0"]
        actual.append(entry["actual_gt"]["f1"])
        p99_random.append(entry["random_edge_null"]["f1"]["p99"])
        p99_perm.append(entry["node_label_permutation_null"]["f1"]["p99"])
    max_null = np.maximum(p99_random, p99_perm)
    ax.vlines(x, max_null, actual, color="#D0D0D0", lw=1.0, zorder=1)
    ax.scatter(x - 0.08, p99_random, s=28, marker="o", color=null_a, edgecolor="white", linewidth=0.5, label="random-edge p99", zorder=3)
    ax.scatter(x + 0.08, p99_perm, s=30, marker="s", color=null_b, edgecolor="white", linewidth=0.5, label="node-label p99", zorder=3)
    ax.scatter(x, actual, s=38, marker="D", color=ema_color, edgecolor="white", linewidth=0.6, label="actual EMA F1", zorder=4)
    for xi, val in zip(x, max_null):
        ax.text(xi, val + 0.045, f"p99 <= {val:.3f}", ha="center", va="bottom", fontsize=5.8, color="#666666")
    ax.set_ylabel("F1")
    ax.set_title("Null-test margin", fontweight="bold", pad=4)
    ax.set_ylim(0, 1.08)
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in datasets])
    ax.grid(axis="y", color="#D4D4D4", lw=0.45, alpha=0.65)
    panel_label(ax, "a")

    # Panel b: shuffled-data F1 collapse.
    ax = axes[1]
    original_f1 = np.ones(len(datasets))
    shuffled_f1 = np.array([shuffled[key]["f1"] for key, _ in datasets])
    random_f1 = np.array([shuffled[key]["random_baseline_f1"] for key, _ in datasets])
    ax.plot(x, original_f1, color=ema_color, lw=0.8, alpha=0.55)
    ax.scatter(x, original_f1, s=36, marker="D", color=ema_color, edgecolor="white", linewidth=0.6, label="original", zorder=4)
    ax.plot(x, shuffled_f1, color=shuffle_color, lw=0.8, alpha=0.75)
    ax.scatter(x, shuffled_f1, s=34, marker="o", color=shuffle_color, edgecolor="white", linewidth=0.6, label="time-shuffled", zorder=4)
    ax.scatter(x + 0.16, random_f1, marker="_", s=150, color="#888888", linewidth=1.3, label="random top-k", zorder=3)
    ax.set_ylabel("F1")
    ax.set_title("Time-shuffle control", fontweight="bold", pad=4)
    ax.set_ylim(0, 1.08)
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in datasets])
    ax.grid(axis="y", color="#D4D4D4", lw=0.45, alpha=0.65)
    panel_label(ax, "b")

    # Panel c: shuffled-data AUROC drops near chance.
    ax = axes[2]
    original_auroc = np.ones(len(datasets))
    shuffled_auroc = np.array([shuffled[key]["auroc"] for key, _ in datasets])
    ax.axhline(0.5, color="#8A8A8A", lw=0.8, ls=":", zorder=1)
    ax.text(0.98, 0.12, "chance", transform=ax.transAxes, ha="right", va="bottom", fontsize=6, color="#777777")
    ax.plot(x, original_auroc, color=ema_color, lw=0.8, alpha=0.55)
    ax.scatter(x, original_auroc, s=36, marker="D", color=ema_color, edgecolor="white", linewidth=0.6, zorder=4)
    ax.plot(x, shuffled_auroc, color=shuffle_color, lw=0.8, alpha=0.75)
    ax.scatter(x, shuffled_auroc, s=34, marker="o", color=shuffle_color, edgecolor="white", linewidth=0.6, zorder=4)
    ax.set_ylabel("AUROC")
    ax.set_title("Ranking after shuffle", fontweight="bold", pad=4)
    ax.set_ylim(0.45, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in datasets])
    ax.grid(axis="y", color="#D4D4D4", lw=0.45, alpha=0.65)
    panel_label(ax, "c")

    handles = [
        Line2D([0], [0], marker="D", color=ema_color, markerfacecolor=ema_color, markeredgecolor="white", lw=0.8, markersize=4.8, label="Original/actual"),
        Line2D([0], [0], marker="o", color=shuffle_color, markerfacecolor=shuffle_color, markeredgecolor="white", lw=0.8, markersize=4.8, label="Time-shuffled"),
        Line2D([0], [0], marker="o", color=null_a, lw=0, markersize=4.5, label="Random-edge null p99"),
        Line2D([0], [0], marker="s", color=null_b, lw=0, markersize=4.5, label="Node-label null p99"),
        Line2D([0], [0], marker="_", color="#888888", lw=0, markeredgewidth=1.3, markersize=8, label="Random top-k F1"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=5, bbox_to_anchor=(0.52, 0.995))
    save_figure(fig, "fig7_negative_controls_v2")
    plt.close(fig)


def main():
    stats = load_root_cause_stats()
    figure_root_cause_rework(stats)
    dynamics, summary = load_dynamics()
    figure_checkpoint_rework(dynamics, summary)
    figure_causaltime_rework()
    permuted, shuffled = load_negative_controls()
    figure_negative_controls_rework(permuted, shuffled)


if __name__ == "__main__":
    main()
