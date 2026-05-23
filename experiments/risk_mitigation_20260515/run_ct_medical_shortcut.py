"""P3: CT_medical Real Concat Shortcut Control.

Checks whether the concat shortcut mechanism (proven in controlled VAR setting)
also manifests on CT_medical — the only inferential dataset with positive ISTF result.

Compares: JRNGC baseline, Concat x-only, Concat full-penalty, ISTF-Mamba.
CT_medical d=40, T=1200, 5 seeds.

Output: risk_mitigation_results/ct_medical_concat_shortcut.{json,csv}
"""
import torch
import numpy as np
import sys, os, json, csv, time
from collections import defaultdict

torch.backends.cudnn.enabled = False

_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _PROJ_ROOT)
sys.path.insert(0, os.path.join(_PROJ_ROOT, "src"))

from src.config import resolve_data_dir
from src.minimal_mamba import MambaBlock
from src.mamba_jrngc_pilot import (
    BaselineJRNGC, MambaJRNGC, MambaFilterJRNGC,
    train_model, compute_metrics
)
# Import fixed full-penalty class from P0 script
from experiments.risk_mitigation_20260515.run_full_aux_penalty import MambaConcatFullPenaltyJRNGC

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT_DIR = os.path.join(_PROJ_ROOT, "risk_mitigation_results")
os.makedirs(OUT_DIR, exist_ok=True)

N_SEEDS = 5
MAX_ITER = 2000  # matches standard CT_medical config
LR = 1e-3
LAMBDA = 0.01
LAG = 1          # matches standard CT_medical config (not 10)
D_STATE = 8      # matches standard CT_medical config (not 4)
D_COND = 8
HIDDEN = 50
LAYERS = 5       # matches standard JRNGC config


def log(msg):
    print(msg, flush=True)


def load_ct_medical(seed):
    """Load CT_medical data (single dataset, seed only affects model init)."""
    # Try standard causaltime path first
    data_dir = resolve_data_dir()
    candidates = [
        os.path.join(data_dir, "causaltime", "medical", "_x.npy"),
        os.path.join(_PROJ_ROOT, "data", "causaltime", "medical", "_x.npy"),
        os.path.join(os.path.dirname(_PROJ_ROOT), "JRNGC", "data", "causaltime", "medical", "_x.npy"),
    ]
    x_path = None
    for p in candidates:
        if os.path.exists(p):
            x_path = p
            break
    if x_path is None:
        raise FileNotFoundError(f"CT_medical data not found. Tried: {candidates}")
    gc_path = os.path.join(os.path.dirname(x_path), "_gc.npy")
    x = np.load(x_path)
    gc = np.load(gc_path)
    if gc.ndim == 3:
        gc = gc.max(axis=2)
    return x, gc


def run_one(model, x, gc, seed, label, max_iter=MAX_ITER, lr=LR):
    torch.manual_seed(seed)
    np.random.seed(seed)
    t0 = time.time()
    log(f"    [{label}] training...")
    model, loss = train_model(model, x, max_iter=max_iter, lr=lr, verbose=False)
    log(f"    [{label}] train done, loss={loss:.6f}")
    gc_pred = model.get_gc_matrix(x)
    train_time = time.time() - t0
    m = compute_metrics(gc, gc_pred)

    jx_norm, jc_norm = None, None
    if hasattr(model, 'get_jacobian_norms'):
        try:
            jx_norm, jc_norm = model.get_jacobian_norms(x)
        except Exception as e:
            log(f"    [{label}] WARNING: jacobian norms failed: {e}")

    log(f"    [{label}] AUROC={m['auroc']:.4f}, SHD={int(m['shd_topk'])}" +
        (f", |Jx|={jx_norm:.4f}, |Jc|={jc_norm:.4f}" if jx_norm is not None else ""))

    result = {
        "auroc": float(m["auroc"]), "auprc": float(m["auprc"]),
        "shd": int(m["shd_topk"]), "f1": float(m.get("f1_topk", 0)),
        "pred_loss": float(loss),
        "jx_norm": jx_norm, "jc_norm": jc_norm,
        "train_time": train_time,
    }
    del model
    torch.cuda.empty_cache()
    return result


def main():
    log("=" * 70)
    log("P3: CT_MEDICAL CONCAT SHORTCUT CONTROL")
    log(f"  d=40, T=1200, lag={LAG}, 5 seeds, max_iter={MAX_ITER}")
    log("  Checking if concat shortcut manifests on real clinical data")
    log("=" * 70)

    d = 40
    all_results = {}

    for seed in range(N_SEEDS):
        seed_key = f"seed_{seed}"
        all_results[seed_key] = {}

        x, gc = load_ct_medical(seed)
        d_actual, T_actual = x.shape
        n_edges = int(gc.sum())
        log(f"\n--- Seed {seed}: d={d_actual}, T={T_actual}, edges={n_edges} ---")

        # 1. JRNGC Baseline
        log("  Baseline...")
        all_results[seed_key]["baseline"] = run_one(
            BaselineJRNGC(d=d_actual, lag=LAG, layers=LAYERS, hidden=HIDDEN,
                          jacobian_lam=LAMBDA).to(device),
            x, gc, seed, "Baseline"
        )

        # 2. Concat x-only
        log("  Concat x-only...")
        all_results[seed_key]["concat_x_only"] = run_one(
            MambaJRNGC(d=d_actual, lag=LAG, layers=LAYERS, hidden=HIDDEN,
                       jacobian_lam=LAMBDA, d_state=D_STATE, d_cond=D_COND,
                       use_time_weight_loss=False).to(device),
            x, gc, seed, "Concat x-only"
        )

        # 3. Concat full-penalty same-lambda
        log("  Concat full-penalty...")
        all_results[seed_key]["concat_full"] = run_one(
            MambaConcatFullPenaltyJRNGC(
                d=d_actual, lag=LAG, layers=LAYERS, hidden=HIDDEN,
                jacobian_lam=LAMBDA, d_state=D_STATE, d_cond=D_COND,
                lam_x=LAMBDA, lam_c=LAMBDA,
                use_time_weight_loss=False).to(device),
            x, gc, seed, "Concat full"
        )

        # 4. ISTF-Mamba
        log("  ISTF-Mamba...")
        all_results[seed_key]["istf"] = run_one(
            MambaFilterJRNGC(d=d_actual, lag=LAG, layers=LAYERS, hidden=HIDDEN,
                             jacobian_lam=LAMBDA, d_state=D_STATE,
                             ortho_lam=0.05, residual_scale=0.1,
                             filter_type="mamba").to(device),
            x, gc, seed, "ISTF-Mamba"
        )

    # Summary
    methods = ["baseline", "concat_x_only", "concat_full", "istf"]
    labels = ["Baseline", "Concat x-only", "Concat full", "ISTF-Mamba"]

    summary = {}
    for method, label in zip(methods, labels):
        vals = defaultdict(list)
        for seed in range(N_SEEDS):
            for k, v in all_results[f"seed_{seed}"][method].items():
                if v is not None:
                    vals[k].append(v)
        summary[method] = {
            "label": label,
            "mean": {k: np.mean(v) for k, v in vals.items()},
            "std": {k: np.std(v, ddof=0) for k, v in vals.items()},
        }

    log(f"\n{'='*80}")
    log(f"{'Method':<20} {'AUROC':>8} {'AUPRC':>8} {'SHD':>5} {'F1':>7} {'|Jx|':>8} {'|Jc|':>8} {'PredLoss':>9}")
    log(f"{'-'*20} {'-'*8} {'-'*8} {'-'*5} {'-'*7} {'-'*8} {'-'*8} {'-'*9}")

    for method, label in zip(methods, labels):
        s = summary[method]
        jx_str = f"{s['mean'].get('jx_norm', float('nan')):.4f}" if s['mean'].get('jx_norm') else "   --"
        jc_str = f"{s['mean'].get('jc_norm', float('nan')):.4f}" if s['mean'].get('jc_norm') else "   --"
        log(f"  {label:<18} {s['mean']['auroc']:7.4f}±{s['std']['auroc']:.4f} "
            f"{s['mean']['auprc']:7.4f}±{s['std']['auprc']:.4f} "
            f"{s['mean']['shd']:4.1f}±{s['std']['shd']:.1f} "
            f"{s['mean']['f1']:6.4f}±{s['std']['f1']:.4f} "
            f"{jx_str}  {jc_str}  "
            f"{s['mean']['pred_loss']:8.6f}")

    # Key: does concat shortcut exist on CT_medical?
    baseline_auroc = summary["baseline"]["mean"]["auroc"]
    concat_auroc = summary["concat_x_only"]["mean"]["auroc"]
    full_auroc = summary["concat_full"]["mean"]["auroc"]
    istf_auroc = summary["istf"]["mean"]["auroc"]

    log(f"\n{'='*70}")
    log("CT_MEDICAL SHORTCUT DIAGNOSIS")
    log(f"  Baseline AUROC:     {baseline_auroc:.4f}")
    log(f"  Concat x-only:      {concat_auroc:.4f}  (does concat degrade?)")
    log(f"  Concat full:        {full_auroc:.4f}  (does full penalty help?)")
    log(f"  ISTF-Mamba:         {istf_auroc:.4f}")
    log(f"  Concateff vs baseline:  {concat_auroc - baseline_auroc:+.4f}")
    log(f"  Full penalty recovery:  {full_auroc - concat_auroc:+.4f}")
    log(f"  ISTF vs baseline:       {istf_auroc - baseline_auroc:+.4f}")

    if concat_auroc < baseline_auroc - 0.01:
        log("  → Concat degrades GC on CT_medical (shortcut confirmed on real data)")
    else:
        log("  → Concat does NOT clearly degrade on CT_medical (shortcut less pronounced)")

    if istf_auroc > full_auroc:
        log("  → ISTF remains best (cleaner than full penalty)")
    else:
        log("  → Full penalty competitive with ISTF on CT_medical")

    # Save
    output = {
        "experiment": "ct_medical_concat_shortcut",
        "setting": f"CT_medical d=40 T=1200 lag={LAG}, 5 seeds, max_iter={MAX_ITER}",
        "per_seed": {
            f"seed_{s}": {k: v for k, v in all_results[f"seed_{s}"].items()}
            for s in range(N_SEEDS)
        },
        "summary": {
            k: {
                "label": v["label"],
                "mean": {mk: float(mv) if isinstance(mv, (np.floating, np.integer))
                         else mv for mk, mv in v["mean"].items()},
                "std": {mk: float(sv) if isinstance(sv, (np.floating, np.integer))
                        else sv for mk, sv in v["std"].items()},
            }
            for k, v in summary.items()
        }
    }

    json_path = os.path.join(OUT_DIR, "ct_medical_concat_shortcut.json")
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    log(f"\nSaved to {json_path}")

    csv_path = os.path.join(OUT_DIR, "ct_medical_concat_shortcut.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "auroc_mean", "auroc_std", "auprc_mean", "auprc_std",
                         "shd_mean", "shd_std", "f1_mean", "f1_std",
                         "jx_norm_mean", "jc_norm_mean", "pred_loss_mean"])
        for method in methods:
            s = summary[method]
            writer.writerow([
                s["label"],
                f"{s['mean']['auroc']:.4f}", f"{s['std']['auroc']:.4f}",
                f"{s['mean']['auprc']:.4f}", f"{s['std']['auprc']:.4f}",
                f"{s['mean']['shd']:.1f}", f"{s['std']['shd']:.1f}",
                f"{s['mean']['f1']:.4f}", f"{s['std']['f1']:.4f}",
                f"{s['mean'].get('jx_norm', 'N/A')}",
                f"{s['mean'].get('jc_norm', 'N/A')}",
                f"{s['mean']['pred_loss']:.6f}",
            ])
    log(f"Saved to {csv_path}")
    return output


if __name__ == "__main__":
    main()
