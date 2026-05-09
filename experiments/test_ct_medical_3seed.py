"""P1-1: CT_medical ISTF-Mamba 3-seed validation.

The #1 P1 priority: strongest real-world signal (AUROC +17.4%) currently at 1 seed.
Runs ISTF-Mamba on CT_medical d=40 with seeds 0, 1, 2.
Also runs baseline JRNGC on all 3 seeds for paired comparison.

Uses max_iter=2000 (same as existing 1-seed for comparability).
"""
import torch
import numpy as np
import sys, os, json, time, argparse

torch.backends.cudnn.enabled = False

_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJ_ROOT)
from src.config import resolve_jrngc_path, resolve_data_dir, resolve_results_dir, resolve_device

_jrngc = resolve_jrngc_path()
if _jrngc:
    sys.path.insert(0, _jrngc)

from mamba_jrngc_pilot import (BaselineJRNGC, MambaFilterJRNGC,
                                train_model, compute_metrics)

device = resolve_device()

log_fh = open("ct_medical_3seed.log", "w", buffering=1)

def log(msg):
    log_fh.write(msg + "\n")
    log_fh.flush()
    print(msg, flush=True)

def run_one(x, gc_true, d, seed, model_type, max_iter=2000, lr=1e-3):
    """Run one training. model_type: 'baseline' or 'mamba'."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    if model_type == "baseline":
        model = BaselineJRNGC(
            d=d, lag=1, layers=5, hidden=50, jacobian_lam=0.01
        ).to(device)
    else:
        model = MambaFilterJRNGC(
            d=d, lag=1, layers=5, hidden=50,
            jacobian_lam=0.01, d_state=8, ortho_lam=0.05,
            residual_scale=0.1, filter_type="mamba"
        ).to(device)

    t0 = time.time()
    model, loss = train_model(model, x, max_iter=max_iter, lr=lr)
    gc_pred = model.get_gc_matrix(x)
    train_time = time.time() - t0
    metrics = compute_metrics(gc_true, gc_pred)
    metrics["train_time"] = train_time
    metrics["train_loss"] = float(loss)
    del model
    torch.cuda.empty_cache()
    return metrics

def main():
    log("=" * 60)
    log("P1-1: CT_medical ISTF-Mamba 3-SEED VALIDATION")
    log("=" * 60)

    results_dir = resolve_results_dir()
    os.makedirs(results_dir, exist_ok=True)

    data_dir = resolve_data_dir()
    p = os.path.join(data_dir, "causaltime", "medical")
    x = np.load(os.path.join(p, "_x.npy"))
    gc = np.load(os.path.join(p, "_gc.npy"))
    d = x.shape[0]
    log(f"CT_medical: d={d}, T={x.shape[1]}, edges={int(gc.sum())}")

    results = {"config": "CT_medical", "d": d, "T": int(x.shape[1]),
               "edges_true": int(gc.sum()), "seeds": {}}

    for seed in range(3):
        log(f"\n--- Seed {seed} ---")

        log(f"  Baseline JRNGC...")
        base_metrics = run_one(x, gc, d, seed, "baseline")
        log(f"    AUROC={base_metrics['auroc']:.4f}, AUPRC={base_metrics['auprc']:.4f}, "
            f"SHD={base_metrics['shd']}, time={base_metrics['train_time']:.0f}s")

        log(f"  ISTF-Mamba...")
        mamba_metrics = run_one(x, gc, d, seed, "mamba")
        log(f"    AUROC={mamba_metrics['auroc']:.4f}, AUPRC={mamba_metrics['auprc']:.4f}, "
            f"SHD={mamba_metrics['shd']}, time={mamba_metrics['train_time']:.0f}s")

        results["seeds"][f"seed_{seed}"] = {
            "baseline": base_metrics,
            "mamba": mamba_metrics,
        }

    # Summary
    b_aurocs = [results["seeds"][f"seed_{s}"]["baseline"]["auroc"] for s in range(3)]
    m_aurocs = [results["seeds"][f"seed_{s}"]["mamba"]["auroc"] for s in range(3)]
    log(f"\n{'='*60}")
    log(f"SUMMARY: CT_medical 3-seed")
    log(f"  Baseline: {np.mean(b_aurocs):.4f} ± {np.std(b_aurocs):.4f}")
    log(f"  Mamba:    {np.mean(m_aurocs):.4f} ± {np.std(m_aurocs):.4f}")
    log(f"  Δ AUROC:  {(np.mean(m_aurocs) - np.mean(b_aurocs)):.4f}")

    out_path = os.path.join(results_dir, "ct_medical_3seed_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    log(f"\nSaved to {out_path}")
    log_fh.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CT_medical ISTF-Mamba 3-seed validation")
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--results-dir", type=str, default=None)
    args = parser.parse_args()
    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device
        global device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main()
