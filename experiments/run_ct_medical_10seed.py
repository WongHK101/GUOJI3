"""CT_medical 10-seed expansion: baseline + ISTF-Mamba.

Runs seeds 0-9, outputs unified schema to results/raw/ct_medical_10seed.json.
Requires GPU.

Usage:
    python experiments/run_ct_medical_10seed.py
    python experiments/run_ct_medical_10seed.py --seeds 3,4,5,6,7,8,9  # only new seeds
"""

import torch
import numpy as np
import sys, os, json, time, argparse

_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJ_ROOT)
from src.config import resolve_jrngc_path, resolve_data_dir, resolve_device
from src.schema import (make_result_entry, make_collection, save_collection,
                        make_provenance, load_collection, expected_results_dir)

_jrngc = resolve_jrngc_path()
if _jrngc:
    sys.path.insert(0, _jrngc)

from mamba_jrngc_pilot import (BaselineJRNGC, MambaFilterJRNGC,
                                train_model, compute_metrics)

device = resolve_device()
DATA_DIR = resolve_data_dir()
RESULTS_DIR = expected_results_dir()
os.makedirs(RESULTS_DIR, exist_ok=True)


def run_one(x, gc_true, d, seed, model_type, max_iter=2000, lr=1e-3):
    """Run one training. Returns (gc_pred, metrics_dict, train_time)."""
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
    metrics["train_time_s"] = train_time
    metrics["train_loss"] = float(loss)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return gc_pred, metrics


def main():
    parser = argparse.ArgumentParser(description="CT_medical 10-seed expansion")
    parser.add_argument("--seeds", type=str, default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    data_dir = args.data_dir or DATA_DIR
    seeds = [int(s) for s in args.seeds.split(",")]

    # Load data
    p = os.path.join(data_dir, "causaltime", "medical")
    x = np.load(os.path.join(p, "_x.npy"))
    gc = np.load(os.path.join(p, "_gc.npy"))
    d = x.shape[0]
    print(f"CT_medical: d={d}, T={x.shape[1]}, edges={int(gc.sum())}")
    print(f"Seeds: {seeds}")

    config = {"data": "CT_medical", "d": d, "T": int(x.shape[1]),
              "max_iter": args.max_iter, "lr": args.lr,
              "d_state": 8, "layers": 5, "hidden": 50}

    all_entries = []

    for seed in seeds:
        print(f"\n--- Seed {seed} ---")

        print(f"  Baseline...")
        t0 = time.time()
        _, base_metrics = run_one(x, gc, d, seed, "baseline",
                                  max_iter=args.max_iter, lr=args.lr)
        dt = time.time() - t0
        base_metrics["train_time_s"] = dt
        print(f"    AUROC={base_metrics['auroc']:.4f} "
              f"AUPRC={base_metrics['auprc']:.4f} "
              f"SHD={base_metrics.get('shd','?')}")

        all_entries.append(make_result_entry(
            dataset="CT_medical", method="baseline", seed=seed,
            metrics=base_metrics,
            config=config,
            runtime={"train_time_s": dt, "device": str(device)},
            provenance=make_provenance(
                "experiments/run_ct_medical_10seed.py", source="rerun"),
        ))

        print(f"  ISTF-Mamba...")
        t0 = time.time()
        _, mamba_metrics = run_one(x, gc, d, seed, "mamba",
                                   max_iter=args.max_iter, lr=args.lr)
        dt = time.time() - t0
        mamba_metrics["train_time_s"] = dt
        print(f"    AUROC={mamba_metrics['auroc']:.4f} "
              f"AUPRC={mamba_metrics['auprc']:.4f} "
              f"SHD={mamba_metrics.get('shd','?')}")

        all_entries.append(make_result_entry(
            dataset="CT_medical", method="mamba", seed=seed,
            metrics=mamba_metrics,
            config=config,
            runtime={"train_time_s": dt, "device": str(device)},
            provenance=make_provenance(
                "experiments/run_ct_medical_10seed.py", source="rerun"),
        ))

    # Summary
    b_aurocs = [e["metrics"]["auroc"] for e in all_entries
                if e["method"] == "baseline"]
    m_aurocs = [e["metrics"]["auroc"] for e in all_entries
                if e["method"] == "mamba"]
    if b_aurocs and m_aurocs:
        print(f"\nSUMMARY: CT_medical {len(seeds)}-seed")
        print(f"  Baseline: {np.mean(b_aurocs):.4f} +/- {np.std(b_aurocs):.4f}")
        print(f"  Mamba:    {np.mean(m_aurocs):.4f} +/- {np.std(m_aurocs):.4f}")
        print(f"  Delta AUROC: {(np.mean(m_aurocs) - np.mean(b_aurocs)):.4f}")

    # Save
    out_path = args.output or os.path.join(RESULTS_DIR, "ct_medical_10seed.json")
    collection = make_collection(
        all_entries,
        description="CT_medical 10-seed baseline + ISTF-Mamba",
    )
    collection["_audit"] = make_provenance(
        "experiments/run_ct_medical_10seed.py", source="rerun",
        extra={"n_seeds": len(seeds), "seeds": seeds}
    )
    save_collection(collection, out_path)
    print(f"\nSaved {len(all_entries)} entries to {out_path}")


if __name__ == "__main__":
    main()
