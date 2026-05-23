"""Factorial pilot calibration: 3 difficulty settings x 4 cells x 3 seeds.

Tests {Stationary, Non-Stationary} x {Linear, Nonlinear} 2x2 design under a
UNIFIED generator family (same graph per seed, controlled axis variation).

Settings (expert-specified):
  A: coeff=0.25 noise=0.20 regime=0.30 nonlinear=0.50
  B: coeff=0.20 noise=0.30 regime=0.40 nonlinear=0.75
  C: coeff=0.22 noise=0.25 regime=0.60 nonlinear=0.50

Goal: find setting where baseline AUROC falls in 0.75-0.90 range.
Uses max_iter=2000 for pilot efficiency.

Usage:
    # All settings, seeds 0-2
    python experiments/test_factorial_pilot.py --gpu 0

    # Single setting
    python experiments/test_factorial_pilot.py --settings A --seeds 0,1,2 --gpu 0
"""
import torch
import numpy as np
import sys, os, json, time, argparse

torch.backends.cudnn.enabled = False

_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJ_ROOT)
from src.config import resolve_jrngc_path, resolve_results_dir, resolve_device

_jrngc = resolve_jrngc_path()
if _jrngc:
    sys.path.insert(0, _jrngc)

from mamba_jrngc_pilot import (BaselineJRNGC, MambaFilterJRNGC,
                                train_model, compute_metrics_multimode)
from src.factorial_data import (FACTORIAL_SETTINGS, FACTORIAL_CELLS,
                                 generate_all_factorial_cells)

device = resolve_device()
print(f"Device: {device}")


def run_one_cell(x, gc_true, d, lag, seed, model_type, max_iter=2000, lr=1e-3):
    """Run one training. Returns metrics dict."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    if model_type == "baseline":
        model = BaselineJRNGC(
            d=d, lag=lag, layers=5, hidden=50, jacobian_lam=0.01
        ).to(device)
    else:
        model = MambaFilterJRNGC(
            d=d, lag=lag, layers=5, hidden=50,
            jacobian_lam=0.01, d_state=4, ortho_lam=0.05,
            residual_scale=0.1, filter_type="mamba"
        ).to(device)

    t0 = time.time()
    model, loss = train_model(model, x, max_iter=max_iter, lr=lr, verbose=False)
    gc_pred = model.get_gc_matrix(x)
    train_time = time.time() - t0
    multimode = compute_metrics_multimode(gc_true, gc_pred)
    # Flatten: top-level keys are mode names, each holds a metrics dict
    metrics = {
        "lag0": multimode["lag0"],
        "summary_max": multimode["summary_max"],
        "summary_mean": multimode["summary_mean"],
    }
    metrics["train_time"] = train_time
    metrics["train_loss"] = float(loss)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Factorial pilot calibration")
    parser.add_argument("--settings", type=str, default="A,B,C",
                        help="Comma-separated setting names (A,B,C)")
    parser.add_argument("--seeds", type=str, default="0,1,2",
                        help="Comma-separated seeds")
    parser.add_argument("--max-iter", type=int, default=2000,
                        help="Max training iterations")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--d", type=int, default=10)
    parser.add_argument("--T", type=int, default=600)
    parser.add_argument("--lag", type=int, default=3)
    parser.add_argument("--gpu", type=str, default=None,
                        help="GPU device ID (e.g. '0')")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    settings = [s.strip() for s in args.settings.split(",")]
    seeds = [int(s.strip()) for s in args.seeds.split(",")]

    results_dir = resolve_results_dir()
    os.makedirs(results_dir, exist_ok=True)
    out_path = args.output or os.path.join(results_dir, "factorial_pilot_results.json")

    print("=" * 70)
    print("FACTORIAL PILOT CALIBRATION")
    print(f"  Settings: {settings}")
    print(f"  Seeds: {seeds}")
    print(f"  d={args.d}, T={args.T}, lag={args.lag}, max_iter={args.max_iter}")
    print(f"  Output: {out_path}")
    print("=" * 70)

    all_results = {}

    for setting_name in settings:
        params = FACTORIAL_SETTINGS[setting_name]
        print(f"\n{'='*70}")
        print(f"SETTING {setting_name}: {params}")
        print(f"{'='*70}")
        all_results[setting_name] = {"params": params, "seeds": {}}

        for seed in seeds:
            print(f"\n  --- Seed {seed} ---")

            # Generate all 4 cells for this setting+seed
            cells = generate_all_factorial_cells(
                setting=setting_name, d=args.d, T=args.T, lag=args.lag, seed=seed
            )
            all_results[setting_name]["seeds"][str(seed)] = {}

            for cell_name, stationary, linear in FACTORIAL_CELLS:
                x, gc_true = cells[cell_name]
                d, T_actual = x.shape
                n_edges = int(gc_true.sum())
                print(f"    {cell_name}: T={T_actual}, edges={n_edges}")

                cell_result = {}

                # Baseline JRNGC
                t0 = time.time()
                base_m = run_one_cell(x, gc_true, args.d, args.lag, seed, "baseline",
                                      max_iter=args.max_iter, lr=args.lr)
                elapsed = time.time() - t0
                # Report summary_max as primary, lag0 as reference
                sm = base_m["summary_max"]
                l0 = base_m["lag0"]
                print(f"      Baseline: AUROC_sm={sm['auroc']:.4f} AUPRC_sm={sm['auprc']:.4f} "
                      f"SHD_sm={sm['shd']} edges_sm={sm['n_edges_true']}  "
                      f"[lag0: AUROC={l0['auroc']:.4f} edges={l0['n_edges_true']}] "
                      f"time={elapsed:.0f}s")
                cell_result["baseline"] = base_m

                # ISTF-Mamba
                t0 = time.time()
                mamba_m = run_one_cell(x, gc_true, args.d, args.lag, seed, "mamba",
                                       max_iter=args.max_iter, lr=args.lr)
                elapsed = time.time() - t0
                sm = mamba_m["summary_max"]
                l0 = mamba_m["lag0"]
                print(f"      ISTF-Mamba: AUROC_sm={sm['auroc']:.4f} AUPRC_sm={sm['auprc']:.4f} "
                      f"SHD_sm={sm['shd']} edges_sm={sm['n_edges_true']}  "
                      f"[lag0: AUROC={l0['auroc']:.4f} edges={l0['n_edges_true']}] "
                      f"time={elapsed:.0f}s")
                cell_result["mamba"] = mamba_m

                all_results[setting_name]["seeds"][str(seed)][cell_name] = cell_result

    # ---- Summary ----
    print(f"\n{'='*70}")
    print("PILOT SUMMARY")
    print(f"{'='*70}")

    for setting_name in settings:
        print(f"\nSetting {setting_name} ({FACTORIAL_SETTINGS[setting_name]}):")
        # ---- summary_max as primary metric ----
        print(f"  {'Cell':<18} {'Baseline':>10} {'Mamba':>10} {'Δ':>9}  |  {'Baseline':>10} {'Mamba':>10}")
        print(f"  {'':18} {'summary_max AUROC':>33}  |  {'lag0 AUROC':>23}")
        print(f"  {'-'*18} {'-'*10} {'-'*10} {'-'*9}  |  {'-'*10} {'-'*10}")
        for cell_name, _, _ in FACTORIAL_CELLS:
            b_sm, m_sm, b_l0, m_l0 = [], [], [], []
            for seed in seeds:
                sd = all_results[setting_name]["seeds"][str(seed)]
                if cell_name in sd:
                    b_sm.append(sd[cell_name]["baseline"]["summary_max"]["auroc"])
                    m_sm.append(sd[cell_name]["mamba"]["summary_max"]["auroc"])
                    b_l0.append(sd[cell_name]["baseline"]["lag0"]["auroc"])
                    m_l0.append(sd[cell_name]["mamba"]["lag0"]["auroc"])
            if b_sm:
                b_sm_m, b_sm_s = np.mean(b_sm), np.std(b_sm)
                m_sm_m, m_sm_s = np.mean(m_sm), np.std(m_sm)
                b_l0_m, b_l0_s = np.mean(b_l0), np.std(b_l0)
                m_l0_m, m_l0_s = np.mean(m_l0), np.std(m_l0)
                delta = m_sm_m - b_sm_m
                flag = " <<< IN RANGE" if 0.75 <= b_sm_m <= 0.90 else ""
                print(f"  {cell_name:<18} {b_sm_m:>8.4f}±{b_sm_s:.3f} {m_sm_m:>8.4f}±{m_sm_s:.3f} {delta:>+8.4f}  |  "
                      f"{b_l0_m:>8.4f}±{b_l0_s:.3f} {m_l0_m:>8.4f}±{m_l0_s:.3f}{flag}")

    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
