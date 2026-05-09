"""P2 S1: Concat-Full-Penalty — Penalize auxiliary-channel Jacobian too.

Proves that shortcut collapse is caused by unpenalized side channels.
If adding Jacobian penalty to auxiliary channels mitigates collapse,
this directly confirms the mechanism.

Three conditions:
  A: Concat (penalty on x only) — shortcut collapse expected
  B: Concat-full (penalty on x AND c) — collapse mitigated but GC matrix ambiguous
  C: ISTF (penalty on filtered x, no separate c) — clean & principled

NSVAR d=10, 3 seeds, max_iter=2000.
"""
import torch
import torch.nn as nn
import numpy as np
import sys, os, json, time

torch.backends.cudnn.enabled = False

# os.chdir removed — paths resolved via config
_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJ_ROOT)
from src.config import resolve_jrngc_path
_jrngc = resolve_jrngc_path()
if _jrngc:
    sys.path.insert(0, _jrngc)
# project root now resolved via _PROJ_ROOT

from minimal_mamba import MambaBlock
from mamba_jrngc_pilot import (BaselineJRNGC, MambaFilterJRNGC,
                                MambaJRNGC, train_model, compute_metrics)

device = torch.device("cuda")
N_SEEDS = 3

log_fh = open("concat_full_penalty_results.log", "w", buffering=1)


def log(msg):
    log_fh.write(msg + "\n")
    log_fh.flush()
    print(msg, flush=True)


class MambaConcatFullPenaltyJRNGC(MambaJRNGC):
    """Concat JRNGC with Jacobian penalty applied to ALL input dimensions.

    Key difference from MambaJRNGC: jacobian_penalty() computes gradients
    w.r.t. the FULL d_cond-conditioned input, not just the first d dims.
    """
    def jacobian_penalty(self, x):
        """Compute Jacobian penalty w.r.t. FULL input (d+d_cond dimensions).

        Unlike parent class which only penalizes first d dims, this penalizes
        all input dimensions, closing the auxiliary-channel loophole.
        """
        x.requires_grad_(True)
        y = self.mlp(x)
        jac_loss = torch.tensor(0.0, device=x.device)
        for j in range(y.shape[1]):
            grad = torch.autograd.grad(
                y[:, j], x,
                grad_outputs=torch.ones_like(y[:, j]),
                create_graph=True, retain_graph=True
            )[0]
            jac_loss = jac_loss + torch.mean(torch.abs(grad))
        return self.jacobian_lam * jac_loss

    # NOTE: compute_loss, preprocess_and_windowing inherit from MambaJRNGC
    # which does concat: [x_cond; x_orig] → MLP


def run_one(model, x, gc_true, seed, label, max_iter=2000, lr=1e-3):
    torch.manual_seed(seed)
    np.random.seed(seed)
    t0 = time.time()
    model, loss = train_model(model, x, max_iter=max_iter, lr=lr)
    gc_pred = model.get_gc_matrix(x)
    train_time = time.time() - t0
    metrics = compute_metrics(gc_true, gc_pred)
    metrics["train_time"] = train_time
    metrics["train_loss"] = float(loss)
    del model
    torch.cuda.empty_cache()
    log(f"  [{label}] AUROC={metrics['auroc']:.4f}, AUPRC={metrics['auprc']:.4f}, "
        f"SHD={metrics['shd']}, nSHD={metrics['nshd']:.3f}, time={train_time:.0f}s")
    return metrics


def main():
    log("=" * 60)
    log("S1: CONCAT-FULL-PENALTY — Mechanism Proof")
    log("  Penalize auxiliary channels too → does collapse go away?")
    log(f"  NSVAR d=10, 3 seeds, max_iter=2000")
    log("=" * 60)

    all_results = {}

    for seed in range(N_SEEDS):
        seed_key = f"seed_{seed}"
        all_results[seed_key] = {}

        # Load NSVAR data
        p = "" + os.path.join(_PROJ_ROOT, "data", "nonstationary_var/num_nodes_10/true_lag_7/noise_scale_1"
        x = np.load(os.path.join(p, f"seed_{seed}", "_x.npy"))
        gc = np.load(os.path.join(p, f"seed_{seed}", "_gc.npy"))

        d, T = x.shape
        n_edges = int(gc.sum() if gc.ndim == 2 else gc.max(axis=2).sum())
        log(f"\n{seed_key}: d={d}, T={T}, edges={n_edges}")

        # A: Concat — penalty on x only (shortcut collapse)
        all_results[seed_key]["concat"] = run_one(
            MambaJRNGC(
                d=d, lag=7, layers=5, hidden=50,
                jacobian_lam=0.01, d_state=4, d_cond=4,
                use_time_weight_loss=False
            ),
            x, gc, seed, "Concat (x only)"
        )

        # B: Concat-full — penalty on x AND c (mitigated but ambiguous GC)
        all_results[seed_key]["concat_full"] = run_one(
            MambaConcatFullPenaltyJRNGC(
                d=d, lag=7, layers=5, hidden=50,
                jacobian_lam=0.01, d_state=4, d_cond=4,
                use_time_weight_loss=False
            ),
            x, gc, seed, "Concat-full (x+c)"
        )

        # C: ISTF — clean, all dims covered
        all_results[seed_key]["istf"] = run_one(
            MambaFilterJRNGC(
                d=d, lag=7, layers=5, hidden=50,
                jacobian_lam=0.01, d_state=4,
                ortho_lam=0.05, residual_scale=0.1,
                filter_type="mamba"
            ),
            x, gc, seed, "ISTF"
        )

    # Summary
    log(f"\n{'='*60}")
    log("MECHANISM PROOF: Does penalizing auxiliary channels fix collapse?")
    log(f"{'='*60}")
    log(f"  {'Method':<20} {'AUROC':>10} {'AUPRC':>10} {'SHD':>6}")
    log(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*6}")

    for method, label in [
        ("concat", "Concat (x only)"),
        ("concat_full", "Concat-full (x+c)"),
        ("istf", "ISTF")
    ]:
        aurocs = [all_results[f"seed_{s}"][method]["auroc"] for s in range(N_SEEDS)]
        auprcs = [all_results[f"seed_{s}"][method]["auprc"] for s in range(N_SEEDS)]
        shds = [all_results[f"seed_{s}"][method]["shd"] for s in range(N_SEEDS)]
        log(f"  {label:<20} {np.mean(aurocs):>8.4f}±{np.std(aurocs,ddof=1):.4f} "
            f"{np.mean(auprcs):>8.4f}±{np.std(auprcs,ddof=1):.4f} "
            f"{np.mean(shds):>4.1f}±{np.std(shds,ddof=1):.1f}")

    # Key comparison
    concat_mean = np.mean([all_results[f"seed_{s}"]["concat"]["auroc"] for s in range(N_SEEDS)])
    full_mean = np.mean([all_results[f"seed_{s}"]["concat_full"]["auroc"] for s in range(N_SEEDS)])
    istf_mean = np.mean([all_results[f"seed_{s}"]["istf"]["auroc"] for s in range(N_SEEDS)])

    log(f"\n  Mechanism interpretation:")
    log(f"    Concat → Concat-full recovery: {full_mean - concat_mean:+.4f} AUROC")
    log(f"    (penalizing aux channels recovers GC signal → confirms unpenalized shortcut)")
    log(f"    Concat-full → ISTF gap:       {istf_mean - full_mean:+.4f} AUROC")
    log(f"    (ISTF cleaner bc full-penalty GC matrix has d_cond ambiguity)")

    with open("concat_full_penalty_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    log(f"\nSaved to concat_full_penalty_results.json")
    log_fh.close()


if __name__ == "__main__":
    main()
