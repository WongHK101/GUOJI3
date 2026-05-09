"""P1 #5: T/d Filter Selection Criterion — design and validation.

Derives a heuristic for choosing Mamba vs TCN filter without ground truth,
validated against all 11 experimental configurations.

Key insight from experiments:
  - T/d < 5: TCN wins (Mamba lacks temporal context to learn selective patterns)
  - T/d ≥ 5, high non-stationarity: Mamba wins (selective mechanism valuable)
  - T/d ≥ 5, low non-stationarity: either works (near baseline)

Run on cloud: /root/autodl-tmp/GUOJI/mamba_enhanced/
"""
import numpy as np
import sys, os, json

sys.path.insert(0, "/root/autodl-tmp/GUOJI/JRNGC")
sys.path.insert(0, "/root/autodl-tmp/GUOJI/mamba_enhanced")

JRNGC_DATA = "/root/autodl-tmp/GUOJI/JRNGC/data"


def compute_nonstationarity_score(x, n_windows=10):
    """Compute non-stationarity score from sliding window variance.

    For each variable, compute variance in each window, then CV of variances.
    High score → variance changes across time → non-stationary.

    Args:
        x: np.ndarray (d, T)
        n_windows: number of windows

    Returns:
        float: mean CV of window variances across variables
    """
    d, T = x.shape
    window_size = max(T // n_windows, 10)
    n_windows = max(T // window_size, 2)

    all_cvs = []
    for i in range(d):
        window_vars = []
        for w in range(n_windows):
            start = w * window_size
            end = min(start + window_size, T)
            if end - start < 5:
                continue
            window_vars.append(np.var(x[i, start:end]))
        if len(window_vars) >= 3 and np.mean(window_vars) > 1e-10:
            cv = np.std(window_vars) / np.mean(window_vars)
            all_cvs.append(cv)

    if not all_cvs:
        return 0.0
    return float(np.mean(all_cvs))


def compute_trend_strength(x):
    """Compute linear trend strength (mean R² across variables)."""
    d, T = x.shape
    t = np.arange(T).astype(np.float64)
    r2s = []
    for i in range(d):
        y = x[i].astype(np.float64)
        # Linear regression
        A = np.stack([np.ones(T), t], axis=1)
        coef, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
        y_pred = A @ coef
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        if ss_tot > 1e-10:
            r2s.append(1 - ss_res / ss_tot)
    return float(np.mean(r2s)) if r2s else 0.0


def recommend_filter(x, verbose=True):
    """Recommend filter type based on T/d ratio and non-stationarity.

    Decision tree:
      1. T/d < 5  → TCN (too few samples for Mamba selective mechanism)
      2. T/d ≥ 5  → check non-stationarity:
           ns_score > 0.5 → Mamba (selective filtering benefits non-stationary)
           ns_score ≤ 0.5 → either (TCN slightly preferred for simplicity)

    Returns:
        dict with recommendation and diagnostic values
    """
    d, T = x.shape[0], x.shape[1]
    t_over_d = T / d if d > 0 else float('inf')

    ns_score = compute_nonstationarity_score(x)
    trend = compute_trend_strength(x)

    # Decision
    if t_over_d < 5:
        rec = "tcn"
        reason = f"T/d={t_over_d:.1f} < 5: insufficient temporal context for selective SSM"
    elif ns_score > 0.5:
        rec = "mamba"
        reason = f"T/d={t_over_d:.1f} ≥ 5, ns={ns_score:.3f} > 0.5: non-stationary, selective filtering beneficial"
    else:
        rec = "either"
        reason = f"T/d={t_over_d:.1f} ≥ 5, ns={ns_score:.3f} ≤ 0.5: stationary or near-stationary, either filter works"

    result = {
        "d": d, "T": T, "T/d": round(t_over_d, 2),
        "ns_score": round(ns_score, 4),
        "trend_strength": round(trend, 4),
        "recommendation": rec,
        "reason": reason,
    }

    if verbose:
        print(f"  d={d}, T={T}, T/d={t_over_d:.1f}, ns={ns_score:.3f}, "
              f"trend={trend:.3f} → {rec.upper()} ({reason.split(':')[0]})")

    return result


# ============================================================
# Validation against all 11 experimental configurations
# ============================================================
def validate_criterion():
    """Compute criterion for all datasets and compare with empirical best."""
    print("=" * 60)
    print("FILTER SELECTION CRITERION VALIDATION")
    print("=" * 60)

    results = []

    # 1. NSVAR d=10 P=7 (5 seeds: use seed 0)
    print("\n--- Synthetic ---")
    for seed in range(5):
        path = f"/root/autodl-tmp/GUOJI/mamba_enhanced/data/nonstationary_var/num_nodes_10/true_lag_7/noise_scale_1/seed_{seed}"
        x = np.load(os.path.join(path, "_x.npy"))
        r = recommend_filter(x, verbose=(seed == 0))
        r["dataset"] = f"NSVAR_d10_seed{seed}"
        results.append(r)

    # 2. Lorenz-96 F=40 (3 seeds)
    for seed in range(3):
        path = f"{JRNGC_DATA}/lorenz/num_nodes_10/F_40/seed_{seed}"
        try:
            x = np.load(os.path.join(path, "_x.npy"))
            r = recommend_filter(x, verbose=(seed == 0))
            r["dataset"] = f"Lorenz96_F40_seed{seed}"
            results.append(r)
        except:
            pass

    # 3. VAR d=50 stationary (3 seeds)
    for seed in range(3):
        path = f"{JRNGC_DATA}/var/num_nodes_50/true_lag_5/noise_scale_1/seed_{seed}"
        try:
            x = np.load(os.path.join(path, "_x.npy"))
            r = recommend_filter(x, verbose=(seed == 0))
            r["dataset"] = f"VAR_d50_stat_seed{seed}"
            results.append(r)
        except:
            pass

    # 4. NSVAR d=50 Plan A
    print("\n--- Plan A ---")
    plana_dir = "/root/autodl-tmp/GUOJI/mamba_enhanced/data/nsvar_plana"
    for seed in range(3):
        try:
            x = np.load(os.path.join(plana_dir, f"seed_{seed}", "_x.npy"))
        except:
            try:
                x = np.load(os.path.join(plana_dir, f"seed{seed}", "_x.npy"))
            except:
                continue
        r = recommend_filter(x, verbose=(seed == 0))
        r["dataset"] = f"NSVAR_PlanA_d50_seed{seed}"
        results.append(r)

    # 5. DREAM3 d=10/50/100
    print("\n--- DREAM3 ---")
    os.chdir("/root/autodl-tmp/GUOJI/JRNGC")
    from tgc.data.dream3 import dream3_trajectories
    for d in [10, 50, 100]:
        for subject in range(3):
            try:
                x, _, gc = dream3_trajectories(d=d, subject=subject)
                if x.ndim == 3:
                    x = x[0]
                r = recommend_filter(x, verbose=(subject == 0))
                r["dataset"] = f"DREAM3_d{d}_s{subject}"
                results.append(r)
            except Exception as e:
                print(f"  SKIP DREAM3 d={d} s={subject}: {e}")

    # 6. fMRI d=15
    print("\n--- fMRI ---")
    import scipy.io
    for subject in range(3):
        try:
            mat = scipy.io.loadmat(f"{JRNGC_DATA}/fMRI_15.mat")
            x = mat.get("X", mat.get("data", None))
            if x is None:
                continue
            if x.ndim == 2:
                x = x.T  # (T, d) → (d, T)
            r = recommend_filter(x, verbose=(subject == 0))
            r["dataset"] = f"fMRI_d15_s{subject}"
            results.append(r)
            break  # same data for all subjects
        except Exception as e:
            print(f"  SKIP fMRI: {e}")

    # 7. CausalTime
    print("\n--- CausalTime ---")
    for ds in ["medical", "pm25", "traffic"]:
        try:
            x = np.load(f"{JRNGC_DATA}/causaltime/{ds}/_x.npy")
            r = recommend_filter(x, verbose=True)
            r["dataset"] = f"CT_{ds}"
            results.append(r)
        except Exception as e:
            print(f"  SKIP CT {ds}: {e}")

    # ---- Compare with empirical best ----
    # Empirical best from experiments (AUROC-based):
    # >0 means Mamba better, <0 means TCN better, ≈0 means tie
    empirical = {
        "NSVAR_d10": ("mamba/tcn", "tie (+1.6%/+1.7%)"),
        "Lorenz96_F40": ("mamba", "+0.2%"),
        "VAR_d50_stat": ("baseline", "tie (-1.8%/-0.8%)"),
        "NSVAR_PlanA_d50": ("baseline", "-1.4%"),
        "fMRI_d15": ("baseline", "-8.2%"),
        "DREAM3_d10": ("tcn", "+8.1% vs +3.3%"),
        "DREAM3_d50": ("tcn", "+6.1% vs +3.2%"),
        "DREAM3_d100": ("baseline", "-2.2%/-0.7%"),
        "CT_medical": ("mamba", "+17.4% vs +1.9%"),
        "CT_pm25": ("mamba", "+8.9% vs -0.9%"),
        "CT_traffic": ("baseline", "-1.9%"),
    }

    # Aggregate by dataset
    print(f"\n{'='*60}")
    print("VALIDATION SUMMARY")
    print(f"{'='*60}")
    print(f"{'Dataset':<20} {'d':>4} {'T':>5} {'T/d':>7} {'ns':>7} {'trend':>7} "
          f"{'Rec':>6} {'Empirical':>10}")
    print(f"{'-'*20} {'-'*4} {'-'*5} {'-'*7} {'-'*7} {'-'*7} {'-'*6} {'-'*10}")

    correct = 0
    total = 0
    for r in results:
        ds_key = r["dataset"].rsplit("_", 1)[0]  # remove seed/subject suffix
        if ds_key.startswith("NSVAR_d10"):
            ds_key = "NSVAR_d10"
        elif ds_key.startswith("Lorenz96"):
            ds_key = "Lorenz96_F40"
        elif ds_key.startswith("VAR_d50"):
            ds_key = "VAR_d50_stat"
        elif ds_key.startswith("NSVAR_PlanA"):
            ds_key = "NSVAR_PlanA_d50"
        elif ds_key.startswith("DREAM3"):
            # Extract d from dataset name
            parts = r["dataset"].split("_")
            d_val = parts[1] if len(parts) > 1 else ""
            ds_key = f"DREAM3_{d_val}"
        elif ds_key.startswith("fMRI"):
            ds_key = "fMRI_d15"
        elif ds_key.startswith("CT"):
            ds_key = r["dataset"]

        emp = empirical.get(ds_key, ("?", "?"))
        print(f"  {ds_key:<20} {r['d']:>4} {r['T']:>5} {r['T/d']:>7.1f} "
              f"{r['ns_score']:>7.3f} {r['trend_strength']:>7.3f} "
              f"{r['recommendation']:>6} {emp[0]:>10}")

    return results


if __name__ == "__main__":
    results = validate_criterion()

    with open("filter_criterion_validation.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to filter_criterion_validation.json")
