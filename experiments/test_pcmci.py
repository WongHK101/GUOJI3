"""PCMCI+ baseline: ParCorr conditional independence test on all 7 datasets."""
import torch, numpy as np, sys, os, time, json
_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJ_ROOT)
from src.config import resolve_jrngc_path
_jrngc = resolve_jrngc_path()
if _jrngc:
    sys.path.insert(0, _jrngc)
# project root now resolved via _PROJ_ROOT
from mamba_jrngc_pilot import compute_metrics
from tigramite import data_processing as pp
from tigramite.independence_tests.parcorr import ParCorr
from tigramite.pcmci import PCMCI

JRNGC_DATA = resolve_data_dir()  # resolved via config
results = {}

def run_pcmci(x, pc_alpha=0.05, tau_max=1):
    """x: (d, T) numpy array. Returns gc_pred: (d, d, 1) continuous score."""
    d, T = x.shape
    # tigramite expects (T, d) DataFrame
    df = pp.DataFrame(x.T)
    parcorr = ParCorr(significance='analytic')
    pcmci = PCMCI(dataframe=df, cond_ind_test=parcorr, verbosity=0)
    res = pcmci.run_pcmci(tau_min=0, tau_max=tau_max, pc_alpha=pc_alpha)
    # p_matrix: (d, d, tau_max+1). Use lag-1 p-values as score.
    p_mat = res['p_matrix'][:, :, 1]  # lag=1 only
    # Convert to continuous score: 1 - p_value (higher = more likely edge)
    gc_pred = (1 - p_mat)[:, :, np.newaxis]
    # Clip to [0,1]
    gc_pred = np.clip(gc_pred, 0, 1)
    return gc_pred

# ====== 1. Non-stationary VAR d=10, P=7 (5 seeds) ======
print("=" * 60)
print("PCMCI+: Non-stationary VAR d=10 P=7 (5 seeds)")
nsvar_aurocs = []
for seed in range(5):
    data_path = os.path.join(_PROJ_ROOT, "data", "nonstationary_var", "num_nodes_10", "true_lag_7", "noise_scale_1")
    x = np.load(os.path.join(data_path, f"seed_{seed}", "_x.npy"))
    gc = np.load(os.path.join(data_path, f"seed_{seed}", "_gc.npy"))
    gc_pred = run_pcmci(x)
    met = compute_metrics(gc, gc_pred)
    nsvar_aurocs.append(met['auroc'])
    print(f"  seed {seed}: AUROC={met['auroc']:.4f} SHD={met['shd']}")
results["nsvar_d10"] = {"auroc_mean": np.mean(nsvar_aurocs), "auroc_std": np.std(nsvar_aurocs),
                         "aurocs": nsvar_aurocs}
print(f"  => PCMCI+: {np.mean(nsvar_aurocs):.4f} ± {np.std(nsvar_aurocs):.4f}")
print(f"  => Our baseline: 0.9296, F_ds8_fix: 0.9457")

# ====== 2. Lorenz-96 F=40 (3 seeds) ======
print("\n" + "=" * 60)
print("PCMCI+: Lorenz-96 F=40 (3 seeds)")
lorenz_aurocs = []
for seed in range(3):
    data_path = f"{JRNGC_DATA}/lorenz/num_nodes_10/F_40/seed_{seed}"
    x = np.load(os.path.join(data_path, "_x.npy"))
    gc = np.load(os.path.join(data_path, "_gc.npy"))
    gc_pred = run_pcmci(x)
    met = compute_metrics(gc, gc_pred)
    lorenz_aurocs.append(met['auroc'])
    print(f"  seed {seed}: AUROC={met['auroc']:.4f} SHD={met['shd']}")
results["lorenz_f40"] = {"auroc_mean": np.mean(lorenz_aurocs), "auroc_std": np.std(lorenz_aurocs),
                          "aurocs": lorenz_aurocs}
print(f"  => PCMCI+: {np.mean(lorenz_aurocs):.4f} ± {np.std(lorenz_aurocs):.4f}")
print(f"  => Our baseline: 0.9350, F_ds8_fix: 0.9374")

# ====== 3. Stationary VAR d=50 (3 seeds) ======
print("\n" + "=" * 60)
print("PCMCI+: Stationary VAR d=50 (3 seeds)")
var50_aurocs = []
for seed in range(3):
    data_path = os.path.join(JRNGC_DATA, "var", "num_nodes_50", "true_lag_5", "noise_scale_1", f"seed_{seed}")
    x = np.load(os.path.join(data_path, "_x.npy"))
    gc = np.load(os.path.join(data_path, "_gc.npy"))
    gc_pred = run_pcmci(x)
    met = compute_metrics(gc, gc_pred)
    var50_aurocs.append(met['auroc'])
    print(f"  seed {seed}: AUROC={met['auroc']:.4f} SHD={met['shd']}")
results["var_d50_stat"] = {"auroc_mean": np.mean(var50_aurocs), "auroc_std": np.std(var50_aurocs),
                            "aurocs": var50_aurocs}
print(f"  => PCMCI+: {np.mean(var50_aurocs):.4f} ± {np.std(var50_aurocs):.4f}")
print(f"  => Our baseline: 0.7145, F_ds8_fix: 0.6963")

# ====== 4. Non-stationary VAR d=50 Plan A (3 seeds) ======
print("\n" + "=" * 60)
print("PCMCI+: NSVAR d=50 Plan A (3 seeds)")
nsvar50_aurocs = []
for seed in [0, 1, 2]:
    data_path = os.path.join(_PROJ_ROOT, "data", "nonstationary_var_planA", "num_nodes_50", "true_lag_14", "noise_scale_1", "seed_{seed}")
    x = np.load(os.path.join(data_path, "_x.npy"))
    gc = np.load(os.path.join(data_path, "_gc.npy"))
    gc_pred = run_pcmci(x)
    met = compute_metrics(gc, gc_pred)
    nsvar50_aurocs.append(met['auroc'])
    print(f"  seed {seed}: AUROC={met['auroc']:.4f} SHD={met['shd']}")
results["nsvar_d50_planA"] = {"auroc_mean": np.mean(nsvar50_aurocs), "auroc_std": np.std(nsvar50_aurocs),
                               "aurocs": nsvar50_aurocs}
print(f"  => PCMCI+: {np.mean(nsvar50_aurocs):.4f} ± {np.std(nsvar50_aurocs):.4f}")
print(f"  => Our baseline: 0.6497 (1 seed), F_ds8_fix: 0.6358")

# ====== 5. fMRI d=15 (3 subjects) ======
print("\n" + "=" * 60)
print("PCMCI+: fMRI d=15 (3 subjects)")
fmri_aurocs = []
for subject in [0, 1, 2]:
    data_path = os.path.join(JRNGC_DATA, "fmri", "num_nodes_15", f"subject_{subject}", "seed_0")
    x = np.load(os.path.join(data_path, "_x.npy"))
    gc = np.load(os.path.join(data_path, "_gc.npy"))
    gc_pred = run_pcmci(x)
    met = compute_metrics(gc, gc_pred)
    fmri_aurocs.append(met['auroc'])
    print(f"  subject {subject}: AUROC={met['auroc']:.4f} SHD={met['shd']}")
results["fmri_d15"] = {"auroc_mean": np.mean(fmri_aurocs), "auroc_std": np.std(fmri_aurocs),
                        "aurocs": fmri_aurocs}
print(f"  => PCMCI+: {np.mean(fmri_aurocs):.4f} ± {np.std(fmri_aurocs):.4f}")
print(f"  => Our baseline: 0.5255, F_ds8_fix: 0.4439")

# ====== 6. DREAM3 (d=10/50/100, 3 subjects each) ======
os.chdir(resolve_jrngc_path() or ".")  # resolved via config
from tgc.data.dream3 import dream3_trajectories

for d in [10, 50, 100]:
    print(f"\n{'='*60}")
    print(f"PCMCI+: DREAM3 d={d}")
    aurocs = []
    for subject in range(3):
        try:
            x, _, gc = dream3_trajectories(d=d, subject=subject)
        except Exception as e:
            print(f"  SKIP ({e})")
            continue
        # x is (N_traj, d, T) — use first trajectory
        if x.ndim == 3:
            x_use = x[0]
        else:
            x_use = x
        gc_pred = run_pcmci(x_use)
        met = compute_metrics(gc, gc_pred)
        aurocs.append(met['auroc'])
        print(f"  d{d}_s{subject}: AUROC={met['auroc']:.4f} SHD={met['shd']}")
    results[f"dream3_d{d}"] = {"auroc_mean": np.mean(aurocs), "auroc_std": np.std(aurocs) if len(aurocs) > 1 else 0,
                                "aurocs": aurocs}
    print(f"  => PCMCI+: {np.mean(aurocs):.4f} ± {np.std(aurocs) if len(aurocs) > 1 else 0:.4f}")
    if d == 10:
        print(f"  => Our baseline: 0.5113, F_ds8_fix: 0.5442")
    elif d == 50:
        print(f"  => Our baseline: 0.4956, F_ds8_fix: 0.5273")
    else:
        print(f"  => Our baseline: 0.5305, F_ds8_fix: 0.5233")

# ====== 7. CausalTime ======
for ds_name in ["traffic", "medical", "pm25"]:
    print(f"\n{'='*60}")
    print(f"PCMCI+: CausalTime {ds_name}")
    data_path = os.path.join(JRNGC_DATA, f"causaltime/{ds_name}")
    x = np.load(os.path.join(data_path, "_x.npy"))
    gc = np.load(os.path.join(data_path, "_gc.npy"))
    gc_pred = run_pcmci(x)
    met = compute_metrics(gc, gc_pred)
    results[f"causaltime_{ds_name}"] = {"auroc": met['auroc'], "shd": met['shd']}
    print(f"  => AUROC={met['auroc']:.4f} SHD={met['shd']}")
    if ds_name == "traffic":
        print(f"  => Our baseline: 0.4084, F_ds8_fix: 0.3889")
    elif ds_name == "medical":
        print(f"  => Our baseline: 0.4766, F_ds8_fix: 0.5596")
    else:
        print(f"  => Our baseline: 0.4288, F_ds8_fix: 0.4668")

# ====== Summary ======
print(f"\n{'='*60}")
print("PCMCI+ FULL SUMMARY")
print(f"{'='*60}")
print(f"{'Dataset':<25} {'PCMCI+ AUROC':<18} {'Our Baseline':<15} {'F_ds8_fix':<15}")
print("-" * 73)
our_baselines = {
    "nsvar_d10": (0.9296, "±0.024"),
    "lorenz_f40": (0.9350, ""),
    "var_d50_stat": (0.7145, ""),
    "nsvar_d50_planA": (0.6497, ""),
    "fmri_d15": (0.5255, ""),
    "dream3_d10": (0.5113, "±0.047"),
    "dream3_d50": (0.4956, "±0.032"),
    "dream3_d100": (0.5305, "±0.028"),
    "causaltime_traffic": (0.4084, ""),
    "causaltime_medical": (0.4766, ""),
    "causaltime_pm25": (0.4288, ""),
}
our_fix = {
    "nsvar_d10": 0.9457,
    "lorenz_f40": 0.9374,
    "var_d50_stat": 0.6963,
    "nsvar_d50_planA": 0.6358,
    "fmri_d15": 0.4439,
    "dream3_d10": 0.5442,
    "dream3_d50": 0.5273,
    "dream3_d100": 0.5233,
    "causaltime_traffic": 0.3889,
    "causaltime_medical": 0.5596,
    "causaltime_pm25": 0.4668,
}
for key, (bl_val, bl_std) in our_baselines.items():
    if "auroc_mean" in results[key]:
        pc_val = f"{results[key]['auroc_mean']:.4f}±{results[key]['auroc_std']:.4f}"
        bl_str = f"{bl_val:.4f}{bl_std}"
    else:
        pc_val = f"{results[key]['auroc']:.4f}"
        bl_str = f"{bl_val:.4f}"
    fx_str = f"{our_fix[key]:.4f}"
    print(f"{key:<25} {pc_val:<18} {bl_str:<15} {fx_str:<15}")

with open("pcmci_results.json", "w") as f:
    json.dump(results, f, indent=2, default=str)
print("\nSaved to pcmci_results.json")
