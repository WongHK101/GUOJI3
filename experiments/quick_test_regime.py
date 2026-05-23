"""Quick test: D with reduced regime_shift_strength on NS+Nonlinear cell only."""
import numpy as np, sys, os, time, json
import torch
torch.backends.cudnn.enabled = False
_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)
from src.config import resolve_jrngc_path
_jrngc = resolve_jrngc_path()
if _jrngc and _jrngc not in sys.path:
    sys.path.insert(0, _jrngc)
from mamba_jrngc_pilot import BaselineJRNGC, train_model
from tgc.metrics import two_classify_metrics, remove_self_connection
device = torch.device("cuda")

# Shared graph (seed 0)
rng = np.random.RandomState(42)
d, T, lag = 10, 600, 3
gc = np.zeros((d, d, lag), dtype=np.float32)
for i in range(d):
    for j in range(d):
        if i != j and rng.rand() < 0.2:
            gc[i, j, rng.randint(0, lag)] = 1.0

# Base coefficients
A_base = []
for k in range(lag):
    Ak = np.zeros((d, d), dtype=np.float32)
    for i in range(d):
        for j in range(d):
            if gc[i,j,k] > 0:
                Ak[i,j] = 0.40 * rng.uniform(0.3, 1.0) * rng.choice([-1,1])
    A_base.append(Ak)

tests = [
    ("D_reg020", 0.20),
    ("D_reg025", 0.25),
    # ("D_reg030", 0.30),  # known: 0.5378
]

for label, regime in tests:
    # Drift for non-stationary
    drift_scale = regime * 0.40
    Ad = []
    for k in range(lag):
        raw = np.cumsum(rng.randn(T, d, d) * drift_scale / np.sqrt(T), axis=0)
        window = max(5, T // 30)
        smoothed = np.zeros_like(raw)
        for t in range(T):
            lo = max(0, t - window)
            smoothed[t] = raw[lo:t+1].mean(axis=0)
        Ad.append(smoothed)

    # Generate NS+Nonlinear
    x = np.zeros((d, T), dtype=np.float32)
    noise = rng.randn(d, T).astype(np.float32) * 0.15
    for t in range(lag):
        x[:, t] = noise[:, t]
    for t in range(lag, T):
        pred = np.zeros(d, dtype=np.float32)
        for k in range(lag):
            pred += (A_base[k] + Ad[k][t]) @ x[:, t - k - 1]
        s = float(np.std(pred)) + 1e-8
        pred = (1.0 - 0.50) * pred + 0.50 * s * np.tanh(pred / s)
        x[:, t] = pred + noise[:, t]

    x = x.astype(np.float32)

    # Train
    torch.manual_seed(0)
    np.random.seed(0)
    model = BaselineJRNGC(d=d, lag=lag, layers=5, hidden=50, jacobian_lam=0.01).to(device)
    model, loss = train_model(model, x, max_iter=2000, lr=1e-3, verbose=False)
    gc_pred = model.get_gc_matrix(x)

    # Metrics
    gt_sm = remove_self_connection((gc.sum(axis=2) > 0).astype(np.int32))
    pr_sm = remove_self_connection(np.max(np.abs(gc_pred), axis=2).astype(np.float64))
    (f1, _), (acc, _), (auroc, _, _), (auprc, _, _) = two_classify_metrics(pr_sm, gt_sm)
    n_ed = int(gt_sm.sum())
    print(f"{label}: NS+Nonlinear AUROC_sm={auroc:.4f} AUPRC_sm={auprc:.4f} n_edges={n_ed}")
    del model
    torch.cuda.empty_cache()
