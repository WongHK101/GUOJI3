"""S3: CT pm25 TCN standalone (run after DREAM3 finishes, on free GPU)."""
import torch, numpy as np, sys, os, json, time
torch.backends.cudnn.enabled = False
os.chdir("/root/autodl-tmp/GUOJI/mamba_enhanced")
sys.path.insert(0, "/root/autodl-tmp/GUOJI/JRNGC")
sys.path.insert(0, "/root/autodl-tmp/GUOJI/mamba_enhanced")
from mamba_jrngc_pilot import MambaFilterJRNGC, train_model, compute_metrics
device = torch.device("cuda")

log_fh = open("pm25_tcn_standalone.log", "w", buffering=1)
def log(msg):
    log_fh.write(msg + "\n"); log_fh.flush(); print(msg, flush=True)

log("=" * 60)
log("S3: CT pm25 TCN STANDALONE")
log("=" * 60)

p = "/root/autodl-tmp/GUOJI/JRNGC/data/causaltime/pm25"
x = np.load(os.path.join(p, "_x.npy"))
gc = np.load(os.path.join(p, "_gc.npy"))
d = x.shape[0]
log(f"x.shape={x.shape}, d={d}")

torch.manual_seed(0)
np.random.seed(0)
model = MambaFilterJRNGC(
    d=d, lag=1, layers=5, hidden=50,
    jacobian_lam=0.01, d_state=4, ortho_lam=0.05,
    residual_scale=0.1, filter_type="tcn"
).to(device)
log(f"Params: {sum(p.numel() for p in model.parameters())}")

t0 = time.time()
log("Training (max_iter=2000)...")
model, loss = train_model(model, x, max_iter=2000, lr=1e-3)
gc_pred = model.get_gc_matrix(x)
train_time = time.time() - t0
metrics = compute_metrics(gc, gc_pred)
metrics["train_time"] = train_time
metrics["train_loss"] = float(loss)
log(f"AUROC={metrics['auroc']:.4f}, AUPRC={metrics['auprc']:.4f}, "
    f"SHD={metrics['shd']}, nSHD={metrics['nshd']:.3f}, time={train_time:.0f}s")

with open("pm25_tcn_standalone_result.json", "w") as f:
    json.dump(metrics, f, indent=2, default=str)
log("Saved to pm25_tcn_standalone_result.json")
log_fh.close()
