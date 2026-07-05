"""Configuration and path resolution for ISTF-Mamba experiments.

All path logic lives here — no hardcoded /root/autodl-tmp/ anywhere else.
"""
import os
import argparse


# ---- Project root ----
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def resolve_data_dir():
    """Resolve data directory from environment or default locations."""
    env = os.environ.get("ISTF_DATA_DIR", "")
    if env and os.path.isdir(env):
        return env
    candidates = [
        os.path.join(PROJECT_ROOT, "data"),
        os.path.join(os.path.dirname(PROJECT_ROOT), "JRNGC", "data"),
    ]
    for c in candidates:
        if os.path.isdir(c):
            return c
    return candidates[0]


def resolve_jrngc_path():
    """Resolve JRNGC source path for sys.path (needed for tgc imports)."""
    env = os.environ.get("JRNGC_PATH", "")
    if env and os.path.isdir(env):
        return env
    candidate = os.path.join(os.path.dirname(PROJECT_ROOT), "JRNGC")
    if os.path.isdir(candidate):
        return candidate
    return ""


def resolve_results_dir():
    """Resolve results output directory."""
    env = os.environ.get("ISTF_RESULTS_DIR", "")
    if env:
        return env
    return os.path.join(PROJECT_ROOT, "results", "raw")


def resolve_device(force_cpu=False):
    """CPU fallback when CUDA is unavailable or --cpu flag is set."""
    import torch
    if force_cpu:
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---- Default hyperparameters ----
DEFAULT_HP = {
    "layers": 5,
    "hidden": 50,
    "jacobian_lam": 0.01,
    "lr": 1e-3,
    "max_iter": 5000,
    "lookback": 10,
    "check_every": 50,
    "weight_decay": 0.0,
    "d_state": 4,
    "ortho_lam": 0.05,
    "residual_scale": 0.1,
}


def base_parser(description=""):
    """Standard argument parser with device, data, output, and HP overrides."""
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--device", type=str, default=None,
                   help="torch device (default: auto-detect CUDA/CPU)")
    p.add_argument("--data-dir", type=str, default=None,
                   help="data root directory")
    p.add_argument("--jrngc-path", type=str, default=None,
                   help="JRNGC source path")
    p.add_argument("--results-dir", type=str, default=None,
                   help="results output directory")
    p.add_argument("--max-iter", type=int, default=5000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--d-state", type=int, default=4,
                   help="Mamba SSM state dimension")
    p.add_argument("--jacobian-lam", type=float, default=0.01)
    p.add_argument("--ortho-lam", type=float, default=0.05)
    p.add_argument("--residual-scale", type=float, default=0.1)
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--cpu", action="store_true", default=False,
                   help="Force CPU even if CUDA is available")
    return p


def add_standard_args(parser):
    """Add standard training flags to an existing ArgumentParser.

    Usage:
        parser = argparse.ArgumentParser(parents=[base_parser()], ...)
        add_standard_args(parser)   # adds model-specific flags
    """
    parser.add_argument("--d", type=int, default=10, help="number of variables")
    parser.add_argument("--T", type=int, default=600, help="time series length")
    parser.add_argument("--lag", type=int, default=3, help="lag order")
    parser.add_argument("--layers", type=int, default=5, help="MLP layers")
    parser.add_argument("--hidden", type=int, default=50, help="hidden dim")
    parser.add_argument("--filter-type", type=str, default="mamba",
                        choices=["mamba", "tcn", "depthwise", "depthwise_gated", "none"],
                        help="temporal filter type")


def setup_env(args=None):
    """Setup environment from parsed args: add JRNGC to path, set data dir, etc."""
    import sys
    jrngc = args.jrngc_path if args and args.jrngc_path else resolve_jrngc_path()
    if jrngc and jrngc not in sys.path:
        sys.path.insert(0, jrngc)
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)
    data_dir = args.data_dir if args and args.data_dir else resolve_data_dir()
    results_dir = args.results_dir if args and args.results_dir else resolve_results_dir()
    return data_dir, results_dir
