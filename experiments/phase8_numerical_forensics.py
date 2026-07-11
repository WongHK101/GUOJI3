"""Non-evidentiary lag-profile forensics for the stopped Phase 8 estimator."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import sys
import time
from pathlib import Path
from typing import Dict, Mapping, Sequence

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
EXPERIMENT_DIR = PROJECT_ROOT / "experiments"
for path in [PROJECT_ROOT, SRC_DIR, EXPERIMENT_DIR]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from phase8_coverage import CoverageAlignedRawChainJRNGC, Phase8ModelConfig, as_raw_bdt  # noqa: E402
from phase8_gpu_runner import generate_var1_data  # noqa: E402


FIXED_LAGS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 384, 499)
STATE_FILES = {
    "iteration_20": ("P8-PRE-005", "checkpoint_iter20.pt"),
    "iteration_100": ("P8-PRE-005", "checkpoint.pt"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stopped-preflight-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_run_artifact(root: Path, record_id: str, name: str) -> Dict[str, object]:
    run_dir = root / "runs" / record_id
    manifest = json.loads((run_dir / "artifact_manifest.json").read_text())
    expected = manifest["files"].get(name)
    path = run_dir / name
    actual = file_sha256(path) if path.is_file() else None
    if expected is None or actual != expected:
        raise RuntimeError(f"Stopped artifact integrity failed for {record_id}/{name}")
    return {"record_id": record_id, "file": name, "sha256": actual}


def model_config(dtype: str) -> Phase8ModelConfig:
    return Phase8ModelConfig(
        d=8,
        lag=1,
        layers=3,
        hidden=32,
        d_cond=4,
        d_state=4,
        d_conv=4,
        jacobian_lam=0.01,
        dtype=dtype,
    )


def group_gradient_diagnostics(
    scalar: torch.Tensor,
    model: torch.nn.Module,
) -> Dict[str, object]:
    named = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    gradients = torch.autograd.grad(
        scalar,
        tuple(parameter for _, parameter in named),
        retain_graph=False,
        allow_unused=True,
    )
    pieces = {"predictor": [], "preprocessor": []}
    finite = {"predictor": True, "preprocessor": True}
    for (name, _), gradient in zip(named, gradients):
        if gradient is None:
            continue
        group = "preprocessor" if name.startswith("preprocessor.") else "predictor"
        finite[group] = finite[group] and bool(torch.isfinite(gradient).all())
        pieces[group].append(gradient.detach().reshape(-1))
    output: Dict[str, object] = {}
    for group in ["predictor", "preprocessor"]:
        vector = torch.cat(pieces[group]) if pieces[group] else torch.zeros(1, device=scalar.device, dtype=scalar.dtype)
        output[f"{group}_gradient_native_l2"] = float(torch.linalg.vector_norm(vector))
        output[f"{group}_gradient_float64_accumulated_l2"] = float(torch.linalg.vector_norm(vector.to(torch.float64)))
        output[f"{group}_gradient_max_abs"] = float(torch.max(torch.abs(vector)))
        output[f"{group}_gradient_all_exact_zero"] = bool(torch.count_nonzero(vector) == 0)
        output[f"{group}_gradient_finite"] = finite[group]
    return output


def scalar_status(value: float, dtype: torch.dtype) -> str:
    if value == 0.0:
        return "exact_zero"
    tiny = torch.finfo(dtype).tiny
    if 0 < abs(value) < tiny:
        return "subnormal"
    return "normal"


def profile_lag(
    model: CoverageAlignedRawChainJRNGC,
    x: np.ndarray,
    *,
    raw_lag: int,
    dtype: torch.dtype,
    device: torch.device,
) -> Dict[str, object]:
    target_u = x.shape[1] - 1
    raw = as_raw_bdt(x, device=device, dtype=dtype, require_grad=True)
    prediction = model.predict_from_raw(raw, [target_u])[0]
    block_sum = torch.zeros((), device=device, dtype=dtype)
    for output in range(model.d):
        gradient = torch.autograd.grad(
            prediction[output],
            raw,
            create_graph=True,
            retain_graph=True,
            allow_unused=False,
        )[0]
        block_sum = block_sum + torch.sum(torch.abs(gradient[0, :, target_u - raw_lag]))
    normalized = block_sum / (model.d * model.d * model.lag)
    diagnostics = group_gradient_diagnostics(normalized, model)
    value = float(normalized.detach())
    return {
        "raw_lag": raw_lag,
        "target_u": target_u,
        "eligible_window_count": x.shape[1] - max(model.lag, raw_lag),
        "dtype": str(dtype).replace("torch.", ""),
        "blockwise_absolute_total_jacobian_sum": float(block_sum.detach()),
        "normalized_blockwise_absolute_total_jacobian": value,
        "value_status": scalar_status(value, dtype),
        **diagnostics,
    }


def classification(
    state_name: str,
    lag: int,
    row32: Mapping[str, object],
    row64: Mapping[str, object] | None,
    lag1_32: float,
) -> str:
    value32 = float(row32["normalized_blockwise_absolute_total_jacobian"])
    value64 = None if row64 is None else float(row64["normalized_blockwise_absolute_total_jacobian"])
    if state_name == "initialization" and lag > 1 and value32 == 0 and value64 == 0:
        return "structural_initialization_zero"
    if value32 == 0 and value64 is not None and value64 > 0:
        return "float32_underflow"
    if row32["value_status"] == "subnormal":
        return "float32_underflow"
    if value32 > 0 and value32 <= max(abs(lag1_32), 1e-300) * 1e-12:
        return "numerically_nonzero_but_negligible"
    if value32 == 0:
        return "unresolved"
    return "not_zero"


def main() -> int:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA forensics requested but unavailable")
    if device.type == "cuda" and os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("CUBLAS_WORKSPACE_CONFIG must be :4096:8")
    torch.use_deterministic_algorithms(True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    x, _, _, data_metadata = generate_var1_data(
        8,
        500,
        12001,
        numpy_seed_offset=1,
        generator_name="legacy_full_aux_penalty_var1",
    )
    torch.manual_seed(22001)
    initial_model = CoverageAlignedRawChainJRNGC(model_config("float32"))
    states = {
        "initialization": {name: value.detach().cpu() for name, value in initial_model.state_dict().items()},
    }
    artifact_verification = []
    for state_name, (record_id, filename) in STATE_FILES.items():
        artifact_verification.append(verify_run_artifact(args.stopped_preflight_root, record_id, filename))
        checkpoint = torch.load(
            args.stopped_preflight_root / "runs" / record_id / filename,
            map_location="cpu",
            weights_only=False,
        )
        states[state_name] = checkpoint["model_state_dict"]

    projection_names = {
        "preprocessor.mamba.out_proj.weight",
        "preprocessor.mamba.ssm.out_proj.weight",
    }
    present_projection_names = projection_names.intersection(states["initialization"])
    zero_initialized_projection = (
        present_projection_names == projection_names
        and all(
            bool(torch.count_nonzero(states["initialization"][name]) == 0)
            for name in projection_names
        )
    )
    if not zero_initialized_projection:
        raise RuntimeError("Expected temporal mixing projections are not present and exactly zero at initialization")
    results = []
    float64_errors = []
    for state_name, state in states.items():
        by_dtype: Dict[str, Dict[int, Dict[str, object]]] = {}
        for dtype_name, dtype in [("float32", torch.float32), ("float64", torch.float64)]:
            try:
                model = CoverageAlignedRawChainJRNGC(model_config(dtype_name))
                model.load_state_dict(state)
                model.to(device=device, dtype=dtype).eval()
                by_dtype[dtype_name] = {
                    lag: profile_lag(model, x, raw_lag=lag, dtype=dtype, device=device)
                    for lag in FIXED_LAGS
                }
            except Exception as exc:
                if dtype_name == "float32":
                    raise
                float64_errors.append({"state": state_name, "error": str(exc)})
                by_dtype[dtype_name] = {}
        lag1_32 = float(by_dtype["float32"][1]["normalized_blockwise_absolute_total_jacobian"])
        for lag in FIXED_LAGS:
            row32 = by_dtype["float32"][lag]
            row64 = by_dtype["float64"].get(lag)
            results.append({
                "state": state_name,
                "raw_lag": lag,
                "float32": row32,
                "float64": row64,
                "forensic_classification": classification(state_name, lag, row32, row64, lag1_32),
            })

    report = {
        "non_evidentiary": True,
        "used_for_horizon_or_lambda_selection": False,
        "fixed_lags": list(FIXED_LAGS),
        "states": list(states),
        "data_metadata": data_metadata,
        "artifact_verification": artifact_verification,
        "initial_temporal_projection_zero_initialized": zero_initialized_projection,
        "float64_supported_for_all_states": not float64_errors,
        "float64_errors": float64_errors,
        "classification_rule": {
            "negligible_relative_to_same_state_lag1": 1e-12,
            "allowed_labels": [
                "structural_initialization_zero",
                "float32_underflow",
                "numerically_nonzero_but_negligible",
                "unresolved",
                "not_zero",
            ],
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "device": str(device),
        },
        "results": results,
        "completed_at_unix": time.time(),
    }
    json_path = args.output_dir / "float32_float64_lag_profile_forensics.json"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    csv_path = args.output_dir / "float32_float64_lag_profile_forensics.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "state", "raw_lag", "forensic_classification",
            "float32_value", "float32_status", "float64_value", "float64_status",
            "float32_predictor_gradient_l2", "float32_preprocessor_gradient_l2",
            "float64_predictor_gradient_l2", "float64_preprocessor_gradient_l2",
            "eligible_window_count",
        ])
        writer.writeheader()
        for row in results:
            row32 = row["float32"]
            row64 = row["float64"] or {}
            writer.writerow({
                "state": row["state"],
                "raw_lag": row["raw_lag"],
                "forensic_classification": row["forensic_classification"],
                "float32_value": row32["normalized_blockwise_absolute_total_jacobian"],
                "float32_status": row32["value_status"],
                "float64_value": row64.get("normalized_blockwise_absolute_total_jacobian"),
                "float64_status": row64.get("value_status"),
                "float32_predictor_gradient_l2": row32["predictor_gradient_float64_accumulated_l2"],
                "float32_preprocessor_gradient_l2": row32["preprocessor_gradient_float64_accumulated_l2"],
                "float64_predictor_gradient_l2": row64.get("predictor_gradient_float64_accumulated_l2"),
                "float64_preprocessor_gradient_l2": row64.get("preprocessor_gradient_float64_accumulated_l2"),
                "eligible_window_count": row32["eligible_window_count"],
            })
    print(json.dumps({
        "json": str(json_path),
        "csv": str(csv_path),
        "row_count": len(results),
        "float64_supported": not float64_errors,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
