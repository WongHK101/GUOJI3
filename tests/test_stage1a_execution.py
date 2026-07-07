"""P0.3c execution-closure tests for Stage 1a runner and aggregator."""

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "experiments") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

from aggregate_stage1a import aggregate, validate_completeness  # noqa: E402
from stage1a_gpu_benchmark import (  # noqa: E402
    atomic_write_json,
    instantiate_paired_method,
    method_cfg,
    predictor_state_dict,
    status_matches_complete,
)


CONFIG_PATH = PROJECT_ROOT / "configs" / "stage1a_frozen_config.json"
SMOKE_CONFIG_PATH = PROJECT_ROOT / "configs" / "stage1a_smoke_config.json"
GPU_SMOKE_CONFIG_PATH = PROJECT_ROOT / "configs" / "stage1a_gpu_infrastructure_smoke_config.json"
SYNTHETIC_CONFIG_HASH = "synthetic-stage1a-config"


def _config():
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return json.load(f)


def _semantic(method, passed=True):
    if method == "baseline":
        return {"passed": True, "track_role": "baseline_no_filter_semantic_gate"}
    if method == "fixed_ema":
        return {"passed": passed, "track_role": "full_H_reference", "metric_track_required": "full_H", "failures": [] if passed else ["ema omitted mass"]}
    return {"passed": passed, "track_role": "nominal_lag_candidate", "failures": [] if passed else ["semantic failure"]}


def _row(cell, method, data_seed, train_seed, auroc, auprc=0.5, mcc=0.2, semantic_passed=True):
    return {
        "cell": cell,
        "method": method,
        "data_seed": data_seed,
        "train_seed": train_seed,
        "status": "complete",
        "formal_result": True,
        "config_sha256": SYNTHETIC_CONFIG_HASH,
        "no_nan_inf": True,
        "metric_track": "full_H" if method == "fixed_ema" else "nominal",
        "semantic_audit": _semantic(method, semantic_passed),
        "metrics": {
            "auroc": auroc,
            "auprc": auprc,
            "f1_exact_topk": 0.3,
            "shd_exact_topk": 10,
            "nshd_exact_topk": 1.0,
            "mcc_exact_topk": mcc,
        },
    }


def _rows(scenario):
    cells = ["Stat+Linear", "Stat+Nonlinear", "NS+Linear", "NS+Nonlinear"]
    rows = []
    for cell in cells:
        for data_seed in [1, 2, 3]:
            for train_seed in [0, 1]:
                base = 0.60
                cp_delta = 0.04 if cell in ["NS+Linear", "NS+Nonlinear"] else 0.01
                if scenario == "cp_baseline_fail":
                    cp_delta = 0.04 if cell == "NS+Linear" else 0.01
                cp = base + cp_delta
                fir = cp - (0.03 if cell == "NS+Linear" else 0.00)
                if scenario == "fir_matches_cp":
                    fir = cp
                ema = cp - 0.01
                if scenario == "ema_dominates":
                    ema = cp + (0.03 if cell in ["Stat+Linear", "NS+Linear"] else 0.005)
                rows.extend([
                    _row(cell, "baseline", data_seed, train_seed, base, auprc=0.50, mcc=0.20),
                    _row(cell, "cp_depthwise", data_seed, train_seed, cp, auprc=0.51, mcc=0.21),
                    _row(cell, "fixed_fir3", data_seed, train_seed, fir, auprc=0.51, mcc=0.21),
                    _row(cell, "fixed_ema", data_seed, train_seed, ema, auprc=0.51, mcc=0.21),
                ])
    return rows


def test_aggregate_go_branch_passes():
    out = aggregate(_rows("pass"), _config(), expected_config_hash=SYNTHETIC_CONFIG_HASH)
    assert out["go_no_go"]["final_go"], out["go_no_go"]
    assert out["go_no_go"]["cp_vs_baseline"]["passed"]
    assert out["go_no_go"]["fixed_fir3_novelty"]["passed"]
    assert not out["go_no_go"]["ema_reference_dominance"]["no_go_triggered"]


def test_aggregate_cp_vs_baseline_failure_branch():
    out = aggregate(_rows("cp_baseline_fail"), _config(), expected_config_hash=SYNTHETIC_CONFIG_HASH)
    assert not out["go_no_go"]["final_go"]
    assert not out["go_no_go"]["cp_vs_baseline"]["passed"]


def test_aggregate_fir_matching_failure_branch():
    out = aggregate(_rows("fir_matches_cp"), _config(), expected_config_hash=SYNTHETIC_CONFIG_HASH)
    assert not out["go_no_go"]["final_go"]
    assert out["go_no_go"]["cp_vs_baseline"]["passed"]
    assert not out["go_no_go"]["fixed_fir3_novelty"]["passed"]


def test_aggregate_ema_dominance_no_go_branch():
    out = aggregate(_rows("ema_dominates"), _config(), expected_config_hash=SYNTHETIC_CONFIG_HASH)
    assert not out["go_no_go"]["final_go"]
    assert out["go_no_go"]["ema_reference_dominance"]["no_go_triggered"]


def test_semantic_gate_failure_blocks_final_go():
    rows = _rows("pass")
    for row in rows:
        if row["method"] == "cp_depthwise" and row["cell"] == "NS+Linear" and row["data_seed"] == 1 and row["train_seed"] == 0:
            row["semantic_audit"] = _semantic("cp_depthwise", passed=False)
            break
    out = aggregate(rows, _config(), expected_config_hash=SYNTHETIC_CONFIG_HASH)
    assert not out["go_no_go"]["final_go"]
    assert not out["semantic_gate"]["passed"]
    assert out["semantic_gate"]["cp_failed_runs"]


def test_stage1a_plan_only_outputs_100_runs_and_smoke_outputs_4_runs():
    tmp = Path(tempfile.mkdtemp(prefix="stage1a_plan_test_"))
    try:
        formal_out = tmp / "formal"
        proc = subprocess.run(
            [
                sys.executable,
                str(PROJECT_ROOT / "experiments" / "stage1a_gpu_benchmark.py"),
                "--config",
                str(CONFIG_PATH),
                "--output-root",
                str(formal_out),
                "--plan-only",
                "--device",
                "cpu",
            ],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert proc.returncode == 0, proc.stderr
        with open(formal_out / "run_manifest.json", encoding="utf-8") as f:
            manifest = json.load(f)
        assert manifest["run_count"] == 100
        assert len(manifest["runs"]) == 100
        assert sum(1 for r in manifest["runs"] if r["role"] == "limited_ablation") == 4

        smoke_out = tmp / "smoke"
        proc = subprocess.run(
            [
                sys.executable,
                str(PROJECT_ROOT / "experiments" / "stage1a_gpu_benchmark.py"),
                "--config",
                str(SMOKE_CONFIG_PATH),
                "--output-root",
                str(smoke_out),
                "--plan-only",
                "--smoke",
                "--device",
                "cpu",
            ],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert proc.returncode == 0, proc.stderr
        with open(smoke_out / "run_manifest.json", encoding="utf-8") as f:
            manifest = json.load(f)
        assert manifest["run_count"] == 4
        assert all(r["cell"] == "NS+Nonlinear" for r in manifest["runs"])

        gpu_smoke_out = tmp / "gpu_smoke"
        proc = subprocess.run(
            [
                sys.executable,
                str(PROJECT_ROOT / "experiments" / "stage1a_gpu_benchmark.py"),
                "--config",
                str(GPU_SMOKE_CONFIG_PATH),
                "--output-root",
                str(gpu_smoke_out),
                "--plan-only",
                "--smoke",
                "--device",
                "cpu",
            ],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert proc.returncode == 0, proc.stderr
        with open(gpu_smoke_out / "run_manifest.json", encoding="utf-8") as f:
            manifest = json.load(f)
        assert manifest["run_count"] == 5
        assert sum(1 for r in manifest["runs"] if r["role"] == "limited_ablation") == 1
    finally:
        resolved = tmp.resolve()
        if str(resolved).startswith(tempfile.gettempdir()):
            shutil.rmtree(resolved, ignore_errors=True)


def test_paired_predictor_initialization_and_filter_seeds():
    cfg = method_cfg(_config(), "baseline")
    models = {}
    for method in ["baseline", "cp_depthwise", "fixed_fir3", "fixed_ema"]:
        m, seeds = instantiate_paired_method(method, method_cfg(_config(), method), data_seed=1, train_seed=0)
        models[method] = (m, seeds)
    base_state = predictor_state_dict(models["baseline"][0])
    for method in ["cp_depthwise", "fixed_fir3", "fixed_ema"]:
        state = predictor_state_dict(models[method][0])
        assert state.keys() == base_state.keys()
        for key in base_state:
            assert torch_equal(state[key], base_state[key]), key
    other, _ = instantiate_paired_method("baseline", cfg, data_seed=1, train_seed=1)
    assert any(not torch_equal(v, predictor_state_dict(other)[k]) for k, v in base_state.items())
    cp = models["cp_depthwise"][0]
    assert float(cp.filter.conv.weight.detach().abs().max()) == 0.0
    mamba, mamba_seeds = instantiate_paired_method("raw_chain_mamba", method_cfg(_config(), "raw_chain_mamba"), data_seed=1, train_seed=0)
    mamba_state = predictor_state_dict(mamba)
    for key in base_state:
        assert torch_equal(mamba_state[key], base_state[key]), key
    assert mamba_seeds["predictor_seed"] == models["baseline"][1]["predictor_seed"]
    assert mamba_seeds["filter_seed"] != mamba_seeds["predictor_seed"]


def torch_equal(a, b):
    import torch
    return torch.equal(a, b)


def test_completeness_gate_rejects_missing_duplicate_hash_and_track_errors():
    cfg = _config()
    rows = _rows("pass")
    assert validate_completeness(rows, cfg, SYNTHETIC_CONFIG_HASH)["passed"]
    assert len(rows) == 96

    missing_train = rows[:-1]
    assert not validate_completeness(missing_train, cfg, SYNTHETIC_CONFIG_HASH)["passed"]

    missing_data = [r for r in rows if not (r["cell"] == "Stat+Linear" and r["method"] == "baseline" and r["data_seed"] == 3)]
    assert not validate_completeness(missing_data, cfg, SYNTHETIC_CONFIG_HASH)["passed"]

    duplicate = rows + [dict(rows[0])]
    assert not validate_completeness(duplicate, cfg, SYNTHETIC_CONFIG_HASH)["passed"]

    bad_hash = [dict(r) for r in rows]
    bad_hash[0]["config_sha256"] = "bad"
    assert not validate_completeness(bad_hash, cfg, SYNTHETIC_CONFIG_HASH)["passed"]

    bad_track = [dict(r) for r in rows]
    for row in bad_track:
        if row["method"] == "fixed_ema":
            row["metric_track"] = "nominal"
            break
    assert not validate_completeness(bad_track, cfg, SYNTHETIC_CONFIG_HASH)["passed"]


def _run_plan_with_config(path, smoke=False):
    tmp = Path(tempfile.mkdtemp(prefix="stage1a_hash_lock_"))
    try:
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "experiments" / "stage1a_gpu_benchmark.py"),
            "--config",
            str(path),
            "--output-root",
            str(tmp / "out"),
            "--plan-only",
            "--device",
            "cpu",
        ]
        if smoke:
            cmd.append("--smoke")
        return subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=120,
        )
    finally:
        resolved = tmp.resolve()
        if str(resolved).startswith(tempfile.gettempdir()):
            shutil.rmtree(resolved, ignore_errors=True)


def _modified_config(tmp, edits):
    tmp.mkdir(parents=True, exist_ok=True)
    path = tmp / "stage1a_frozen_config.json"
    shutil.copyfile(CONFIG_PATH, path)
    with open(path, encoding="utf-8") as f:
        cfg = json.load(f)
    for dotted, value in edits.items():
        cur = cfg
        parts = dotted.split(".")
        for part in parts[:-1]:
            cur = cur[part]
        cur[parts[-1]] = value
    path.write_text(json.dumps(cfg, indent=2, sort_keys=True), encoding="utf-8")
    return path


def test_frozen_config_sha_lock_rejects_scientific_field_changes():
    tmp = Path(tempfile.mkdtemp(prefix="stage1a_config_lock_"))
    try:
        unchanged = tmp / "unchanged.json"
        shutil.copyfile(CONFIG_PATH, unchanged)
        assert _run_plan_with_config(unchanged).returncode == 0
        edits = [
            {"model.predictor.hidden": 32},
            {"training.identity_lam.cp_depthwise": 0.01},
            {"attribution.horizons.fixed_ema": 32},
            {"methods.formal": ["baseline", "cp_depthwise", "fixed_ema"]},
            {"training.seed_rules.predictor_seed": "changed"},
            {"training.primary_checkpoint": 400},
            {"evaluation.chunk_size": 32},
            {"stage1a_go_no_go_gates.cp_vs_baseline.mean_delta_auprc_min": -0.5},
        ]
        for i, edit in enumerate(edits):
            proc = _run_plan_with_config(_modified_config(tmp / f"edit_{i}", edit))
            assert proc.returncode != 0, edit
            assert "Frozen config SHA mismatch" in (proc.stderr + proc.stdout)
    finally:
        resolved = tmp.resolve()
        if str(resolved).startswith(tempfile.gettempdir()):
            shutil.rmtree(resolved, ignore_errors=True)


def test_atomic_write_and_resume_hash_guard():
    tmp = Path(tempfile.mkdtemp(prefix="stage1a_resume_test_"))
    try:
        run_dir = tmp / "run"
        (run_dir / "checkpoints").mkdir(parents=True)
        (run_dir / "scores").mkdir(parents=True)
        checkpoint_iter = 20
        config_hash = "abc123"
        for rel in [
            "config_sha256.txt",
            "commit_hash.txt",
            "generator_metadata.json",
            "schedule.json",
            "loss_trace.json",
            "metrics.json",
            "diagnostics.json",
            "runtime.json",
        ]:
            p = run_dir / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("{}" if rel.endswith(".json") else "x", encoding="utf-8")
        (run_dir / "config_snapshot.json").write_text("{}", encoding="utf-8")
        (run_dir / "environment.json").write_text("{}", encoding="utf-8")
        (run_dir / "checkpoints" / "iter_0020.pt").write_bytes(b"checkpoint")
        for name in ["raw_chain_j_bar.npy", "score_nominal.npy", "score_full_H.npy"]:
            (run_dir / "scores" / name).write_bytes(b"score")
        atomic_write_json(run_dir / "status.json", {
            "status": "complete",
            "config_sha256": config_hash,
        })
        assert not (run_dir / "status.json.tmp").exists()
        assert status_matches_complete(run_dir, config_hash, checkpoint_iter)
        atomic_write_json(run_dir / "status.json", {
            "status": "complete",
            "config_sha256": "different",
        })
        try:
            status_matches_complete(run_dir, config_hash, checkpoint_iter)
        except RuntimeError:
            pass
        else:
            raise AssertionError("config hash mismatch did not block resume reuse")
    finally:
        resolved = tmp.resolve()
        if str(resolved).startswith(tempfile.gettempdir()):
            shutil.rmtree(resolved, ignore_errors=True)


if __name__ == "__main__":
    tests = [
        obj for name, obj in sorted(globals().items())
        if name.startswith("test_") and callable(obj)
    ]
    for test in tests:
        test()
        print(f"PASS: {test.__name__}")
