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

from aggregate_stage1a import aggregate  # noqa: E402
from stage1a_gpu_benchmark import atomic_write_json, status_matches_complete  # noqa: E402


CONFIG_PATH = PROJECT_ROOT / "configs" / "stage1a_frozen_config.json"
SMOKE_CONFIG_PATH = PROJECT_ROOT / "configs" / "stage1a_smoke_config.json"


def _config():
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return json.load(f)


def _row(cell, method, data_seed, train_seed, auroc, auprc=0.5, mcc=0.2):
    return {
        "cell": cell,
        "method": method,
        "data_seed": data_seed,
        "train_seed": train_seed,
        "metric_track": "full_H" if method == "fixed_ema" else "nominal",
        "metrics": {
            "auroc": auroc,
            "auprc": auprc,
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
    out = aggregate(_rows("pass"), _config())
    assert out["go_no_go"]["final_go"], out["go_no_go"]
    assert out["go_no_go"]["cp_vs_baseline"]["passed"]
    assert out["go_no_go"]["fixed_fir3_novelty"]["passed"]
    assert not out["go_no_go"]["ema_reference_dominance"]["no_go_triggered"]


def test_aggregate_cp_vs_baseline_failure_branch():
    out = aggregate(_rows("cp_baseline_fail"), _config())
    assert not out["go_no_go"]["final_go"]
    assert not out["go_no_go"]["cp_vs_baseline"]["passed"]


def test_aggregate_fir_matching_failure_branch():
    out = aggregate(_rows("fir_matches_cp"), _config())
    assert not out["go_no_go"]["final_go"]
    assert out["go_no_go"]["cp_vs_baseline"]["passed"]
    assert not out["go_no_go"]["fixed_fir3_novelty"]["passed"]


def test_aggregate_ema_dominance_no_go_branch():
    out = aggregate(_rows("ema_dominates"), _config())
    assert not out["go_no_go"]["final_go"]
    assert out["go_no_go"]["ema_reference_dominance"]["no_go_triggered"]


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
