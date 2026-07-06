"""P0.1 repaired ISTF semantic tests.

These tests use tiny synthetic series only. They verify method definitions,
raw-chain attribution semantics, causal filtering, and deterministic schedules;
they are not performance benchmarks.
"""

import os
import sys

import numpy as np
import torch


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from repaired_istf import (  # noqa: E402
    CPDepthwiseISTFJRNGC,
    FixedEMAJRNGC,
    RawChainMambaISTFJRNGC,
    RawTargetBaselineJRNGC,
    RepairedISTFConfig,
    adjacency_to_edge_set_local,
    aggregate_window_jacobians,
    canonical_baseline_equivalence_audit,
    canonical_baseline_penalty,
    deterministic_sample_indices,
    eligible_target_indices,
    evaluate_repaired_model,
    exact_topk_metrics,
    filtered_coordinate_jacobian_for_windows,
    make_cyclic_schedule,
    raw_chain_jacobian_for_windows,
    raw_chain_jacobian_penalty,
    schedule_hash,
    topk_edges_exact_local,
)
from knowledge_metrics import adjacency_to_edge_set, topk_edges_exact  # noqa: E402


def _x(d=3, t=18, seed=123, dtype=np.float64):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(d, t)).astype(dtype)


def _cfg(d=3, lag=2, h=5, dtype="float64"):
    return RepairedISTFConfig(
        d=d,
        lag=lag,
        attribution_horizon=h,
        layers=1,
        hidden=5,
        dtype=dtype,
        jacobian_lam=0.01,
        identity_lam=0.05,
        residual_gain=0.1,
        depthwise_kernel_size=3,
        d_state=2,
        mamba_expand=2,
        mamba_d_conv=2,
    )


def _models(dtype="float64"):
    cfg = _cfg(dtype=dtype)
    torch.manual_seed(7)
    return [
        CPDepthwiseISTFJRNGC(cfg),
        FixedEMAJRNGC(cfg),
        RawChainMambaISTFJRNGC(cfg),
    ]


def _pred_scalar(model, x_arr, target_u, out_target):
    raw = torch.as_tensor(x_arr, dtype=next(model.parameters()).dtype).unsqueeze(0)
    with torch.no_grad():
        batch = model.make_histories(raw, target_indices=[target_u], require_grad=False)
        pred = model(batch["filtered_history"])
    return float(pred[0, out_target].detach().cpu())


def test_evaluation_aggregation_abs_mean_max():
    jac = torch.tensor(
        [
            [
                [[1.0, -2.0, 0.5], [0.0, -3.0, 4.0]],
                [[-1.0, 1.5, -2.5], [2.0, 0.0, -1.0]],
            ],
            [
                [[3.0, -4.0, 1.5], [2.0, -1.0, 0.0]],
                [[1.0, -0.5, 0.5], [-2.0, 5.0, -6.0]],
            ],
        ]
    )
    agg = aggregate_window_jacobians(jac, lag=2)
    expected_jbar = torch.mean(torch.abs(jac), dim=0).numpy()
    np.testing.assert_allclose(agg["j_bar"], expected_jbar)
    np.testing.assert_allclose(agg["score_nominal"], expected_jbar[:, :, -2:].max(axis=2))
    np.testing.assert_allclose(agg["score_full_H"], expected_jbar.max(axis=2))


def test_topk_orientation_target_source_to_source_target():
    scores = np.zeros((3, 3), dtype=np.float64)
    scores[1, 0] = 10.0
    assert topk_edges_exact_local(scores, 1, exclude_diag=True) == {(0, 1)}
    gc = np.zeros((3, 3), dtype=np.int32)
    gc[1, 0] = 1
    metrics = exact_topk_metrics(scores, gc)
    assert metrics["f1_exact_topk"] == 1.0


def test_canonical_vs_local_exact_topk_equivalence():
    rng = np.random.default_rng(999)
    scores = rng.normal(size=(5, 5))
    scores[2, 1] = scores[3, 0]  # exercise deterministic tie handling
    gc = np.zeros((5, 5), dtype=np.int32)
    gc[1, 0] = 1
    gc[3, 2] = 1
    gc[4, 1] = 1
    for k in [0, 1, 3, 20]:
        assert topk_edges_exact(scores, k, exclude_diag=True) == topk_edges_exact_local(scores, k, exclude_diag=True)
    assert adjacency_to_edge_set(gc, exclude_diag=True) == adjacency_to_edge_set_local(gc, exclude_diag=True)
    canonical_metrics = exact_topk_metrics(scores, gc)
    true_edges = adjacency_to_edge_set_local(gc, exclude_diag=True)
    pred_edges = topk_edges_exact_local(scores, k=len(true_edges), exclude_diag=True)
    tp = len(true_edges & pred_edges)
    precision = tp / max(len(pred_edges), 1)
    recall = tp / max(len(true_edges), 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    assert abs(canonical_metrics["f1_exact_topk"] - f1) < 1e-12


def test_causal_no_future_and_raw_target_isolation():
    x = _x()
    u = 9
    for model in _models():
        model.eval()
        base = model.make_histories(x, target_indices=[u], require_grad=False)
        pert = x.copy()
        pert[:, u:] += 1000.0
        changed = model.make_histories(pert, target_indices=[u], require_grad=False)
        np.testing.assert_allclose(
            base["filtered_history"].detach().cpu().numpy(),
            changed["filtered_history"].detach().cpu().numpy(),
            rtol=0.0,
            atol=1e-10,
        )
        assert not np.allclose(
            base["raw_target"].detach().cpu().numpy(),
            changed["raw_target"].detach().cpu().numpy(),
        )
        pred0 = model(base["filtered_history"]).detach().cpu().numpy()
        pred1 = model(changed["filtered_history"]).detach().cpu().numpy()
        np.testing.assert_allclose(pred0, pred1, rtol=0.0, atol=1e-10)


def test_depthwise_finite_memory_beyond_receptive_field():
    x = _x()
    model = CPDepthwiseISTFJRNGC(_cfg())
    with torch.no_grad():
        model.filter.conv.weight.fill_(0.25)
    u = 11
    base = model.make_histories(x, target_indices=[u], require_grad=False)["filtered_history"]
    pert = x.copy()
    pert[:, 3] += 100.0
    changed = model.make_histories(pert, target_indices=[u], require_grad=False)["filtered_history"]
    np.testing.assert_allclose(
        base.detach().cpu().numpy(),
        changed.detach().cpu().numpy(),
        rtol=0.0,
        atol=1e-10,
    )


def test_finite_difference_raw_chain_depthwise_and_mamba():
    eps = 1e-6
    x = _x(t=16)
    checks = [
        (0, 0, 4),
        (1, 2, 3),
        (2, 1, 1),
        (0, 2, 0),
    ]
    for model in [CPDepthwiseISTFJRNGC(_cfg()), RawChainMambaISTFJRNGC(_cfg())]:
        model.eval()
        u = 10
        jac, _ = raw_chain_jacobian_for_windows(model, x, [u], attribution_horizon=5)
        auto = jac.detach().cpu().numpy()[0]
        for out_target, source, h_pos in checks:
            x_plus = x.copy()
            x_minus = x.copy()
            raw_t = u - model.attribution_horizon + h_pos
            x_plus[source, raw_t] += eps
            x_minus[source, raw_t] -= eps
            fd = (_pred_scalar(model, x_plus, u, out_target) -
                  _pred_scalar(model, x_minus, u, out_target)) / (2.0 * eps)
            au = auto[out_target, source, h_pos]
            tol = 1e-5 + 1e-3 * max(abs(fd), abs(au))
            assert abs(fd - au) <= tol, (type(model).__name__, fd, au, tol)


def test_second_order_gradient_reaches_predictor_and_filter_params():
    x = _x(dtype=np.float32)
    for model in [CPDepthwiseISTFJRNGC(_cfg(dtype="float32")), RawChainMambaISTFJRNGC(_cfg(dtype="float32"))]:
        model.train()
        model.zero_grad(set_to_none=True)
        penalty = raw_chain_jacobian_penalty(
            model,
            x,
            target_indices=[6, 7],
            output_targets=[0, 1],
            create_graph=True,
        )
        penalty.backward()
        predictor_grads = [
            p.grad for name, p in model.named_parameters()
            if ("filter" not in name) and p.grad is not None
        ]
        filter_grads = [
            p.grad for name, p in model.named_parameters()
            if ("filter" in name) and p.grad is not None
        ]
        assert predictor_grads and any(torch.isfinite(g).all() and torch.sum(torch.abs(g)) > 0 for g in predictor_grads)
        assert filter_grads and any(torch.isfinite(g).all() and torch.sum(torch.abs(g)) > 0 for g in filter_grads)


def test_baseline_penalty_equivalence_and_out_of_lag_growth():
    x = _x(dtype=np.float32)
    torch.manual_seed(11)
    b_lag = RawTargetBaselineJRNGC(_cfg(h=2, dtype="float32"))
    torch.manual_seed(11)
    b_h = RawTargetBaselineJRNGC(_cfg(h=5, dtype="float32"))
    b_h.load_state_dict(b_lag.state_dict())
    target_indices = [6, 7]
    output_targets = [0, 1]
    canonical = canonical_baseline_penalty(b_lag, x, target_indices, output_targets)
    repaired_h_lag = raw_chain_jacobian_penalty(b_lag, x, target_indices, output_targets, create_graph=False)
    repaired_h5 = raw_chain_jacobian_penalty(b_h, x, target_indices, output_targets, create_graph=False)
    assert torch.allclose(canonical, repaired_h_lag, atol=1e-7)
    assert torch.allclose(canonical, repaired_h5, atol=1e-7)
    jac, _ = raw_chain_jacobian_for_windows(b_h, x, target_indices, attribution_horizon=5)
    denom = len(target_indices) * len(output_targets) * b_h.d * b_h.lag
    p0 = torch.sum(torch.abs(jac[:, output_targets, :, :])) / denom
    jac[:, output_targets, :, 0] += 0.25
    p1 = torch.sum(torch.abs(jac[:, output_targets, :, :])) / denom
    assert p1 > p0


def test_true_canonical_baseline_equivalence():
    x = _x(d=3, t=12, dtype=np.float32)
    gc = np.zeros((3, 3, 2), dtype=np.float32)
    gc[1, 0, 0] = 1.0
    gc[2, 1, 1] = 1.0
    torch.manual_seed(1234)
    model = RawTargetBaselineJRNGC(_cfg(d=3, lag=2, h=5, dtype="float32"))
    report = canonical_baseline_equivalence_audit(
        model,
        x,
        target_indices=[5, 6, 7],
        output_targets=[0, 1],
        gc_true=gc,
        tolerance=1e-6,
    )
    assert report["passed"], report


def test_raw_chain_coordinate_detach_failure():
    class BrokenDetachedCP(CPDepthwiseISTFJRNGC):
        def filter_sequence(self, raw_bdt):
            return super().filter_sequence(raw_bdt.detach())

    x = _x(dtype=np.float32)
    good = CPDepthwiseISTFJRNGC(_cfg(dtype="float32"))
    jac, _ = raw_chain_jacobian_for_windows(good, x, [6], attribution_horizon=5)
    assert torch.isfinite(jac).all()
    broken = BrokenDetachedCP(_cfg(dtype="float32"))
    try:
        raw_chain_jacobian_for_windows(broken, x, [6], attribution_horizon=5)
    except RuntimeError:
        return
    raise AssertionError("raw-chain Jacobian unexpectedly succeeded after raw input detach")


def test_shared_target_window_schedule_and_indices():
    cfg = _cfg(dtype="float32")
    target_indices = eligible_target_indices(T=18, lag=cfg.lag, attribution_horizon=cfg.attribution_horizon)
    score_windows = deterministic_sample_indices(target_indices, n=4, seed=9103)
    score_windows_repeat = deterministic_sample_indices(target_indices, n=4, seed=9103)
    schedule = make_cyclic_schedule(target_indices, cfg.d, max_iter=5, seed=7101)
    methods = [
        RawTargetBaselineJRNGC(cfg),
        CPDepthwiseISTFJRNGC(cfg),
        RawChainMambaISTFJRNGC(cfg),
        FixedEMAJRNGC(cfg),
    ]
    hashes = []
    for model in methods:
        batch = model.make_histories(_x(dtype=np.float32), target_indices=target_indices)
        np.testing.assert_array_equal(batch["target_indices"].cpu().numpy(), target_indices)
        hashes.append(schedule_hash(schedule))
    assert len(set(hashes)) == 1
    np.testing.assert_array_equal(score_windows, score_windows_repeat)
    assert set(score_windows.tolist()).issubset(set(target_indices.tolist()))


def test_same_seed_determinism_score_max_abs_diff():
    x = _x(dtype=np.float32)
    target_indices = [6, 7]
    torch.manual_seed(77)
    m1 = CPDepthwiseISTFJRNGC(_cfg(dtype="float32"))
    torch.manual_seed(77)
    m2 = CPDepthwiseISTFJRNGC(_cfg(dtype="float32"))
    j1, _ = raw_chain_jacobian_for_windows(m1, x, target_indices, attribution_horizon=5)
    j2, _ = raw_chain_jacobian_for_windows(m2, x, target_indices, attribution_horizon=5)
    s1 = aggregate_window_jacobians(j1, lag=2)["score_nominal"]
    s2 = aggregate_window_jacobians(j2, lag=2)["score_nominal"]
    assert float(np.max(np.abs(s1 - s2))) < 1e-7


def test_evaluate_repaired_model_basic_outputs_are_finite():
    x = _x(d=3, t=20, dtype=np.float32)
    gc = np.zeros((3, 3, 2), dtype=np.float32)
    gc[1, 0, 0] = 1.0
    gc[2, 1, 1] = 1.0
    model = RawTargetBaselineJRNGC(_cfg(dtype="float32"))
    out = evaluate_repaired_model(model, x, gc, target_indices=[6, 7], attribution_horizon=5)
    assert out["score_nominal"].shape == (3, 3)
    assert out["score_full_H"].shape == (3, 3)
    assert np.isfinite(out["metrics_nominal"]["auroc"])
    assert np.isfinite(out["eval_raw_prediction_loss"])


if __name__ == "__main__":
    tests = [
        obj for name, obj in sorted(globals().items())
        if name.startswith("test_") and callable(obj)
    ]
    for test in tests:
        test()
        print(f"PASS: {test.__name__}")
