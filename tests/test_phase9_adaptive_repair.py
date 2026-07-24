import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from phase9_adaptive_repair import (  # noqa: E402
    ConstrainedAdaptiveFIRFilter,
    ConstrainedAdaptiveFIRJRNGC,
    project_coverage_gradient,
)
from repaired_istf import RepairedISTFConfig, raw_chain_jacobian_penalty  # noqa: E402


def test_adaptive_fir_constraints_and_fixed_fir3_interior_initialization():
    filt = ConstrainedAdaptiveFIRFilter(
        3,
        kernel_size=3,
        gate_max=0.25,
        init_gate=0.1,
        dtype=torch.float64,
    )
    weights = filt.simplex_weights()
    gates = filt.gates()
    assert torch.all(weights >= 0)
    assert torch.allclose(weights.sum(dim=1), torch.ones(3, dtype=torch.float64))
    assert torch.all(gates > 0)
    assert torch.all(gates < 0.25)
    expected = torch.tensor([1.0 - 0.1 + 0.1 / 3, 0.1 / 3, 0.1 / 3], dtype=torch.float64)
    assert torch.allclose(filt.effective_impulse_response()[0], expected, atol=1e-12, rtol=0)


def test_adaptive_fir_is_coordinate_preserving():
    torch.manual_seed(7)
    filt = ConstrainedAdaptiveFIRFilter(4, dtype=torch.float64)
    raw = torch.randn(1, 8, 4, dtype=torch.float64, requires_grad=True)
    output = filt(raw)
    for target in range(4):
        grad = torch.autograd.grad(output[0, -1, target], raw, retain_graph=True)[0]
        off = grad[0, :, [source for source in range(4) if source != target]]
        assert torch.max(torch.abs(off)).item() < 1e-12


def test_adaptive_fir_raw_target_and_second_order_gradients():
    cfg = RepairedISTFConfig(
        d=3,
        lag=2,
        attribution_horizon=4,
        layers=1,
        hidden=8,
        jacobian_lam=0.01,
        identity_lam=0.0,
        dtype="float64",
    )
    torch.manual_seed(11)
    model = ConstrainedAdaptiveFIRJRNGC(cfg)
    x = np.random.default_rng(12).normal(size=(3, 10))
    batch = model.make_histories(x, target_indices=[5, 6], require_grad=False)
    np.testing.assert_allclose(batch["raw_target"].detach().numpy(), x[:, [5, 6]].T)
    penalty = raw_chain_jacobian_penalty(
        model,
        x,
        target_indices=[5, 6],
        output_targets=[0, 1],
        create_graph=True,
    )
    gradients = torch.autograd.grad(
        penalty,
        [model.filter.kernel_logits, model.filter.gate_logits, model.inputgate.weight],
        allow_unused=False,
    )
    for gradient in gradients:
        assert torch.isfinite(gradient).all()
        assert torch.linalg.norm(gradient).item() > 0


def test_projection_removes_conflict_and_caps_norm_deterministically():
    parameter = torch.nn.Parameter(torch.zeros(2, dtype=torch.float64))
    g_pred = [torch.tensor([1.0, 0.0], dtype=torch.float64)]
    g_cov = [torch.tensor([-2.0, 4.0], dtype=torch.float64)]
    combined_a, diag_a = project_coverage_gradient(
        g_pred,
        g_cov,
        [parameter],
        max_coverage_to_prediction_ratio=0.5,
    )
    combined_b, diag_b = project_coverage_gradient(
        g_pred,
        g_cov,
        [parameter],
        max_coverage_to_prediction_ratio=0.5,
    )
    adjusted = combined_a[0] - g_pred[0]
    assert diag_a.conflict_projected
    assert diag_a.norm_capped
    assert torch.dot(g_pred[0], adjusted).item() >= -1e-12
    assert torch.linalg.norm(adjusted).item() <= 0.5 + 1e-12
    assert torch.equal(combined_a[0], combined_b[0])
    assert diag_a == diag_b


def test_projection_keeps_constructive_coverage_direction():
    parameter = torch.nn.Parameter(torch.zeros(2, dtype=torch.float64))
    g_pred = [torch.tensor([1.0, 0.0], dtype=torch.float64)]
    g_cov = [torch.tensor([0.2, 0.1], dtype=torch.float64)]
    combined, diag = project_coverage_gradient(g_pred, g_cov, [parameter])
    assert not diag.conflict_projected
    assert torch.allclose(combined[0], g_pred[0] + g_cov[0])

