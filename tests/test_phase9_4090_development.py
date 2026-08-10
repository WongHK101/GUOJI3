from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src", ROOT / "experiments"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from phase9_4090_development import (  # noqa: E402
    generate_phase8_var1,
    make_d2_model,
    model_seed,
    parameter_displacement,
    tensor_state_sha256,
)


def test_all_d2_methods_share_predictor_initialization():
    hashes = []
    for method in (
        "baseline",
        "cp_depthwise",
        "fixed_fir3",
        "adaptive_fir",
        "contextual_fir",
    ):
        _, _, digest = make_d2_model(method, data_seed=0, train_seed=0)
        hashes.append(digest)
    assert len(set(hashes)) == 1


def test_phase8_development_generator_is_deterministic_and_stable():
    first = generate_phase8_var1(d=5, T=30, seed=13001)
    second = generate_phase8_var1(d=5, T=30, seed=13001)
    for left, right in zip(first[:3], second[:3]):
        np.testing.assert_array_equal(left, right)
    assert np.isfinite(first[0]).all()
    assert float(np.max(np.abs(np.linalg.eigvals(first[2])))) <= 0.800001


def test_tensor_state_hash_and_displacement():
    torch.manual_seed(model_seed(0, 0))
    model = torch.nn.Linear(3, 2)
    initial = {name: value.detach().clone() for name, value in model.state_dict().items()}
    digest = tensor_state_sha256(initial)
    assert digest == tensor_state_sha256(initial)
    zero = parameter_displacement(model, initial)
    assert zero["all_l2"] == 0.0
    with torch.no_grad():
        model.weight.add_(1.0)
    moved = parameter_displacement(model, initial)
    assert moved["all_l2"] > 0.0
