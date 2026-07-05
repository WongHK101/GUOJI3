"""Mamba-Enhanced JRNGC Pilot Experiment.

Three-channel comparison on non-stationary VAR:
  Channel A: Pure JRNGC (baseline)
  Channel B: JRNGC + Mamba preprocessing (no time-weighted loss)
  Channel C: JRNGC + Mamba preprocessing + time-weighted loss

Go/No-Go criteria:
  - Channel C SHD reduction vs Channel A >= 15%
  - Distance-stratified SHD: far-lag (6-10) discovery rate not degraded
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys, os, time, json
from collections import defaultdict

_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJ_ROOT)
from src.config import resolve_jrngc_path, resolve_data_dir, resolve_results_dir, resolve_device

_jrngc = resolve_jrngc_path()
if _jrngc:
    sys.path.insert(0, _jrngc)

from minimal_mamba import MambaBlock
from tgc.metrics import two_classify_metrics, remove_self_connection


# ---- Mamba Preprocessing Module ----
class MambaPreprocessor(nn.Module):
    """Lightweight Mamba front-end for non-stationarity perception.

    Outputs:
    - Z_t ∈ R^(d_cond): time-varying condition vector (CONCATENATED with x_t, not multiplied)
    - w_t ∈ [0,1]: dynamic time weights for loss weighting only

    Architecture:
      x(d, T) → MambaBlock → features F
                ├→ cond_proj → Z_t (condition vector)
                └→ weight_head → sigmoid → w_t (loss weights)
    """
    def __init__(self, d, d_state=16, d_conv=4, d_cond=4):
        super().__init__()
        self.d = d
        self.d_cond = d_cond
        self.mamba = MambaBlock(d_model=d, d_state=d_state, d_conv=d_conv, expand=2)
        self.cond_proj = nn.Linear(d, d_cond)
        self.weight_head = nn.Sequential(
            nn.Linear(d, d // 2),
            nn.ReLU(),
            nn.Linear(d // 2, 1)
        )

    def forward(self, x):
        """x: (B, d, T) → z: (B, T, d_cond), weights: (B, T)"""
        if x.dim() == 2:
            x = x.unsqueeze(0)  # (1, d, T)
        batch, d, T = x.shape

        x_t = x.transpose(1, 2)  # (B, T, d)
        features = self.mamba(x_t)  # (B, T, d)

        z = self.cond_proj(features)  # (B, T, d_cond) — concatenated with input
        w_raw = self.weight_head(features).squeeze(-1)  # (B, T)
        weights = torch.sigmoid(w_raw)

        return z, weights


# ---- Mamba-Enhanced JRNGC (concatenation-based) ----
class MambaJRNGC(nn.Module):
    """JRNGC with Mamba condition vector concatenated (NOT multiplied) with input.

    Key design (per external advisor review):
      1. Mamba → Z_t ∈ R^(d_cond): time-varying condition vector
      2. Window: CONCATENATE [x_t, Z_t] along variable dim → (d+d_cond, lag)
      3. JRNGC MLP sees (d+d_cond)*lag input — original signal preserved intact
      4. Jacobian L1 computed on original d variables only (not on Z_t portion)
      5. Time-weighted loss uses separate weight head (does not touch input)
    """
    def __init__(self, d, lag, layers=5, hidden=50, dropout=0.0,
                 jacobian_lam=0.01, d_state=16, use_time_weight_loss=False, d_cond=4):
        super().__init__()
        self.d = d
        self.lag = lag
        self.jacobian_lam = jacobian_lam
        self.use_time_weight_loss = use_time_weight_loss
        self.d_cond = d_cond

        # Mamba preprocessing
        self.preprocessor = MambaPreprocessor(d, d_state=d_state, d_cond=d_cond)

        # JRNGC MLP: input is (d + d_cond) * lag
        total_dim = (d + d_cond) * lag
        self.inputgate = nn.Linear(total_dim, hidden)
        self.outputgate = nn.Linear(hidden, d)

        self.encoders = nn.ModuleList([
            ResidualBlock(hidden, hidden, hidden, dropout)
            for _ in range(layers)
        ])

        self.loss_fn = nn.MSELoss(reduction='none')

    def forward(self, x_flat):
        """Forward through MLP backbone.

        Args:
            x_flat: (batch, (d+d_cond)*lag) flattened concatenated input
        Returns:
            pred: (batch, d) next-step prediction
        """
        h = self.inputgate(x_flat.to(torch.float32))
        for net in self.encoders:
            h = net(h)
        return self.outputgate(h)

    def preprocess_and_windowing(self, x_full):
        """Extract windows: original x concatenated with Mamba condition Z_t.

        Args:
            x_full: (d, T) full time series
        Returns:
            windows: (N, d+d_cond, lag+1) — last col is target for d vars only
            t_weights: (N,) average time weight per window
        """
        device = next(self.parameters()).device
        x_tensor = torch.tensor(x_full, device=device, dtype=torch.float32)

        if x_tensor.dim() == 2:
            x_tensor = x_tensor.unsqueeze(0)  # (1, d, T)

        # Mamba preprocessing: Z_t condition vector + time weights
        z_t, weights = self.preprocessor(x_tensor)  # z: (1,T,d_cond), w: (1,T)

        # Original x windows: (1,d,T) → (1,T,d) → unfold → (1,N,d,lag+1) → (N,d,lag+1)
        x_t = x_tensor.transpose(1, 2)
        x_win = x_t.unfold(1, self.lag + 1, 1).squeeze(0)  # (N, d, lag+1)

        # Condition Z windows: (1,T,d_cond) → unfold → (1,N,d_cond,lag+1) → (N,d_cond,lag+1)
        z_win = z_t.unfold(1, self.lag + 1, 1).squeeze(0)  # (N, d_cond, lag+1)

        # Concatenate along variable dim: (N, d+d_cond, lag+1)
        xz_win = torch.cat([x_win, z_win], dim=1)

        # Time weights per window (average over input lag steps)
        w = weights.squeeze(0)  # (T,)
        w_win = w.unfold(0, self.lag + 1, 1)  # (N, lag+1)
        t_weights = w_win[:, :self.lag].mean(dim=1)  # (N,)

        return xz_win, t_weights

    def compute_jacobian_loss(self, xz_window):
        """Jacobian regularizer on original d variables only.

        Splits concatenated input, applies autograd only to the d-var portion.
        """
        # xz_window: (M, d+d_cond, lag+1)
        xz_full = xz_window[:, :, :self.lag]  # (M, d+d_cond, lag)
        x_orig = xz_full[:, :self.d, :].detach().clone().requires_grad_(True)  # (M, d, lag)
        x_cond = xz_full[:, self.d:, :].detach()  # (M, d_cond, lag)

        x_cat = torch.cat([x_orig, x_cond], dim=1).flatten(start_dim=1)  # (M, (d+d_cond)*lag)
        y = self(x_cat)

        jac = torch.zeros((x_orig.shape[0], y.shape[1], x_orig.shape[1], x_orig.shape[2]),
                         device=x_orig.device)
        for j in range(y.shape[1]):
            grad = torch.autograd.grad(y[:, j], x_orig,
                                       grad_outputs=torch.ones_like(y[:, j]),
                                       create_graph=True)[0]
            jac[:, j] = grad
        return self.jacobian_lam * torch.mean(torch.abs(jac))

    def compute_loss(self, x_full, return_pred_loss=False):
        """Compute total loss = prediction + Jacobian + optional time-weighted."""
        windows, t_weights = self.preprocess_and_windowing(x_full)

        x_input = windows[:, :, :self.lag]  # (N, d+d_cond, lag)
        x_target = windows[:, :self.d, -1]  # (N, d) — only d original output vars

        pred = self(x_input.flatten(start_dim=1))

        per_elem_loss = self.loss_fn(pred, x_target)  # (N, d)

        if self.use_time_weight_loss:
            weighted_loss = per_elem_loss * t_weights.unsqueeze(1)
            pred_loss = weighted_loss.mean()
        else:
            pred_loss = per_elem_loss.mean()

        jac_loss = self.compute_jacobian_loss(windows[:min(len(windows), 100)])

        total_loss = pred_loss + jac_loss

        if return_pred_loss:
            return total_loss, pred_loss.detach()
        return total_loss

    def get_gc_matrix(self, x_full):
        """Extract GC matrix: Jacobian of output w.r.t. original lagged inputs only."""
        windows, _ = self.preprocess_and_windowing(x_full)
        xz_full = windows[:, :, :self.lag]  # (N, d+d_cond, lag)
        x_orig = xz_full[:, :self.d, :].detach().clone().requires_grad_(True)
        x_cond = xz_full[:, self.d:, :].detach()

        x_cat = torch.cat([x_orig, x_cond], dim=1).flatten(start_dim=1)

        jac = torch.zeros((x_orig.shape[0], x_orig.shape[1], x_orig.shape[1], x_orig.shape[2]),
                        device=x_orig.device)
        for j in range(x_orig.shape[1]):
            y = self(x_cat)[:, j]
            y.backward(torch.ones_like(y))
            jac[:, j] = x_orig.grad
            x_orig.grad.zero_()
        jac = torch.mean(torch.abs(jac), dim=0)  # (d, d, lag)
        return jac.cpu().numpy()


# ---- FiLM-Conditioned JRNGC (Z_t modulates hidden layers, not input) ----
class FiLMResidualBlock(nn.Module):
    """ResidualBlock with FiLM modulation from condition vector Z."""
    def __init__(self, input_dim, hidden, output_dim, dropout, d_cond):
        super().__init__()
        self.linear_1 = nn.Linear(input_dim, hidden)
        self.linear_2 = nn.Linear(hidden, output_dim)
        self.linear_res = nn.Linear(input_dim, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.layernorm = nn.LayerNorm(output_dim)

        # FiLM: Z → gamma, beta (initialized near identity: gamma≈0, beta≈0)
        self.film_gamma = nn.Linear(d_cond, output_dim)
        self.film_beta = nn.Linear(d_cond, output_dim)
        nn.init.zeros_(self.film_gamma.weight)
        nn.init.zeros_(self.film_gamma.bias)
        nn.init.zeros_(self.film_beta.weight)
        nn.init.zeros_(self.film_beta.bias)

    def forward(self, x, z):
        h = self.linear_1(x)
        h = self.relu(h)
        h = self.linear_2(h)
        h = self.dropout(h)
        res = self.linear_res(x)
        out = h + res

        gamma = self.film_gamma(z)
        beta = self.film_beta(z)
        out = out * (1 + gamma) + beta  # identity when gamma=beta=0

        out = self.layernorm(out)
        return out


class MambaJRNGC_FiLM(nn.Module):
    """JRNGC with Mamba FiLM conditioning at hidden layers.

    Key difference from MambaJRNGC (concat):
    - Z_t modulates hidden representations INSIDE the MLP (FiLM)
    - Input to MLP is only d*lag (original variables) — no shortcut path
    - Jacobian ∂y/∂x_orig is directly the GC score (no d/d_cond separation needed)
    - FiLM params initialized to zero → model starts as standard JRNGC
    """
    def __init__(self, d, lag, layers=5, hidden=50, dropout=0.0,
                 jacobian_lam=0.01, d_state=16, use_time_weight_loss=False, d_cond=4):
        super().__init__()
        self.d = d
        self.lag = lag
        self.jacobian_lam = jacobian_lam
        self.use_time_weight_loss = use_time_weight_loss
        self.d_cond = d_cond

        self.preprocessor = MambaPreprocessor(d, d_state=d_state, d_cond=d_cond)

        # Input is only d*lag (no condition concatenation)
        self.inputgate = nn.Linear(d * lag, hidden)
        self.outputgate = nn.Linear(hidden, d)

        self.encoders = nn.ModuleList([
            FiLMResidualBlock(hidden, hidden, hidden, dropout, d_cond)
            for _ in range(layers)
        ])

        self.loss_fn = nn.MSELoss(reduction='none')

    def forward(self, x_flat, z):
        """x_flat: (N, d*lag), z: (N, d_cond)"""
        h = self.inputgate(x_flat.to(torch.float32))
        for net in self.encoders:
            h = net(h, z)
        return self.outputgate(h)

    def preprocess_and_windowing(self, x_full):
        device = next(self.parameters()).device
        x_tensor = torch.tensor(x_full, device=device, dtype=torch.float32)
        if x_tensor.dim() == 2:
            x_tensor = x_tensor.unsqueeze(0)

        z_t, weights = self.preprocessor(x_tensor)  # z: (1,T,d_cond), w: (1,T)

        # x windows from original signal (no condition)
        x_t = x_tensor.transpose(1, 2)  # (1, T, d)
        x_win = x_t.unfold(1, self.lag + 1, 1).squeeze(0)  # (N, d, lag+1)

        # Z averaged over the input lag window → (N, d_cond)
        z_win = z_t.unfold(1, self.lag + 1, 1).squeeze(0)  # (N, d_cond, lag+1)
        z_avg = z_win[:, :, :self.lag].mean(dim=2)  # (N, d_cond)

        # Time weights
        w = weights.squeeze(0)
        w_win = w.unfold(0, self.lag + 1, 1)
        t_weights = w_win[:, :self.lag].mean(dim=1)

        return x_win, z_avg, t_weights

    def compute_loss(self, x_full, return_pred_loss=False):
        x_win, z_avg, t_weights = self.preprocess_and_windowing(x_full)

        x_input = x_win[:, :, :self.lag]  # (N, d, lag)
        x_target = x_win[:, :, -1]  # (N, d)

        pred = self(x_input.flatten(start_dim=1), z_avg)

        per_elem_loss = self.loss_fn(pred, x_target)

        if self.use_time_weight_loss:
            pred_loss = (per_elem_loss * t_weights.unsqueeze(1)).mean()
        else:
            pred_loss = per_elem_loss.mean()

        n_jac = min(len(x_win), 100)
        jac_loss = self.compute_jacobian_loss(
            x_win[:n_jac], z_avg[:n_jac])

        total_loss = pred_loss + jac_loss
        if return_pred_loss:
            return total_loss, pred_loss.detach()
        return total_loss

    def compute_jacobian_loss(self, x_window, z):
        x_input = x_window[:, :, :self.lag].detach().clone().requires_grad_(True)
        y = self(x_input.flatten(start_dim=1), z.detach())

        jac = torch.zeros((x_input.shape[0], y.shape[1],
                           x_input.shape[1], x_input.shape[2]),
                         device=x_input.device)
        for j in range(y.shape[1]):
            grad = torch.autograd.grad(y[:, j], x_input,
                                       grad_outputs=torch.ones_like(y[:, j]),
                                       create_graph=True)[0]
            jac[:, j] = grad
        return self.jacobian_lam * torch.mean(torch.abs(jac))

    def get_gc_matrix(self, x_full):
        x_win, z_avg, _ = self.preprocess_and_windowing(x_full)
        x_input = x_win[:, :, :self.lag].detach().clone().requires_grad_(True)

        jac = torch.zeros((x_input.shape[0], x_input.shape[1],
                           x_input.shape[1], x_input.shape[2]),
                        device=x_input.device)
        for j in range(x_input.shape[1]):
            y = self(x_input.flatten(start_dim=1), z_avg.detach())[:, j]
            y.backward(torch.ones_like(y))
            jac[:, j] = x_input.grad
            x_input.grad.zero_()
        jac = torch.mean(torch.abs(jac), dim=0)
        return jac.cpu().numpy()


# ---- Original JRNGC (copied for self-contained comparison) ----
class ResidualBlock(nn.Module):
    def __init__(self, input_dim, hidden, output_dim, dropout):
        super().__init__()
        self.linear_1 = nn.Linear(input_dim, hidden)
        self.linear_2 = nn.Linear(hidden, output_dim)
        self.linear_res = nn.Linear(input_dim, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.layernorm = nn.LayerNorm(output_dim)

    def forward(self, x):
        h = self.linear_1(x)
        h = self.relu(h)
        h = self.linear_2(h)
        h = self.dropout(h)
        res = self.linear_res(x)
        out = h + res
        out = self.layernorm(out)
        return out


# ---- Mamba Time-Weight-Only JRNGC ----
class MambaTimeWeightJRNGC(nn.Module):
    """Standard JRNGC with Mamba time-weighted loss ONLY.

    Key design: Mamba does NOT condition the model — it only provides
    per-sample weights for the training loss. The prediction model is
    standard JRNGC (proven causal discovery). Mamba's role:
    - Detect non-stationarity periods → w_t low (down-weight)
    - Detect stationary periods → w_t high (focus training here)

    This avoids the "shortcut" problem because Mamba has no path
    to influence predictions directly.
    """
    def __init__(self, d, lag, layers=5, hidden=50, dropout=0.0,
                 jacobian_lam=0.01, d_state=4, weight_budget_lam=0.1):
        super().__init__()
        self.d = d
        self.lag = lag
        self.jacobian_lam = jacobian_lam
        self.weight_budget_lam = weight_budget_lam

        # Standard JRNGC MLP (unchanged)
        self.inputgate = nn.Linear(d * lag, hidden)
        self.outputgate = nn.Linear(hidden, d)
        self.encoders = nn.ModuleList([
            ResidualBlock(hidden, hidden, hidden, dropout)
            for _ in range(layers)
        ])

        # Mamba for time-weight prediction only
        self.preprocessor = MambaPreprocessor(d, d_state=d_state, d_cond=1)

        self.loss_fn = nn.MSELoss(reduction='none')

    def forward(self, x_window):
        """Standard JRNGC forward (no Mamba conditioning)."""
        x = x_window.flatten(start_dim=1).to(torch.float32)
        x = self.inputgate(x)
        for net in self.encoders:
            x = net(x)
        return self.outputgate(x)

    def make_windows(self, x_full):
        """Standard windowing from original signal."""
        device = next(self.parameters()).device
        x = torch.tensor(x_full, device=device, dtype=torch.float32)
        if x.dim() == 2:
            x = x.unsqueeze(0)
        x = x.transpose(1, 2).unfold(1, self.lag + 1, 1)
        x = x.reshape(x.shape[0] * x.shape[1], x.shape[2], x.shape[3])
        return x

    def compute_loss(self, x_full, return_pred_loss=False):
        device = next(self.parameters()).device
        x_tensor = torch.tensor(x_full, device=device, dtype=torch.float32)
        if x_tensor.dim() == 2:
            x_tensor = x_tensor.unsqueeze(0)

        # Get time weights from Mamba
        _, weights = self.preprocessor(x_tensor)  # weights: (1, T)

        # Standard JRNGC windows
        windows = self.make_windows(x_full)
        x_input = windows[:, :, :self.lag]
        x_target = windows[:, :, -1]

        pred = self(x_input)
        per_elem_loss = self.loss_fn(pred, x_target)  # (N, d)

        # Time-weighted prediction loss (with weight floor to prevent collapse)
        w = weights.squeeze(0)  # (T,)
        w_win = w.unfold(0, self.lag + 1, 1)  # (N, lag+1)
        t_weights = w_win[:, :self.lag].mean(dim=1)  # (N,)
        t_weights = torch.clamp(t_weights, min=0.05)  # floor: no sample fully ignored
        pred_loss = (per_elem_loss * t_weights.unsqueeze(1)).mean()

        # Weight budget: penalize if mean weight deviates from 1.0
        weight_budget_loss = self.weight_budget_lam * (1.0 - t_weights.mean()) ** 2

        # Jacobian loss (standard)
        jac_x = windows[:min(len(windows), 100), :, :self.lag].detach().clone()
        jac_x.requires_grad_(True)
        y = self(jac_x)
        jac = torch.zeros((jac_x.shape[0], jac_x.shape[1],
                           jac_x.shape[1], jac_x.shape[2]),
                         device=jac_x.device)
        for j in range(jac_x.shape[1]):
            grad = torch.autograd.grad(y[:, j], jac_x,
                                       grad_outputs=torch.ones_like(y[:, j]),
                                       create_graph=True)[0]
            jac[:, j] = grad
        jac_loss = self.jacobian_lam * torch.mean(torch.abs(jac))

        total_loss = pred_loss + jac_loss + weight_budget_loss
        if return_pred_loss:
            return total_loss, pred_loss.detach()
        return total_loss

    def get_gc_matrix(self, x_full):
        """Standard JRNGC GC extraction (Jacobian w.r.t. lagged inputs)."""
        windows = self.make_windows(x_full)
        x_input = windows[:, :, :self.lag].detach().clone().requires_grad_(True)
        jac = torch.zeros((x_input.shape[0], x_input.shape[1],
                           x_input.shape[1], x_input.shape[2]),
                        device=x_input.device)
        for j in range(x_input.shape[1]):
            y = self(x_input)[:, j]
            y.backward(torch.ones_like(y))
            jac[:, j] = x_input.grad
            x_input.grad.zero_()
        jac = torch.mean(torch.abs(jac), dim=0)
        return jac.cpu().numpy()


# ---- Mamba Input-Filter JRNGC ----
class MambaFilterJRNGC(nn.Module):
    """JRNGC with Mamba as input filter (not condition generator).

    Key design (per advisor feedback):
    - Mamba transforms raw x → filtered x' within SAME d-dimensional space
    - NO new dimensions, NO condition vectors, NO external information
    - Jacobian ∂y/∂x' is directly the GC score (no separation needed)
    - MambaBlock's residual connection ensures near-identity at init
    - Model CANNOT route predictions to external channels (no channels exist)
    - ortho_lam: penalizes deviation from identity (stabilizes variance)
    - residual_scale: dampens Mamba contribution (default 0.1)

    Mamba's role: selective temporal filtering — attenuate non-stationary
    noise while preserving causal signal in each variable's time series.
    """
    def __init__(self, d, lag, layers=5, hidden=50, dropout=0.0,
                 jacobian_lam=0.01, d_state=4, ortho_lam=0.01, residual_scale=0.1,
                 filter_type="mamba"):
        super().__init__()
        self.d = d
        self.lag = lag
        self.jacobian_lam = jacobian_lam
        self.ortho_lam = ortho_lam

        # Filter: (B, L, d) → (B, L, d) — same dimension, temporal processing
        from minimal_mamba import MambaBlock, TCNBlock, DepthwiseCausalFilter
        if filter_type == "tcn":
            self.filter_mamba = TCNBlock(d_model=d, kernel_size=3, dilation=2,
                                         residual_scale=residual_scale)
        elif filter_type == "depthwise":
            self.filter_mamba = DepthwiseCausalFilter(d_model=d, kernel_size=3,
                                                      residual_scale=residual_scale)
        else:
            self.filter_mamba = MambaBlock(d_model=d, d_state=d_state, d_conv=4, expand=2,
                                           residual_scale=residual_scale)

        # Standard JRNGC MLP (unchanged)
        self.inputgate = nn.Linear(d * lag, hidden)
        self.outputgate = nn.Linear(hidden, d)
        self.encoders = nn.ModuleList([
            ResidualBlock(hidden, hidden, hidden, dropout)
            for _ in range(layers)
        ])

        self.loss_fn = nn.MSELoss(reduction='mean')

    def forward(self, x_window):
        """Standard JRNGC forward."""
        x = x_window.flatten(start_dim=1).to(torch.float32)
        x = self.inputgate(x)
        for net in self.encoders:
            x = net(x)
        return self.outputgate(x)

    def make_filtered_windows(self, x_full):
        """Apply Mamba filter then extract windows.

        Returns:
            x_win: (N, d, lag+1) filtered windows
            x_orig_t: (1, T, d) original signal (for ortho loss)
            x_filt_t: (1, T, d) filtered signal (for ortho loss)
        """
        device = next(self.parameters()).device
        x = torch.tensor(x_full, device=device, dtype=torch.float32)
        if x.dim() == 2:
            x = x.unsqueeze(0)  # (1, d, T)

        # Mamba filter: (1, d, T) → (1, T, d) → Mamba → (1, T, d) → (1, d, T)
        x_t = x.transpose(1, 2)  # (1, T, d)
        x_filtered = self.filter_mamba(x_t)  # (1, T, d)
        x_filtered_t = x_filtered.transpose(1, 2)  # (1, d, T)

        # Standard windowing on filtered signal
        x_win = x_filtered.unfold(1, self.lag + 1, 1)  # (1, N, d, lag+1)
        x_win = x_win.reshape(-1, self.d, self.lag + 1)  # (N, d, lag+1)
        return x_win, x_t.detach(), x_filtered

    def compute_loss(self, x_full, return_pred_loss=False):
        windows, x_orig_t, x_filt_t = self.make_filtered_windows(x_full)
        x_input = windows[:, :, :self.lag]
        x_target = windows[:, :, -1]

        pred = self(x_input)
        pred_loss = self.loss_fn(pred, x_target)

        # Orthogonality regularization: penalize large filter deviations
        # loss_ortho = λ * ||x_filt - x_orig||² / (||x_filt||·||x_orig||)
        diff_sq = torch.mean((x_filt_t - x_orig_t) ** 2)
        x_orig_norm = torch.sqrt(torch.mean(x_orig_t ** 2) + 1e-8)
        x_filt_norm = torch.sqrt(torch.mean(x_filt_t ** 2) + 1e-8)
        loss_ortho = self.ortho_lam * diff_sq / (x_orig_norm * x_filt_norm + 1e-8)

        # Jacobian loss (standard, on filtered windows)
        jac_x = windows[:min(len(windows), 100), :, :self.lag].detach().clone()
        jac_x.requires_grad_(True)
        y = self(jac_x)
        jac = torch.zeros((jac_x.shape[0], jac_x.shape[1],
                           jac_x.shape[1], jac_x.shape[2]),
                         device=jac_x.device)
        for j in range(jac_x.shape[1]):
            grad = torch.autograd.grad(y[:, j], jac_x,
                                       grad_outputs=torch.ones_like(y[:, j]),
                                       create_graph=True)[0]
            jac[:, j] = grad
        jac_loss = self.jacobian_lam * torch.mean(torch.abs(jac))

        total_loss = pred_loss + jac_loss + loss_ortho
        if return_pred_loss:
            return total_loss, pred_loss.detach()
        return total_loss

    def get_gc_matrix(self, x_full):
        windows, _, _ = self.make_filtered_windows(x_full)
        x_input = windows[:, :, :self.lag].detach().clone().requires_grad_(True)
        jac = torch.zeros((x_input.shape[0], x_input.shape[1],
                           x_input.shape[1], x_input.shape[2]),
                        device=x_input.device)
        for j in range(x_input.shape[1]):
            y = self(x_input)[:, j]
            y.backward(torch.ones_like(y))
            jac[:, j] = x_input.grad
            x_input.grad.zero_()
        jac = torch.mean(torch.abs(jac), dim=0)
        return jac.cpu().numpy()


class BaselineJRNGC(nn.Module):
    """Original JRNGC without Mamba preprocessing."""
    def __init__(self, d, lag, layers=5, hidden=50, dropout=0.0, jacobian_lam=0.01):
        super().__init__()
        self.d = d
        self.lag = lag
        self.jacobian_lam = jacobian_lam

        self.inputgate = nn.Linear(d * lag, hidden)
        self.outputgate = nn.Linear(hidden, d)
        self.encoders = nn.ModuleList([
            ResidualBlock(hidden, hidden, hidden, dropout)
            for _ in range(layers)
        ])
        self.loss_fn = nn.MSELoss(reduction='mean')

    def forward(self, x):
        x = x.flatten(start_dim=1).to(torch.float32)
        x = self.inputgate(x)
        for net in self.encoders:
            x = net(x)
        x = self.outputgate(x)
        return x

    def make_windows(self, x_full):
        device = next(self.parameters()).device
        x = torch.tensor(x_full, device=device, dtype=torch.float32)
        if x.dim() == 2:
            x = x.unsqueeze(0)
        x = x.transpose(1, 2).unfold(1, self.lag + 1, 1)
        x = x.reshape(x.shape[0] * x.shape[1], x.shape[2], x.shape[3])
        return x

    def compute_loss(self, x_full):
        windows = self.make_windows(x_full)
        x_input = windows[:, :, :self.lag]
        x_target = windows[:, :, -1]
        pred = self(x_input)
        pred_loss = self.loss_fn(pred, x_target)

        # Jacobian loss
        jac_x = windows[:min(len(windows), 100), :, :self.lag].detach().clone()
        jac_x.requires_grad_(True)
        y = self(jac_x)
        jac = torch.zeros((jac_x.shape[0], jac_x.shape[1], jac_x.shape[1], jac_x.shape[2]),
                         device=jac_x.device)
        for j in range(jac_x.shape[1]):
            grad = torch.autograd.grad(y[:, j], jac_x,
                                       grad_outputs=torch.ones_like(y[:, j]),
                                       create_graph=True)[0]
            jac[:, j] = grad
        jac_loss = self.jacobian_lam * torch.mean(torch.abs(jac))

        return pred_loss + jac_loss

    def get_gc_matrix(self, x_full):
        windows = self.make_windows(x_full)
        x_input = windows[:, :, :self.lag].detach().clone().requires_grad_(True)
        jac = torch.zeros((x_input.shape[0], x_input.shape[1], x_input.shape[1], x_input.shape[2]),
                        device=x_input.device)
        for j in range(x_input.shape[1]):
            y = self(x_input)[:, j]
            y.backward(torch.ones_like(y))
            jac[:, j] = x_input.grad
            x_input.grad.zero_()
        jac = torch.mean(torch.abs(jac), dim=0)
        return jac.cpu().numpy()


# ---- Metrics ----
def _compute_metrics_core(gt_2d, pr_2d, gc_true_full=None, gc_pred_full=None):
    """Core metric computation given collapsed 2D ground-truth and prediction.

    All lag-stratified-AUROC logic lives here; the caller decides how to
    collapse the 3D arrays into 2D.
    """
    # Threshold-dependent metrics
    (f1, f1_trd), (acc, acc_trd), (auroc, _, _), (auprc, _, _) = two_classify_metrics(pr_2d, gt_2d)

    # ---- SHD and normalized SHD (top-k thresholding) ----
    gt_int = gt_2d.astype(np.int32)
    n_edges_true = int(np.sum(gt_int))
    if n_edges_true > 0:
        thr = np.sort(pr_2d.ravel())[-n_edges_true]
        pred_binary = (pr_2d >= thr).astype(np.int32)
    else:
        pred_binary = np.zeros_like(gt_int, dtype=np.int32)
    shd = int(np.sum(np.abs(gt_int - pred_binary)))
    nshd = shd / max(n_edges_true, 1)

    # ---- MCC (Matthews Correlation Coefficient) ----
    tp = int(np.sum((pred_binary == 1) & (gt_int == 1)))
    tn = int(np.sum((pred_binary == 0) & (gt_int == 0)))
    fp = int(np.sum((pred_binary == 1) & (gt_int == 0)))
    fn = int(np.sum((pred_binary == 0) & (gt_int == 1)))
    mcc_denom = np.sqrt(float(max((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn), 0)))
    mcc = float(tp * tn - fp * fn) / max(mcc_denom, 1e-8)

    # ---- Distance-stratified AUROC ----
    d = gt_2d.shape[0]
    lag_metrics = {}
    if gc_true_full is not None and gc_pred_full is not None \
            and gc_true_full.ndim == 3 and gc_pred_full.ndim == 3:
        max_lag = min(gc_true_full.shape[2], gc_pred_full.shape[2])
        for start, end in [(0, 2), (2, 4), (4, 6), (6, max_lag)]:
            if start >= max_lag:
                break
            end = min(end, max_lag)
            gt_slice = gc_true_full[:, :, start:end]
            pr_slice = gc_pred_full[:, :, start:end]
            gt_bin = (gt_slice.sum(axis=2) > 0).astype(np.int32)
            pr_score = pr_slice.max(axis=2)
            gt_bin = remove_self_connection(gt_bin)
            pr_score = remove_self_connection(pr_score.astype(np.float64))
            (_, _), (_, _), (auroc_lag, _, _), (_, _, _) = two_classify_metrics(pr_score, gt_bin)
            lag_metrics[f"lag_{start}-{end}"] = float(auroc_lag)

    return {
        "auroc": float(auroc),
        "auprc": float(auprc),
        "f1": float(f1),
        "acc": float(acc),
        "shd_topk": shd,
        "nshd_topk": float(nshd),
        "mcc_topk": float(mcc),
        "n_edges_true": n_edges_true,
        **lag_metrics
    }


def compute_metrics(gc_true, gc_pred):
    """Compute comprehensive metrics using lag-0 ground truth (backward-compatible).

    NOTE: This uses gc_true[:,:,0] as the ground-truth graph. For factorial
    experiments where edges are distributed across lags, prefer
    compute_metrics_multimode() which additionally reports summary_max and
    summary_mean metrics that aggregate across all lags.
    """
    # Collapse to 2D summary if needed
    if gc_true.ndim == 3:
        gc_true_2d = gc_true[:, :, 0]
    else:
        gc_true_2d = gc_true
    if gc_pred.ndim == 3:
        gc_pred_full = gc_pred
        gc_pred_summary = np.max(np.abs(gc_pred), axis=2)
    else:
        gc_pred_full = None
        gc_pred_summary = gc_pred

    # Remove self-connections for metric computation
    gt = remove_self_connection(gc_true_2d.astype(np.int32))
    pr = remove_self_connection(gc_pred_summary.astype(np.float64))

    return _compute_metrics_core(gt, pr, gc_true, gc_pred_full)


def compute_metrics_multimode(gc_true, gc_pred):
    """Compute metrics in three summary modes: lag0, summary_max, summary_mean.

    - **lag0**: ground truth = gc_true[:,:,0] (edges in first lag only).
      Prediction = max(|pred|) over lags.  This is the original JRNGC metric.
    - **summary_max**: ground truth = 1[∃k: A_{ij}^{(k)} ≠ 0] (edge exists at
      any lag).  Prediction = max_k |pred_{ij}^{(k)}|.
    - **summary_mean**: same ground truth as summary_max.  Prediction =
      (1/K) Σ_k |pred_{ij}^{(k)}|.

    Returns:
        dict with keys "lag0", "summary_max", "summary_mean", each holding
        a metrics dict (auroc, auprc, f1, shd, nshd, mcc, n_edges_true, ...).
    """
    if gc_true.ndim != 3 or gc_pred.ndim != 3:
        # Fallback: all modes are identical for 2D inputs
        single = compute_metrics(gc_true, gc_pred)
        return {"lag0": single, "summary_max": single, "summary_mean": single}

    d0, d1, lag = gc_true.shape
    result = {}

    # ---- lag0 ----
    gt_lag0 = gc_true[:, :, 0]
    pr_lag0 = np.max(np.abs(gc_pred), axis=2)
    gt_lag0 = remove_self_connection(gt_lag0.astype(np.int32))
    pr_lag0 = remove_self_connection(pr_lag0.astype(np.float64))
    result["lag0"] = _compute_metrics_core(gt_lag0, pr_lag0, gc_true, gc_pred)

    # ---- summary (shared ground truth) ----
    gt_summary = (gc_true.sum(axis=2) > 0).astype(np.int32)
    gt_summary = remove_self_connection(gt_summary)

    # summary_max
    pr_max = np.max(np.abs(gc_pred), axis=2)
    pr_max = remove_self_connection(pr_max.astype(np.float64))
    result["summary_max"] = _compute_metrics_core(gt_summary, pr_max, gc_true, gc_pred)

    # summary_mean
    pr_mean = np.mean(np.abs(gc_pred), axis=2)
    pr_mean = remove_self_connection(pr_mean.astype(np.float64))
    result["summary_mean"] = _compute_metrics_core(gt_summary, pr_mean, gc_true, gc_pred)

    return result


def compute_seed_confidence_intervals(metrics_list, confidence=0.95):
    """Compute mean ± t-distribution CI from list of per-seed metric dicts.

    Args:
        metrics_list: list of dicts, each from compute_metrics()
        confidence: confidence level (default 0.95 for 95% CI)

    Returns:
        dict: {metric_name: {"mean": float, "std": float,
                             "ci_lower": float, "ci_upper": float}}
    """
    from scipy import stats

    if len(metrics_list) < 2:
        result = {}
        for k, v in metrics_list[0].items():
            if isinstance(v, (int, float, np.floating, np.integer)):
                result[k] = {"mean": float(v), "std": 0.0,
                             "ci_lower": float(v), "ci_upper": float(v)}
        return result

    # Filter to numeric keys only
    keys = [k for k in metrics_list[0]
            if isinstance(metrics_list[0][k], (int, float, np.floating, np.integer))]
    result = {}
    for k in keys:
        vals = np.array([float(m[k]) for m in metrics_list])
        mean = np.mean(vals)
        std = np.std(vals, ddof=1)
        se = std / np.sqrt(len(vals))
        t_crit = stats.t.ppf((1 + confidence) / 2, df=len(vals) - 1)
        result[k] = {
            "mean": float(mean),
            "std": float(std),
            "ci_lower": float(mean - t_crit * se),
            "ci_upper": float(mean + t_crit * se),
        }
    return result


def train_model(model, x, max_iter=5000, lr=1e-3, weight_decay=0.0,
                lookback=10, check_every=50, verbose=False):
    """Train a model and return trained model + metrics."""
    device = next(model.parameters()).device
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_loss = float('inf')
    best_it = 0
    best_state = None

    for it in range(max_iter):
        model.train()
        optimizer.zero_grad()
        loss = model.compute_loss(x)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Stability
        optimizer.step()

        if it % check_every == 0:
            loss_val = loss.item()
            if loss_val < best_loss:
                best_loss = loss_val
                best_it = it
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            elif (it - best_it) >= lookback * check_every and it > 1000:
                if verbose:
                    print(f"    Early stop at iter {it}")
                break

            if verbose and it % 200 == 0:
                print(f"    iter {it}: loss={loss_val:.5f}")

    # Restore best
    if best_state is not None:
        model.load_state_dict(best_state)

    return model, best_loss


def run_experiment(x, gc_true, seed, verbose=True):
    """Run 3-channel comparison on one dataset."""
    d = x.shape[0]
    lag = 7
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    results = {}

    # --- Channel A: Baseline JRNGC ---
    if verbose:
        print("  [Channel A] Baseline JRNGC...")
    torch.manual_seed(seed)
    np.random.seed(seed)
    model_a = BaselineJRNGC(d=d, lag=lag, layers=5, hidden=50, jacobian_lam=0.01).to(device)
    model_a, loss_a = train_model(model_a, x, verbose=verbose)
    gc_pred_a = model_a.get_gc_matrix(x)
    metrics_a = compute_metrics(gc_true, gc_pred_a)
    metrics_a["train_loss"] = float(loss_a)
    results["A_baseline"] = metrics_a
    if verbose:
        print(f"    AUROC={metrics_a['auroc']:.4f}  SHD={metrics_a['shd']}")

    # --- Channel B: Mamba concat, no time-weighted loss ---
    if verbose:
        print("  [Channel B] Mamba-JRNGC concat (no time-weight)...")
    torch.manual_seed(seed)
    np.random.seed(seed)
    model_b = MambaJRNGC(d=d, lag=lag, layers=5, hidden=50,
                         jacobian_lam=0.01, d_state=16, d_cond=4,
                         use_time_weight_loss=False).to(device)
    model_b, loss_b = train_model(model_b, x, verbose=verbose)
    gc_pred_b = model_b.get_gc_matrix(x)
    metrics_b = compute_metrics(gc_true, gc_pred_b)
    metrics_b["train_loss"] = float(loss_b)
    results["B_mamba_concat_no_tw"] = metrics_b
    if verbose:
        print(f"    AUROC={metrics_b['auroc']:.4f}  SHD={metrics_b['shd']}")

    # --- Channel C: Mamba concat + time-weighted loss ---
    if verbose:
        print("  [Channel C] Mamba-JRNGC concat + time-weighted loss...")
    torch.manual_seed(seed)
    np.random.seed(seed)
    model_c = MambaJRNGC(d=d, lag=lag, layers=5, hidden=50,
                         jacobian_lam=0.01, d_state=16, d_cond=4,
                         use_time_weight_loss=True).to(device)
    model_c, loss_c = train_model(model_c, x, verbose=verbose)
    gc_pred_c = model_c.get_gc_matrix(x)
    metrics_c = compute_metrics(gc_true, gc_pred_c)
    metrics_c["train_loss"] = float(loss_c)
    results["C_mamba_concat_tw"] = metrics_c
    if verbose:
        print(f"    AUROC={metrics_c['auroc']:.4f}  SHD={metrics_c['shd']}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str,
                        default=os.path.join(resolve_data_dir(), "nonstationary_var"))
    parser.add_argument("--d", type=int, default=10)
    parser.add_argument("--lag", type=int, default=7)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--output", type=str,
                        default=os.path.join(resolve_results_dir(), "pilot_results.json"))
    parser.add_argument("--max_iter", type=int, default=5000)
    args = parser.parse_args()

    print(f"{'='*60}")
    print(f"  Mamba-Enhanced JRNGC Pilot Experiment")
    print(f"  d={args.d}, lag={args.lag}, seeds={args.seeds}")
    print(f"{'='*60}")

    all_results = {}
    seed_results = defaultdict(lambda: defaultdict(list))

    for seed in range(args.seeds):
        print(f"\n--- Seed {seed} ---")

        # Load data
        data_path = os.path.join(args.data_dir, f"num_nodes_{args.d}",
                                 f"true_lag_{args.lag}", "noise_scale_1", f"seed_{seed}")
        x = np.load(os.path.join(data_path, "_x.npy"))
        gc = np.load(os.path.join(data_path, "_gc.npy"))

        results = run_experiment(x, gc, seed, verbose=True)
        all_results[f"seed_{seed}"] = results

        # Aggregate
        for channel, metrics in results.items():
            for k, v in metrics.items():
                seed_results[channel][k].append(v)

    # Summary
    print(f"\n{'='*60}")
    print(f"  SUMMARY (mean ± std over {args.seeds} seeds)")
    print(f"{'='*60}")

    summary = {}
    for channel in ["A_baseline", "B_mamba_concat_no_tw", "C_mamba_concat_tw"]:
        s = {}
        for k in seed_results[channel]:
            vals = seed_results[channel][k]
            if isinstance(vals[0], (int, float, np.floating, np.integer)):
                s[k] = {
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals))
                }
        summary[channel] = s

        label = {"A_baseline": "A: Baseline JRNGC",
                 "B_mamba_concat_no_tw": "B: +Mamba-concat (no TW)",
                 "C_mamba_concat_tw": "C: +Mamba-concat +TimeWeight"}
        print(f"\n{label[channel]}:")
        for k, v in s.items():
            print(f"  {k:20s}: {v['mean']:.4f} ± {v['std']:.4f}")

    # Go/No-Go check
    if "C_mamba_concat_tw" in summary and "A_baseline" in summary:
        shd_a = summary["A_baseline"].get("shd", {}).get("mean", 1.0)
        shd_c = summary["C_mamba_concat_tw"].get("shd", {}).get("mean", 1.0)
        if shd_a > 0:
            shd_reduction = (shd_a - shd_c) / shd_a * 100
        else:
            shd_reduction = 0

        auroc_a = summary["A_baseline"].get("auroc", {}).get("mean", 0.0)
        auroc_c = summary["C_mamba_concat_tw"].get("auroc", {}).get("mean", 0.0)

        print(f"\n{'='*60}")
        print(f"  GO/NO-GO DECISION")
        print(f"{'='*60}")
        print(f"  SHD reduction (C vs A): {shd_reduction:.1f}%  [need >= 15%]")
        print(f"  AUROC delta (C vs A):   {auroc_c - auroc_a:+.4f}")

        if shd_reduction >= 15:
            print(f"\n  ✓ PASS — SHD reduction >= 15%")
            print(f"  → Proceed with full implementation")
            go = True
        else:
            print(f"\n  ✗ FAIL — SHD reduction < 15%")
            print(f"  → Fall back to Plan 2 (JRNGC + time-varying condition input)")
            go = False

        summary["go_nogo"] = {
            "shd_reduction_pct": shd_reduction,
            "auroc_delta": auroc_c - auroc_a,
            "pass": go
        }

    # Save results
    output_dir = os.path.dirname(args.output) or '.'
    os.makedirs(output_dir, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({"per_seed": all_results, "summary": summary}, f, indent=2)
    print(f"\nResults saved to {args.output}")


import argparse
if __name__ == "__main__":
    main()
