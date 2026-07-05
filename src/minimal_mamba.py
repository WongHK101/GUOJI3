"""Minimal Selective SSM (Mamba) block in pure PyTorch.

Captures the essential mechanics of Mamba for feasibility testing:
- Input-dependent Δ (delta), B, C parameters via linear projections
- Selective scan along time dimension
- Residual connection + layer norm

Reference: Mamba paper (Gu & Dao, 2023), Section 3.4
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelectiveSSM(nn.Module):
    """Minimal selective state space model for feasibility verification.

    x(t) ─> Δ(x), B(x), C(x) via linear projections
              └─> selective scan SSM ─> y(t) + residual
    """
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv

        # Input projections for selective parameters
        self.x_proj = nn.Linear(d_model, d_conv * d_model + d_state * 2, bias=False)
        self.dt_proj = nn.Linear(d_conv * d_model, d_model, bias=True)

        # SSM parameters (learned, not input-dependent)
        # A_log: log of diagonal A values; A = -exp(A_log) ensures A < 0 (stable)
        self.A_log = nn.Parameter(torch.randn(d_model, d_state) * 0.1)
        self.D = nn.Parameter(torch.zeros(d_model))

        # Output projection (zero-init → near-identity residual at start)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        nn.init.zeros_(self.out_proj.weight)

        # Residual normalization
        self.norm = nn.LayerNorm(d_model)

    def _selective_scan(self, u, delta, A_log, B, C, D):
        """Discretize and scan.

        Per-channel SSM: each of d_model channels has its own A, state, and scan.

        u:     (batch, length, d_model)       - input sequence
        delta: (batch, length, d_model)        - per-channel timestep (softplus-ed, clamped)
        A_log: (d_model, d_state)              - log of -A (A = -exp(A_log) < 0 → stable)
        B:     (batch, length, d_state)        - input projection
        C:     (batch, length, d_state)        - output projection
        D:     (d_model,)                      - skip connection
        """
        batch, length, d_model = u.shape
        d_state = self.d_state

        # A = -exp(A_log) — ensures A < 0, so exp(Δ·A) ∈ (0, 1)
        A = -torch.exp(A_log)

        # Clamp delta for numerical stability
        delta = torch.clamp(delta, max=20.0)

        # Discretize: Ā = exp(Δ ⊗ A)
        # Δ: (B, L, D, 1); A: (1, 1, D, N) → Ā: (B, L, D, N)
        A_bar = torch.exp(delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))

        # Discretize B: B̄ = Δ ⊗ B (per-channel outer product)
        # Δ: (B, L, D, 1); B: (B, L, 1, N) → B̄: (B, L, D, N)
        B_bar = delta.unsqueeze(-1) * B.unsqueeze(2)

        # Input modulation: bu = B̄ ⊙ u
        # u: (B, L, D, 1); bu: (B, L, D, N)
        bu = B_bar * u.unsqueeze(-1)

        # Parallel scan: h_t = Ā_t ⊙ h_{t-1} + bu_t  (O(log L) steps)
        h = self._parallel_scan(A_bar, bu)  # (B, L, D, N)

        # Output: y_t = C_t · h_t + D ⊙ u_t
        C_expanded = C.unsqueeze(2)  # (B, L, 1, N)
        y = torch.sum(C_expanded * h, dim=-1)  # (B, L, D)
        y = y + D.unsqueeze(0).unsqueeze(1) * u

        return y

    def _iterative_scan(self, a, b):
        """Iterative prefix scan (fallback for correctness verification).

        O(L) Python loop — use _parallel_scan for speed.
        """
        h_list = [b[:, 0]]
        for t in range(1, a.shape[1]):
            h_t = a[:, t] * h_list[-1] + b[:, t]
            h_list.append(h_t)
        return torch.stack(h_list, dim=1)

    def _parallel_scan(self, a, b):
        """Parallel prefix scan: O(log L) Python steps via Hillis-Steele.

        Recurrence: h_t = a_t ⊙ h_{t-1} + b_t,  h_0 = b_0

        Uses (carry c, data d) pair representation with associative op:
          (c₁,d₁) • (c₂,d₂) = (c₁·c₂, c₁·d₂ + d₁)

        Each position starts with (a_t, b_t). After log₂(L) combine steps,
        d_t = h_t (the prefix result).
        """
        B, L, D, N = a.shape
        c = a.clone()  # carry coefficients
        d = b.clone()  # data accumulator (becomes h)

        step = 1
        while step < L:
            src = torch.arange(step, L, device=a.device)
            dst = src - step

            new_c = c.clone()
            new_d = d.clone()
            new_c[:, src] = c[:, src] * c[:, dst]
            new_d[:, src] = c[:, src] * d[:, dst] + d[:, src]
            c, d = new_c, new_d
            step *= 2

        return d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (batch, length, d_model) input sequence

        Returns:
            y: (batch, length, d_model) output sequence
        """
        residual = x
        x = self.norm(x)

        batch, length, d_model = x.shape

        # Project input to get Δ, B, C
        # x_proj: d_model -> d_conv*d_model + 2*d_state
        proj = self.x_proj(x)  # (B, L, d_conv*d_model + 2*d_state)

        # Split projections
        conv_dim = self.d_conv * d_model
        z = proj[:, :, :conv_dim]  # (B, L, d_conv*d_model)
        B = proj[:, :, conv_dim:conv_dim + self.d_state]  # (B, L, d_state)
        C = proj[:, :, conv_dim + self.d_state:]  # (B, L, d_state)

        # Delta projection (no causal conv1d for pilot)
        delta = F.softplus(self.dt_proj(z))  # (B, L, d_model)
        delta = torch.clamp(delta, max=15.0)  # Numerical stability

        # Selective scan
        y = self._selective_scan(x, delta, self.A_log, B, C, self.D)

        # Output projection + residual
        y = self.out_proj(y)
        return y + residual


class MambaBlock(nn.Module):
    """Single Mamba block wrapping SelectiveSSM with proper normalization."""
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2,
                 residual_scale: float = 0.1):
        super().__init__()
        self.residual_scale = residual_scale
        self.norm = nn.LayerNorm(d_model)
        d_inner = d_model * expand
        self.in_proj = nn.Linear(d_model, d_inner, bias=False)
        self.ssm = SelectiveSSM(d_inner, d_state, d_conv)
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)
        nn.init.zeros_(self.out_proj.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = self.in_proj(x)
        x = F.silu(x)  # Swish activation
        x = self.ssm(x)
        x = self.out_proj(x) * self.residual_scale
        return x + residual


class TCNBlock(nn.Module):
    """WaveNet-style TCN as Mamba filter replacement for ablation.

    Same interface as MambaBlock: (B, L, d) → (B, L, d).
    Uses causal dilated 1D conv + residual connection.
    Near-identity at init (zero output projection, residual_scale=0.1).
    """
    def __init__(self, d_model: int, kernel_size: int = 3, dilation: int = 2,
                 residual_scale: float = 0.1):
        super().__init__()
        self.residual_scale = residual_scale
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.pad = (kernel_size - 1) * dilation

        self.norm = nn.LayerNorm(d_model)
        # Layer 1: causal conv with dilation
        self.conv1 = nn.Conv1d(d_model, d_model * 2, kernel_size,
                               dilation=dilation, bias=False)
        # Layer 2: 1x1 conv back to d_model
        self.conv2 = nn.Conv1d(d_model * 2, d_model, 1, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        # Zero-init for near-identity at start
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.out_proj.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, d) → transpose → Conv1d → transpose → (B, L, d)
        residual = x
        x = self.norm(x)

        # (B, L, d) → (B, d, L)
        x_t = x.transpose(1, 2)
        # Causal padding: pad left only
        x_t = F.pad(x_t, (self.pad, 0))
        x_t = F.silu(self.conv1(x_t))
        x_t = self.conv2(x_t)  # 1x1 conv
        # (B, d, L) → (B, L, d)
        x = x_t.transpose(1, 2)

        x = self.out_proj(x) * self.residual_scale
        return x + residual


class DepthwiseCausalFilter(nn.Module):
    """Coordinate-preserving causal temporal filter.

    Same interface as MambaBlock/TCNBlock: (B, L, d) -> (B, L, d).
    The grouped convolution prevents cross-variable mixing: each variable is
    filtered only along its own temporal trajectory.
    """
    def __init__(self, d_model: int, kernel_size: int = 3,
                 residual_scale: float = 0.1):
        super().__init__()
        self.kernel_size = kernel_size
        self.residual_scale = residual_scale
        self.conv = nn.Conv1d(
            d_model,
            d_model,
            kernel_size=kernel_size,
            groups=d_model,
            bias=False,
        )
        nn.init.zeros_(self.conv.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x_t = x.transpose(1, 2)
        x_t = F.pad(x_t, (self.kernel_size - 1, 0))
        y = self.conv(x_t).transpose(1, 2)
        return residual + self.residual_scale * y


class DepthwiseGatedCausalFilter(nn.Module):
    """Coordinate-preserving gated causal temporal filter.

    Each variable is processed by its own grouped causal convolution. The
    value and gate branches increase per-channel temporal capacity without
    allowing cross-variable mixing.
    """
    def __init__(self, d_model: int, kernel_size: int = 3,
                 residual_scale: float = 0.1):
        super().__init__()
        self.kernel_size = kernel_size
        self.residual_scale = residual_scale
        self.conv = nn.Conv1d(
            d_model,
            2 * d_model,
            kernel_size=kernel_size,
            groups=d_model,
            bias=True,
        )
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x_t = x.transpose(1, 2)
        x_t = F.pad(x_t, (self.kernel_size - 1, 0))
        y = self.conv(x_t).transpose(1, 2)
        value, gate = torch.chunk(y, 2, dim=-1)
        y = torch.tanh(value) * torch.sigmoid(gate)
        return residual + self.residual_scale * y


def test_scan():
    """Verify parallel scan matches iterative scan."""
    torch.manual_seed(42)
    B, L, D, N = 2, 64, 8, 4
    a = torch.rand(B, L, D, N) * 0.9  # <1 to avoid explosion
    b = torch.randn(B, L, D, N)

    ssm = SelectiveSSM(d_model=8, d_state=4)
    h_iter = ssm._iterative_scan(a, b)
    h_par = ssm._parallel_scan(a, b)

    max_diff = (h_iter - h_par).abs().max().item()
    print(f"Scan max diff: {max_diff:.2e}")
    assert max_diff < 1e-5, f"Parallel scan mismatch: {max_diff}"
    print("  ✓ Parallel scan matches iterative scan")


def test_forward():
    """Quick forward pass test."""
    model = MambaBlock(d_model=32, d_state=16).cuda()
    x = torch.randn(2, 64, 32).cuda()
    y = model(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {y.shape}")
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")


if __name__ == "__main__":
    test_scan()
    test_forward()
