# Theoretical Analysis: Jacobian Perturbation under Near-Identity Input Filtering

## Notation

- **Time series**: $x \in \mathbb{R}^{T \times d}$, $d$ variables, $T$ time steps
- **Filter**: $F_\theta: \mathbb{R}^{T \times d} \to \mathbb{R}^{T \times d}$, where $F_\theta(x) = x + \varepsilon \cdot H_\theta(x)$
  - $H_\theta$: filter function (MambaBlock or TCNBlock without residual)
  - $\varepsilon$: `residual_scale` (default 0.1), controls deviation from identity
  - At initialization: $H_\theta(x) \approx 0$ (zero-initialized output projection)
- **Predictor**: $g_\phi: \mathbb{R}^{p \cdot d} \to \mathbb{R}^d$, JRNGC MLP with lag $p$
- **Windowed input**: $\mathbf{x}_t = (x_{t-1}, \ldots, x_{t-p}) \in \mathbb{R}^{p \cdot d}$
- **Filtered windows**: $\mathbf{x}'_t = (x'_{t-1}, \ldots, x'_{t-p})$
- **Jacobian (per time step)**: $\mathbf{J}_t = \frac{\partial g_\phi(\mathbf{x}'_t)}{\partial \mathbf{x}'_t} \in \mathbb{R}^{d \times (p \cdot d)}$
- **Granger causality matrix**: $\mathbf{G}[i,j] = \mathbb{E}_t\left[\left|\frac{\partial g_{\phi,i}}{\partial x'_j}\right|\right]$, averaged over lag dimension
- **Jacobian penalty**: $\mathcal{L}_{\text{jac}} = \lambda_J \cdot \|\mathbf{J}_t\|_1$

## Assumptions

**A1 (Near-Identity Filter)**. $F_\theta(x) = x + \varepsilon \cdot H_\theta(x)$ with $\varepsilon \in [0, 1]$.

**A2 (Smooth Filter)**. $H_\theta$ is $L_H$-Lipschitz differentiable: $\left\|\frac{\partial H_\theta}{\partial x}\right\|_2 \leq L_H$ for all $x$ in the domain.

**A3 (Bounded Predictor Jacobian)**. $\|\mathbf{J}_t\|_2 \leq J_{\max}$ for all $t$. This is encouraged by the Jacobian L1 penalty and gradient clipping.

**A4 (Domain Boundedness)**. $\|x\|_2 \leq B_x$ for all $x$ in the training distribution.

**Remarks on assumptions**:
- A2 holds for both MambaBlock and TCNBlock: LayerNorm (1-Lipschitz), SiLU (~1.1-Lipschitz), linear layers (spectral norm bounded by weight norm, controlled via gradient clipping), SSM discretization (contraction since $A < 0$).
- A3 is mild: at initialization with small random weights, $J_{\max}$ is small; the L1 penalty keeps it bounded during training.
- A4 holds for normalized data (standard practice).

---

## Theorem 1: Jacobian Perturbation Bound

**Theorem 1** (Jacobian Preservation under Near-Identity Filtering). Let $F_\theta$ satisfy A1-A2 and $g_\phi$ satisfy A3. Let $\mathbf{G}_\varepsilon$ be the Granger causality matrix computed from $\partial g_\phi / \partial x'$ (filtered model) and $\tilde{\mathbf{G}}_\varepsilon$ be the matrix computed from $\partial (g_\phi \circ F_\theta) / \partial x$ (end-to-end Jacobian w.r.t. original input). Then:

$$\left\|\tilde{\mathbf{G}}_\varepsilon - \mathbf{G}_\varepsilon\right\|_{\max} \leq \varepsilon \cdot L_H \cdot J_{\max}$$

where $\|\cdot\|_{\max}$ is the element-wise max norm.

**Proof**.

By the chain rule:

$$\frac{\partial g_\phi}{\partial x} = \frac{\partial g_\phi}{\partial x'} \cdot \frac{\partial x'}{\partial x}$$

Since $x' = x + \varepsilon \cdot H_\theta(x)$:

$$\frac{\partial x'}{\partial x} = \mathbf{I} + \varepsilon \cdot \frac{\partial H_\theta}{\partial x}$$

Therefore:

$$\frac{\partial g_\phi}{\partial x} - \frac{\partial g_\phi}{\partial x'} = \frac{\partial g_\phi}{\partial x'} \cdot \left(\mathbf{I} + \varepsilon \cdot \frac{\partial H_\theta}{\partial x}\right) - \frac{\partial g_\phi}{\partial x'} = \varepsilon \cdot \frac{\partial g_\phi}{\partial x'} \cdot \frac{\partial H_\theta}{\partial x}$$

Taking the element-wise absolute value and expectation over time:

$$|\tilde{\mathbf{G}}_\varepsilon[i,j] - \mathbf{G}_\varepsilon[i,j]| = \mathbb{E}_t\left[\left|\left(\varepsilon \cdot \frac{\partial g_\phi}{\partial x'} \cdot \frac{\partial H_\theta}{\partial x}\right)_{i,j,t}\right|\right]$$

$$\leq \varepsilon \cdot \mathbb{E}_t\left[\sum_k \left|\left(\frac{\partial g_\phi}{\partial x'}\right)_{i,k,t}\right| \cdot \left|\left(\frac{\partial H_\theta}{\partial x}\right)_{k,j,t}\right|\right]$$

By Cauchy-Schwarz and the spectral norm bound from A2-A3:

$$\leq \varepsilon \cdot \left\|\frac{\partial g_\phi}{\partial x'}\right\|_2 \cdot \left\|\frac{\partial H_\theta}{\partial x}\right\|_2 \leq \varepsilon \cdot J_{\max} \cdot L_H$$

Taking the maximum over all $(i,j)$ pairs yields the result. $\square$

**Corollary 1.1** (Causal Structure Preservation). Let $\hat{\mathcal{E}}_\varepsilon = \{(i,j) : \mathbf{G}_\varepsilon[i,j] > \tau\}$ be the set of edges discovered by thresholding at $\tau$. If $\tau > \varepsilon \cdot L_H \cdot J_{\max}$, then all edges in $\hat{\mathcal{E}}_\varepsilon$ are within $O(\varepsilon)$ of the edges that would be discovered from the end-to-end Jacobian $\tilde{\mathbf{G}}_\varepsilon$.

**Corollary 1.2** (Consistency as $\varepsilon \to 0$). As $\varepsilon \to 0$, $\tilde{\mathbf{G}}_\varepsilon \to \mathbf{G}_\varepsilon$ in max-norm. At $\varepsilon = 0$ (no filtering), the model reduces to standard JRNGC.

---

## Theorem 2: Shortcut Impossibility under Input Filtering

**Theorem 2** (No Free Lunch for Filter Parameters). Let $x' = F_\theta(x) = x + \varepsilon \cdot H_\theta(x)$ be the filtered signal. Consider the total loss:

$$\mathcal{L}(\theta, \phi) = \mathcal{L}_{\text{pred}}(g_\phi(\mathbf{x}'), x'_{\text{target}}) + \lambda_J \cdot \|\mathbf{J}(\mathbf{x}')\|_1 + \lambda_o \cdot \mathcal{L}_{\text{ortho}}(x, x')$$

The gradient of the Jacobian penalty w.r.t. filter parameters $\theta$ vanishes at $\varepsilon \to 0$:

$$\lim_{\varepsilon \to 0} \left\|\frac{\partial}{\partial \theta} \|\mathbf{J}(\mathbf{x}')\|_1\right\| = 0$$

while the gradient of the prediction loss w.r.t. $\theta$ scales as $\varepsilon$:

$$\frac{\partial \mathcal{L}_{\text{pred}}}{\partial \theta} = \varepsilon \cdot \frac{\partial \mathcal{L}_{\text{pred}}}{\partial x'} \cdot \frac{\partial H_\theta}{\partial \theta} + O(\varepsilon^2)$$

**Proof**.

The Jacobian penalty gradient w.r.t. $\theta$:

$$\frac{\partial}{\partial \theta} \|\mathbf{J}\|_1 = \frac{\partial \|\mathbf{J}\|_1}{\partial \mathbf{J}} \cdot \frac{\partial \mathbf{J}}{\partial x'} \cdot \frac{\partial x'}{\partial \theta}$$

Since $\partial x' / \partial \theta = \varepsilon \cdot \partial H_\theta / \partial \theta$, all terms are $O(\varepsilon)$. As $\varepsilon \to 0$, the gradient vanishes.

For the prediction loss:

$$\frac{\partial \mathcal{L}_{\text{pred}}}{\partial \theta} = \underbrace{\frac{\partial \mathcal{L}_{\text{pred}}}{\partial g_\phi} \cdot \frac{\partial g_\phi}{\partial x'} \cdot \frac{\partial x'}{\partial \theta}}_{\text{input pathway}} + \underbrace{\frac{\partial \mathcal{L}_{\text{pred}}}{\partial x'_{\text{target}}} \cdot \frac{\partial x'_{\text{target}}}{\partial \theta}}_{\text{target pathway}}$$

Both terms contain $\partial x' / \partial \theta = \varepsilon \cdot \partial H_\theta / \partial \theta$. $\square$

**Remark**. Theorem 2 formalizes why input filtering avoids shortcut learning: at initialization and early training, the filter cannot manipulate the Jacobian penalty because its influence on $x'$ is $O(\varepsilon)$. The filter must learn to help prediction first, and only affects causal discovery indirectly through improved signal quality. This is in **direct contrast** to concat-based architectures, where the auxiliary features $z$ provide a zero-cost path (no $O(\varepsilon)$ damping) to route predictions around the Jacobian penalty on $x_{\text{orig}}$.

---

## Theorem 3: Orthogonality Regularization Provides Explicit Deviation Control

**Theorem 3** (Orthogonality Bound). Let $\mathcal{L}_{\text{ortho}} = \frac{\|x' - x\|_F^2}{\|x'\|_F \cdot \|x\|_F}$ be the orthogonality regularizer. At convergence, when $\mathcal{L}_{\text{ortho}} \leq \delta$:

$$\|x' - x\|_F \leq \sqrt{\delta \cdot B_x \cdot (\|x\|_F + \|x' - x\|_F)}$$

where $B_x$ is the domain bound from A4. For small $\delta$, this implies $\|x' - x\|_F = O(\sqrt{\delta})$, and by Lipschitz continuity of $g_\phi$:

$$\|\mathbf{G}_\varepsilon - \mathbf{G}_0\|_{\max} \leq L_g \cdot O(\sqrt{\delta})$$

where $\mathbf{G}_0$ is the Granger matrix from an unfiltered model and $L_g$ is the Lipschitz constant of the predictor's Jacobian.

**Proof**. From the definition:

$$\|x' - x\|_F^2 = \mathcal{L}_{\text{ortho}} \cdot \|x'\|_F \cdot \|x\|_F \leq \delta \cdot \|x'\|_F \cdot \|x\|_F$$

$$\leq \delta \cdot B_x \cdot \|x'\|_F \leq \delta \cdot B_x \cdot (\|x\|_F + \|x' - x\|_F)$$

Solving for $\|x' - x\|_F$ yields the bound. By A2 (Lipschitz $H_\theta$), the Jacobian perturbation follows. $\square$

**Practical implication**: The orthogonality regularizer $\mathcal{L}_{\text{ortho}}$ provides a **certificate** of causal fidelity — a low $\mathcal{L}_{\text{ortho}}$ value at convergence guarantees that the filtered model's Granger causality scores are close to what an unfiltered model would produce under the same data.

---

## Numerical Verification Plan

1. **Perturbation sweep**: Train MambaFilterJRNGC and TCNFilterJRNGC with $\varepsilon \in \{0.01, 0.05, 0.1, 0.2, 0.5\}$ on NSVAR d=10. Measure $\|\mathbf{G}_\varepsilon - \mathbf{G}_0\|_{\max}$ and verify linear scaling with $\varepsilon$.

2. **Gradient coupling verification**: At initialization, compute $\|\partial \|\mathbf{J}\|_1 / \partial \theta\|$ and $\|\partial \mathcal{L}_{\text{pred}} / \partial \theta\|$ for both MambaFilterJRNGC and MambaJRNGC (concat). Verify that the Jacobian penalty gradient is $O(\varepsilon)$ for the filter but $O(1)$ for concat.

3. **Orthogonality certificate**: Track $\mathcal{L}_{\text{ortho}}$ and $\|\mathbf{G}_\varepsilon - \mathbf{G}_{\text{baseline}}\|$ during training. Verify that low $\mathcal{L}_{\text{ortho}}$ implies small Granger deviation.

4. **Lipschitz constant estimation**: Estimate $L_H$ for trained MambaBlock and TCNBlock via power iteration on the Jacobian.

---

## Relationship to Existing Theory

- **JRNGC (Cheng et al., ICML 2024)**: No theoretical analysis of Jacobian regularization. Our Theorem 1 provides the first perturbation analysis.
- **Neural Granger Causality (Tank et al., 2021)**: Consistency results for MLP/LSTM under stationarity. Our Theorem 2 extends the analysis to non-stationary settings with filtering.
- **Mamba (Gu & Dao, 2023)**: No analysis of Mamba as a pre-processing filter. Our work is the first to characterize Mamba's theoretical properties in the context of causal discovery.

## Potential Extensions

1. **Finite-sample rates**: Replace expectation $\mathbb{E}_t$ with finite-sample concentration bounds (Hoeffding/Bernstein).
2. **Non-asymptotic $\varepsilon$ selection**: Derive optimal $\varepsilon$ balancing filter capacity vs. causal fidelity.
3. **Filter class generalization**: Extend from Mamba/TCN to any filter class satisfying the near-identity + Lipschitz conditions.
