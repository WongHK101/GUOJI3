# Causality-Preserving Input-Space Filtering: Design and Rationale

This document constitutes Section 3.2 of the paper: "Causality-Preserving Filter Design."
It can be integrated directly into the Methods.

---

## 3.2 Causality-Preserving Filter Design

We present a principled framework for incorporating temporal information into Jacobian-regularized neural Granger causality without introducing shortcut-learning vulnerabilities. Our design is governed by three principles, each motivated by a specific failure mode of prior approaches.

### 3.2.1 Structural Vulnerability of Prior Approaches

Jacobian-regularized GC (JRNGC; Cheng et al., ICML 2024) trains a predictive model $g_\phi: \mathbb{R}^{p \cdot d} \to \mathbb{R}^d$ with loss:

$$\mathcal{L} = \mathcal{L}_{\text{pred}}(g_\phi(\mathbf{x}_t), x_{t+1}) + \lambda_J \cdot \left\|\frac{\partial g_\phi}{\partial \mathbf{x}_t}\right\|_1 \quad (1)$$

The L1 penalty on the input Jacobian enforces sparsity in the causal graph: variable $j$ is inferred to Granger-cause variable $i$ if $\mathbb{E}_t[|\partial g_{\phi,i} / \partial x_{j,t-\tau}|] > \tau$.

When auxiliary temporal features $\mathbf{z}_t \in \mathbb{R}^{d_{\text{aux}}}$ are introduced via concatenation or feature-wise modulation (FiLM), the predictor becomes $\tilde{g}_\phi([\mathbf{x}_t; \mathbf{z}_t])$ and the Jacobian penalty is computed only on the original $d$ variables (since only these carry causal semantics). This creates an **information asymmetry**:

$$\frac{\partial \mathcal{L}_{\text{pred}}}{\partial \phi_{\text{aux}}} \text{ can reduce } \mathcal{L}_{\text{pred}} \text{ without affecting } \left\|\frac{\partial \tilde{g}_\phi}{\partial \mathbf{x}_t}\right\|_1 \quad (2)$$

Gradient descent exploits this asymmetry: predictive information is progressively routed through $\mathbf{z}_t$, allowing the Jacobian penalty on $\mathbf{x}_t$ to collapse to near-zero while maintaining low prediction error. The result is catastrophic degradation of causal discovery (AUROC $\to 0.5$), which we term **Jacobian shortcut learning**.

### 3.2.2 Principle 1: Input-Space Confinement

The root cause of shortcut learning is the existence of **auxiliary information channels** that bypass the Jacobian penalty. Our first principle eliminates this vulnerability at the architectural level:

> **Principle 1 (Input-Space Confinement).** All temporal processing must occur *within* the $d$-dimensional input space, without introducing new dimensions, latent codes, or modulation pathways that the Jacobian penalty does not cover.

Concretely, we define a filter $F_\theta: \mathbb{R}^{T \times d} \to \mathbb{R}^{T \times d}$ that transforms the raw time series before windowing:

$$x' = F_\theta(x), \quad \mathbf{x}'_t = (x'_{t-1}, \ldots, x'_{t-p}) \quad (3)$$

The predictor $g_\phi$ receives *only* $\mathbf{x}'_t$ as input. The Jacobian $\partial g_\phi / \partial \mathbf{x}'_t$ is computed on the *sole* information-carrying variables, and directly serves as the Granger causality score. There is no separate auxiliary path — every bit of information that reaches the predictor must pass through the Jacobian penalty.

**Contrast with prior approaches.** Concat-based architectures (MambaJRNGC, Section 4.1) create a $(d + d_{\text{aux}})$-dimensional input space with Jacobian penalty on only $d$ dimensions. FiLM-based architectures inject temporal information multiplicatively into hidden layers, bypassing the input Jacobian entirely. Our filter architecture maintains exact alignment between the information space and the penalty space.

### 3.2.3 Principle 2: Near-Identity Initialization

A filter that arbitrarily transforms the input could distort causal relationships even without shortcut learning. Our second principle ensures that training begins from a known-good configuration:

> **Principle 2 (Near-Identity Initialization).** At initialization, the filter must approximate the identity map, so the model starts from the standard JRNGC baseline.

We implement this via a residual architecture with damped initialization:

$$F_\theta(x) = x + \varepsilon \cdot H_\theta(x) \quad (4)$$

where:
- $H_\theta$ is the filter function (MambaBlock or TCNBlock)
- $\varepsilon = 0.1$ (`residual_scale`) controls deviation magnitude
- The output projection of $H_\theta$ is zero-initialized: $\mathbb{E}[H_\theta(x)] = 0$ at $t=0$

At initialization, $F_\theta(x) \approx x$, so the model is functionally equivalent to standard JRNGC. As training proceeds, the filter gradually learns to apply temporal smoothing, with the residual scale $\varepsilon$ controlling the rate and extent of deviation from identity.

**Theoretical guarantee.** Under Principle 2, Theorem 1 (Section 3.3) establishes that the Granger causality matrix from the filtered model, $\mathbf{G}_\varepsilon$, deviates from the end-to-end Jacobian $\tilde{\mathbf{G}}_\varepsilon$ by at most $\varepsilon \cdot L_H \cdot J_{\max}$ in max-norm. The filter's influence on causal discovery is explicitly controlled by the hyperparameter $\varepsilon$.

### 3.2.4 Principle 3: Orthogonality Regularization

Even with near-identity initialization, the filter could drift during training and distort causal structure. Our third principle provides an explicit training signal to prevent this:

> **Principle 3 (Orthogonality Regularization).** The filter deviation from identity must be explicitly penalized, providing a verifiable certificate of causal fidelity.

We introduce the orthogonality loss:

$$\mathcal{L}_{\text{ortho}} = \lambda_o \cdot \frac{\|x' - x\|_F^2}{\|x'\|_F \cdot \|x\|_F} \quad (5)$$

This normalized squared deviation penalizes the filter for producing outputs that differ from the input. The normalization by $\|x'\|_F \cdot \|x\|_F$ makes the penalty scale-invariant and comparable across datasets.

The total training objective becomes:

$$\mathcal{L} = \mathcal{L}_{\text{pred}} + \lambda_J \cdot \left\|\frac{\partial g_\phi}{\partial \mathbf{x}'}\right\|_1 + \lambda_o \cdot \mathcal{L}_{\text{ortho}} \quad (6)$$

**Certificate property.** Theorem 3 (Section 3.3) establishes that a low $\mathcal{L}_{\text{ortho}}$ value at convergence certifies small Granger causality deviation: $\mathcal{L}_{\text{ortho}} \leq \delta \Rightarrow \|\mathbf{G}_\varepsilon - \mathbf{G}_0\|_{\max} \leq O(\sqrt{\delta})$, where $\mathbf{G}_0$ is the Granger matrix from an unfiltered model. This provides a **computable, per-dataset guarantee** of causal fidelity without requiring ground truth.

### 3.2.5 Filter Instantiation: Generality through Architectural Diversity

To demonstrate that our framework does not depend on a specific filter architecture, we provide two instantiations with fundamentally different mechanisms:

**Mamba Filter (Selective SSM).** The MambaBlock (Gu & Dao, 2023) uses input-dependent state-space parameters $(\Delta, B, C)$ to perform selective temporal processing. With $O(L \cdot d)$ complexity via parallel scan, it adapts filtering behavior per time step — attenuating noise during non-stationary periods while preserving signal during stationary ones.

**TCN Filter (Causal Dilated Convolution).** The TCNBlock uses fixed-kernel causal Conv1d with dilation, following the WaveNet architecture (van den Oord et al., 2016). With the same $O(L \cdot d)$ complexity but **no input-dependent parameters**, it applies uniform temporal smoothing regardless of local signal characteristics.

Both filters share the identical interface: $(B, L, d) \to (B, L, d)$, and both satisfy Principles 1-3 (input-space confinement, near-identity init, orthogonality-compatible). Their mechanistic differences — selective (Mamba) vs. uniform (TCN), learned state-space vs. fixed convolution — provide a strong test of whether the **filtering architecture** or the **specific mechanism** drives performance improvements.

**Empirical finding.** Our ablation (Section 5.3) shows that TCN matches or exceeds Mamba on short sequences ($T < 200$) while Mamba dominates on long non-stationary sequences ($T = 1200$). Both outperform the unfiltered baseline on 5/7 datasets. This pattern confirms that the input-filtering architecture — not any specific filter mechanism — is the primary driver of improvement, with filter choice providing additional data-dependent gains.

### 3.2.6 Design Summary

| Principle | Motivation | Implementation | Verification |
|-----------|-----------|----------------|-------------|
| P1: Input-Space Confinement | Eliminate auxiliary information channels | $F_\theta: \mathbb{R}^{T \times d} \to \mathbb{R}^{T \times d}$, no $d_{\text{aux}}$ | Concat/FiLM baseline comparison |
| P2: Near-Identity Init | Start from JRNGC baseline | $F_\theta = I + \varepsilon \cdot H_\theta$, zero out-proj | Theorem 1: $O(\varepsilon)$ perturbation bound |
| P3: Orthogonality Reg. | Prevent causal drift during training | $\mathcal{L}_{\text{ortho}}$ penalizes $\|x'-x\|$ | Theorem 3: certificate property |

Together, these principles provide a **structurally guaranteed** approach to incorporating temporal information into Jacobian-regularized GC — one that cannot be exploited by gradient descent to bypass the causal discovery objective.
