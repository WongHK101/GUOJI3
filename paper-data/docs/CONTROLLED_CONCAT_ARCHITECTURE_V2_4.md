# Controlled Concat Architecture: Source-Verified Definition

## Scope

This definition covers only the frozen controlled concat diagnostics implemented
by `MambaJRNGC`. It was produced by read-only source inspection. It does not
define external-covariate models or every conditional Granger architecture.

## Distinct graph objects

1. **External covariate `c`:** describes data origin, not the graph object by itself.
2. **Conditional graph over `X` given `c`:** `c` may be deliberately conditioned upon and declared exempt, but that declaration must be explicit.
3. **Joint graph over `(X,c)`:** requires score columns and edge semantics for both variable blocks.
4. **Transform `c=g_phi(X)`:** partial `X` derivatives and the total raw-chain derivative are different objects.

The controlled diagnostics instantiate case 4.

## Verified implementation

| Property | Source-verified definition |
| --- | --- |
| Origin of `c` | `c_u=g_phi(X_{0:u})` is learned from the same raw sequence; it is not external. |
| `g_phi` | `MambaBlock(d_model=d,d_state=4,d_conv=4,expand=2,residual_scale=0.1)` followed by `Linear(d,d_cond)`. The block uses LayerNorm, cross-channel linear projections, SiLU, a causal selective recurrence, and a residual path. Here `d_conv` controls projection width in this minimal implementation; it is not a temporal convolution. |
| Temporal support | The selective recurrence is prefix-stateful, so `c_u` may depend on the full prefix `X_{0:u}`. State is not reset at a prediction-window boundary. For raw target `x_t`, the predictor uses only `c_{t-K:t-1}`; `x_t` and future values are excluded by the causal recurrence and window slicing. |
| Predictor input | Concatenate raw and auxiliary lag windows along the coordinate dimension, then flatten to `(d+d_cond)K`: `[X_{t-K:t-1};c_{t-K:t-1}]`. The target is raw `x_t`. |
| Prediction detach | None. Prediction loss backpropagates through `g_phi`. |
| x-only score | Detach both stored blocks from preprocessing, clone raw `X` with gradients enabled, hold `c` fixed, and compute `partial xhat / partial X`. Absolute values are averaged across windows before lag aggregation. This is not the total raw-chain derivative through `g_phi`. |
| x-only penalty | The same partial coordinate object, reduced by a blockwise mean absolute value and multiplied by `lambda_x`. The frozen implementation samples at most the first 100 windows. |
| Full auxiliary penalty | Enable gradients for the complete concatenated tensor and compute `lambda_x mean(abs(J_x)) + lambda_c mean(abs(J_c))`. |
| Graph score after penalty expansion | Unchanged: the class inherits the x-only partial graph extractor, so expanded penalty coverage does not make the graph score route-complete. |

## Read-only sources

- `E:\GUOJI\mamba_enhanced\src\mamba_jrngc_pilot.py`
- `E:\GUOJI\mamba_enhanced\src\minimal_mamba.py`
- `E:\GUOJI\mamba_enhanced\experiments\risk_mitigation_20260515\run_full_aux_penalty.py`
- `E:\GUOJI\mamba_enhanced\experiments\risk_mitigation_20260515\run_concat_posthoc_jacobian.py`

The package source manifest records their SHA256 values. No controlled-
architecture field required by v2.4 remains unresolved.
