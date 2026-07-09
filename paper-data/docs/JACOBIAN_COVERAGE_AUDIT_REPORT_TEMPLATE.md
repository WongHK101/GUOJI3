# Claim-Specific Jacobian Coverage Audit Report Template

Use one report for each stated directed Granger-predictive dependency claim. Do not infer a positive label from low prediction loss, graph metrics, or the absence of a known defect.

| Field | Required entry |
| --- | --- |
| Graph-score claim | `[State the intended directed Granger-predictive dependency interpretation.]` |
| Predictor inputs and transformations | `[List raw history, auxiliary variables, feature maps, filters, state-space blocks, normalization, and target timing.]` |
| `V_score` | `[Coordinates represented in the reported score.]` |
| Exact score derivative | `[Partial or total/raw-chain derivative, aggregation over windows/lags, orientation, diagonal treatment.]` |
| `V_penalty` | `[Coordinates covered by sparsity or causal regularization.]` |
| Exact penalty derivative | `[Derivative, weights, reduction, sampling estimator, and declared exemptions.]` |
| `P_pred` route ledger | `[For each architecture-declared route class r: rho_score(r), rho_penalty(r), exempt(r). Use scored/not-scored/unknown, penalized/unpenalized/unknown, and declared-exempt/not-exempt/not-applicable.]` |
| `M_coord` | `[Map each score column to one original source variable, or record the ambiguity.]` |
| `H_attr` | `[Retained horizon, transformation support/receptive field, truncation rule, and omitted-mass evidence.]` |
| Direct evidence | `[Artifact paths, scripts, configuration, data/train seeds, aggregation unit, and hashes.]` |
| Audit dimensions | `[Score-route completeness; penalty-route completeness; score-penalty alignment; coordinate and horizon validity.]` |
| Claim-specific label | `[CLAIM-COVERED / PARTIALLY COVERED / COORDINATE-AMBIGUOUS / HORIZON-TRUNCATED / UNASSESSED.]` |
| Unresolved items | `[Unknown routes, unavailable provenance, missing horizons, or unsupported mappings.]` |

The label applies only to the declared architecture, score claim, attribution horizon, and available evidence. It is not a causal-identifiability guarantee or a model-wide certificate.

