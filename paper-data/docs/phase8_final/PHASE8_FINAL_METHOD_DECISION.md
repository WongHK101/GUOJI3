# Phase 8 Final Method Decision

## Locked outcome

`STOP_METHOD_DEVELOPMENT_AUDIT_BOUNDARY_MANUSCRIPT`

The final bounded lambda pilot completed 18/18 new final-checkpoint runs with no
NaN/Inf, missing output, or release-lock failure. The frozen aggregate reused 30
comparator/previous-pilot runs. All four repair strengths passed the graph-effect,
missing-route, and semantic-compute gates, but none passed the complete pilot-go
rule.

| Lambda | Delta AUROC | Delta AUPRC | Delta coefficient r | Relative pure-MSE degradation | Other failure | Eligible |
|---:|---:|---:|---:|---:|---|---|
| 0.0003 | +0.072164 | +0.094945 | +0.143337 | +9.774% | One seed +20.262% MSE; baseline-safety AUROC -0.110548 | No |
| 0.001 | +0.139170 | +0.175411 | +0.209181 | +40.477% | Pure-MSE gate | No |
| 0.003 | +0.175262 | +0.225664 | +0.227613 | +74.227% | Pure-MSE gate | No |
| 0.01 | +0.189590 | +0.271267 | +0.246743 | +105.396% | Pure-MSE gate | No |

Effects are repair minus concat after averaging two model seeds within each of
three data seeds. The data seed is the statistical unit.

## Consequences

- No lambda was selected.
- Held-out confirmation was not eligible and was not executed.
- No further repair tuning is authorized for this paper.
- The repair can be reported only as a graph--prediction frontier and a boundary
  result, not as a competitive method.
- The manuscript must center on the Jacobian coverage audit, replicated
  prediction--knowledge decoupling, corrected intervention evidence, and the
  failure of total-score-only correction.

