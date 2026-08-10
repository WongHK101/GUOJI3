# Phase 8 Final Bounded Repair Trade-off Execution Lock

Status: authorized final method-development cycle.

## Scope

- New repair strengths: 0.0003, 0.001, and 0.003.
- Pilot units: data seeds 12001-12003 and model seeds 22001-22002.
- New scientific records: 18 repair-only runs.
- Frozen comparator and lambda=0.01 records are read-only inputs from release
  `6f489b154253405448e08d4a63f566f7d875ef08`.
- The architecture, stratified nominal-plus-history estimator, schedule seed
  32001, optimizer, 2,000 iterations, final-checkpoint gate, score objects,
  metrics, and pilot thresholds are unchanged.

The final matrix contains no comparator records. Comparator reruns therefore
cannot be selected by the authorized orchestrator.

## Pilot decision

Each model-seed pair is averaged within its data seed before gate evaluation.
A lambda is eligible only if all semantic, compute, graph, coefficient, pure
MSE, comparator-safety, and missing-route gates pass. If several lambdas are
eligible, selection is ordered by:

1. smallest mean relative pure-MSE degradation versus concat;
2. largest mean AUROC improvement versus concat;
3. smaller lambda.

The frozen lambda=0.01 result is included in the trade-off summary without
rerunning it. If no lambda is eligible, method development stops permanently
for this paper.

## Conditional confirmation

Only a passing pilot aggregate can generate a confirmation release token. A
token freezes the selected lambda, source commit, pilot aggregate, config,
40-row matrix, estimator schedule, thresholds, and hashes. Confirmation uses
data seeds 13001-13005, model seeds 23001-23002, and four methods: baseline,
concat x-only, full auxiliary lc10, and the selected repair.

Confirmation classification is restricted to:

- `CONFIRMED_POSITIVE_REPAIR`;
- `CONFIRMED_SEMANTIC_BUT_NONCOMPETITIVE`;
- `NONCONFIRMED_REPAIR`.

No further lambda or architecture tuning is authorized after this cycle.

## Manuscript consequence

The independent manuscript revision must withdraw auxiliary-route dominance
and legacy fixed-target-MSE interpretations. It must report corrected
fixed-target interventions, five-seed capacity and coefficient replication,
the failure of total-score-only correction, and the final graph-prediction
trade-off. The canonical `istf_kbs.tex` and frozen v2.4 package remain
unchanged.
