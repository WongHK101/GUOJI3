# Phase 8 Final Experiment and Manuscript Changelog

## Experiments

- Added only the pre-authorized repair strengths `0.0003`, `0.001`, and `0.003`.
- Ran 18 new records: three data seeds, two model seeds, and three lambdas.
- Reused the frozen comparator and lambda `0.01` runs without retraining.
- Completed all runs at 2,000 iterations and the final checkpoint.
- Aggregated model seeds within data seed before every gate.
- Found no eligible lambda; did not generate a confirmation release token and did
  not execute the 40-run held-out confirmation.

## Scientific interpretation

- Retained the 5/5 capacity-based prediction--knowledge decoupling result.
- Retained the 5/5 graph and coefficient-fidelity degradation result.
- Withdrew auxiliary-route dominance after the corrected intervention failed 0/5.
- Reported that raw history was more prediction-critical under fixed-target
  interventions.
- Reported total-score-only evaluation as a failed correction.
- Reported full-prefix regularization as a graph--prediction frontier, not a
  successful method.

## Manuscript

- Created an independent Phase 8 manuscript; canonical `istf_kbs.tex` and the
  frozen v2.4 archive were not overwritten.
- Replaced single-run capacity, intervention, and coefficient panels with
  five-pair plots showing seed points and population SD.
- Added the bounded repair trade-off and frozen gate matrix.
- Corrected the frozen full-penalty `pred_loss` label to best total objective.
- Retained legacy ISTF-Mamba only as a compact score-semantics diagnostic.
- Removed the redundant manuscript provenance table; detailed provenance remains
  in the package traceability document.
- Final LaTeX build: 16 pages, three passes, zero errors, undefined references,
  missing figures, and overfull boxes.

