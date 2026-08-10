# Recovery Implementation Changelog

- Replaced the repair-only lag sampler with the frozen nominal-plus-history
  stratified estimator.
- Kept the exact full lag-balanced objective and all scientific settings.
- Added separate nominal and historical penalty fields.
- Added per-step nominal/historical predictor and preprocessor gradient traces
  for the non-evidentiary 100-iteration benchmark.
- Replaced the invalid per-draw historical nonzero gate with the approved
  nominal and cumulative-stratum gates.
- Added fixed-lag float32/float64 forensics.
- Added support for an independently produced Track A replication aggregate
  when unlocking Track B pilot execution.
- Retained the exact 135-row run matrix and all pilot/confirmation thresholds.
- Did not modify any legacy comparator source.
