# Phase 9 Stage B Manuscript Evidence Traceability

Date: 2026-07-26

## Stage B release and archive

| Item | Immutable value |
| --- | --- |
| Release source commit | `0397e8af27c4f396d7713b129e0d7307da732681` |
| Release bundle SHA256 | `f7b09806b09a6be7113ce025814f2b3ddaaab236a246008904b35da4fe754147` |
| Stage A gate canonical LF SHA256 | `68cff029f6d192260abce0567de5f46bb9de73bf5355b61e64dfa923f8406166` |
| Stage B archive | `E:\GUOJI\results_kbs\phase9_stageb\phase9_stageb_confirmation_0397e8a.tar.gz` |
| Stage B archive SHA256 | `35b0729dfd5d463a74812f686f1f100e70d53f66565964765d23e35781ae0c76` |
| Server decision copy | `E:\GUOJI\results_kbs\phase9_stageb\stageb_confirmation_decision_server_original.json` |
| Decision canonical LF SHA256 | `8d473d5aa67b32bc9d1f568b02934a6ea73d6ea9423234ce0d6a15b4ae0a4660` |
| Artifact manifest | `archival_extract_0397e8a\phase9_stageb_artifact_0397e8a_manifest.sha256` |
| Manifest validation | 685/685 files, zero missing or mismatched |

## Manuscript-location map

| Planned location | Claim or object | Exact source | Score/diagnostic semantics | Boundary |
| --- | --- | --- | --- | --- |
| Abstract; Introduction | Bounded cross-architecture and cross-domain reproduction | `stageb_confirmation_decision_server_original.json::allowed_claim` | Held-out audit-generality confirmation | Not graph-performance confirmation |
| Results: held-out validation | 36/36 complete and all semantic gates passed | `stageb_confirmation_decision.json`, `execution_summary.json` | Release-locked formal execution | Runtime/integrity facts are not scientific effect sizes |
| Fig. 3a; Results | Missing-route qualification 6/6 for both architectures | `architecture_gates.*.missing_route_qualifying_units` and `unit_records` | Off-diagonal L1 omitted-route mass divided by total raw-chain mass | Coordinate-scale dependent |
| Fig. 3b; Results | Positive fixed-target `mask_c` in 6/6 units | `architecture_gates.*.mask_c_qualifying_units` and `unit_records` | Pure prediction-MSE delta with clean raw target fixed | Route-use sensitivity, not causal contribution |
| Fig. 3c; Results | NetSim partial-total disagreement 4/4 for both architectures | `architecture_gates.*.netsim_discrepancy_units` and NetSim `unit_records` | Nominal partial-versus-total Pearson and exact-top-k Jaccard | Weak baseline graph operating point |
| Fig. 3d; Results | Coordinate mixing and bounded temporal support | `coordinate_entropy_unit_median`, `temporal_tail_unit_median`, `h64_h128_unit_minimum` | Source-coordinate entropy, mass outside nominal lags, H64/H128 measured ratio | Mass beyond H128 unassessed |
| Results/Discussion boundary | NetSim baseline median AUROC 0.495 | `graph_context.baseline_netsim_auroc_median` | Context only | `performance_gate=false` |
| Methods/Reproducibility | Exact subjects, segments, seeds, horizons, release lock | `config_snapshot.json`, per-run `config.json`, release protocol and frozen matrix | Reproducibility metadata | No post-hoc changes |

## Per-unit data source

The canonical per-unit plotting source is:

`stageb_confirmation_decision_server_original.json::unit_records`

It contains six held-out units per architecture:

- MoCap run segment `[728,1228)`;
- MoCap salsa segment `[3000,3500)`;
- NetSim subjects 0, 10, 16, and 30.

Each unit is the average of two preregistered Stage B training replicates.
Replicates remain available under:

`archival_extract_0397e8a\phase9_stageb_formal_0397e8a\runs\`

## Baseline control

The baseline has no auxiliary route. Across the formal artifact:

- partial-total nominal maximum absolute difference: `0`;
- missing-route maximum: `0`;
- all paired predictor initialization checks: `12/12`.

These checks establish evaluator specificity for the declared missing-route
object. They do not establish the accuracy of the inferred NetSim graph.

## Lorenz-96 strong-operating-point confirmation

| Item | Immutable value |
| --- | --- |
| Release source commit | `2637bb798a2bd09c4f60fe3ead32abbffec3b8ca` |
| Formal archive root | `E:\GUOJI\results_kbs\phase9_lorenz_901_confirmation\2637bb7\` |
| Archive SHA256 | `959916f42e58c61d632d5113b977a1907461b5b89991abac51b3f9baf8a53209` |
| Formal aggregate | `extracted\phase9_lorenz_formal_2637bb7\aggregate\formal_confirmation_summary.json` |
| Data-seed table | `extracted\phase9_lorenz_formal_2637bb7\aggregate\data_seed_level_metrics.csv` |
| Internal artifact manifest | 832/832 files verified |
| Formal design | 2 methods x 5 fresh data seeds x 2 fresh training seeds = 20 runs |
| Statistical unit | Data seed after averaging two training seeds |
| Dataset | Lorenz-96, `F=40`, `d=10`, `T=500`, nominal lag `K=1` |
| Decision | All 14 aggregate checks passed; manuscript evidence eligible |

| Manuscript object | Exact source | Semantics | Boundary |
| --- | --- | --- | --- |
| Strong baseline AUROC `0.8628 +/- 0.0177` | `data_seed_level_metrics.csv::baseline_total_nominal_auroc` | Total raw-chain nominal-lag score, sample SD across five data-seed units | One known-graph dynamical family |
| Concat pure-MSE reduction `41.6%` | `data_seed_level_metrics.csv::concat_vs_baseline_relative_mse` | Fixed raw target; Jacobian penalty excluded | Prediction evidence, not graph superiority |
| Concat total/partial AUROC `0.8589 / 0.8032` | Same CSV, corresponding nominal columns | Same checkpoint; total raw-chain versus x-only partial score | Total AUROC is not above baseline |
| Missing-route magnitude `0.5614` | Same CSV, `mamba_concat_missing_route_relative_magnitude` | Off-diagonal L1 omitted-route mass divided by total raw-chain mass | Coordinate-scale dependent |
| Auxiliary-mask delta `0.5275` | Same CSV, `mamba_concat_mask_c_fixed_target_mse_delta` | Clean raw target and raw-X route held fixed | Route-use sensitivity, not causal contribution |
| Pearson/Jaccard `0.8401 / 0.6577` | Same CSV, partial-total columns | Nominal direct-graph score discrepancy | Does not imply graph-performance gain |
| Horizon diagnostics | Same CSV and aggregate JSON | H=64/H=128 measured attribution stability | No completeness claim beyond H=128 |

The formal aggregate was repeated on the server and independently recomputed
locally. CSV outputs were byte-identical and JSON objects were canonically
identical. The result may be used only as a strong-operating-point
score-coverage case study. It cannot support Mamba/ISTF superiority, a
successful repair, universal invalidity of auxiliary conditioning, or strong
significance claims from five data-seed units.

## Integrity rules for derived manuscript assets

1. Derived CSV files must be generated only from the immutable decision JSON.
2. Every figure directory must retain a copy of the exact source JSON or a
   manifest entry pointing to its SHA256.
3. Unit ordering must be fixed as
   `NetSim-0, NetSim-10, NetSim-16, NetSim-30, MoCap-run, MoCap-salsa`.
4. Mamba and TCN values must not be pooled into a single architecture mean
   when an architecture-specific claim is made.
5. No graph metric is computed for MoCap.
6. No data beyond H=128 are inferred from the stored horizon ratios.
7. Lorenz-96 manuscript summaries must be recomputed from the five
   data-seed-level rows, not from 10 train-level runs treated as independent.
8. The Lorenz-96 total nominal score is the primary direct-graph object;
   x-only partial scoring is retained only as the route-coverage diagnostic.
