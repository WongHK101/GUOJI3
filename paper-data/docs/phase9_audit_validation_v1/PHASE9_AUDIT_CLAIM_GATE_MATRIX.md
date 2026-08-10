# Phase 9 Audit Claim and Gate Matrix

| Claim target | Required evidence | Numeric gate | Failure wording |
| --- | --- | --- | --- |
| Raw-chain implementation is sound | CPU semantic suite and baseline negative control | all tests pass; baseline partial-total max diff and `M_missing <=1e-7` | implementation invalid; no scientific interpretation |
| Auxiliary route is actually used | fixed-target `mask_c` intervention | positive in at least 5/6 unit means per architecture; qualifying units have 2/3 positive seeds; median delta `>=0.05` | auxiliary-use generality not established |
| Omitted route is nontrivial | full-minus-partial raw-chain mass | `M_missing >=0.20` in at least 5/6 units per architecture; qualifying units have 2/3 passing seeds | missing-route generality not established |
| Partial score differs from total direct score | nominal partial-total discrepancy | median Pearson `<=0.90`; at least 3/4 NetSim units have Pearson `<=0.90` or Jaccard `<=0.80` | direct score discrepancy not reproduced |
| Auxiliary coordinates mix raw sources | condition-coordinate Jacobian entropy | median normalized entropy `>=0.75` per architecture | coordinate ambiguity not established |
| Older raw history contributes | bounded temporal tail mass | median per-unit tail mass `>=0.05` per architecture | historical route use not established |
| H=64 bounded audit is numerically adequate relative to H=128 | same-window horizon sensitivity | 5/6 unit ratios `>=0.99`, minimum `>=0.95`, nominal max diff `<=1e-7` | H=64 inadequate; stop and replan |
| Result is not Mamba-specific | independent causal TCN control | every Mamba and TCN gate passes | at most architecture-specific evidence |
| NetSim direct-graph context is interpretable | baseline graph score | median baseline AUROC `>=0.60` | graph context is weak-operating-point only |
| Stage A is reproducible | separate-root duplicate | score/loss diff `<=1e-6`, metric diff `<=1e-6`, identical top-k | Stage A fails determinism |
| Stage B can start | complete Stage A package | every semantic, generality, horizon, integrity, and runtime gate passes | AutoDL remains closed |
| Manuscript generality claim | independent Stage B confirmation | Stage A and Stage B both pass | development/validation evidence remains out of manuscript |

## Non-claims

No gate in this protocol can support:

- superiority of ISTF, Mamba, TCN, or filtering;
- a successful causal-discovery repair;
- full-prefix attribution completeness;
- direct causal graph validity on MoCap;
- universal validity across neural time-series architectures;
- a deployment or operating-regime claim.
