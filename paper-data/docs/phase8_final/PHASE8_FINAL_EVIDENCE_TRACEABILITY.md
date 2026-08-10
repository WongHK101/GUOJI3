# Phase 8 Final Evidence Traceability

## Authoritative frozen artifacts

| Evidence | Package-relative location | SHA256 / lock | Permitted claim |
|---|---|---|---|
| Track A five-pair aggregate | `artifacts/phase8_trackA_replication_dee0d30/replication_aggregate_and_gates.json` | `2a474e434937e567b49d16d37a7446789f4b4f94365923a6ce9afb5d27e82a9a` | Capacity and coefficient degradation replicated 5/5; fixed-target auxiliary dominance did not replicate 0/5 |
| Final bounded-lambda aggregate | `artifacts/phase8_final_lambda_tradeoff_78a85ac/lambda_tradeoff_aggregate_and_decision.json` | `505a7ab8462661695e6dcdf5d520cd536d3183455d1ae6908ca1ffa95608335a` | Bounded graph--prediction trade-off across tested strengths; no eligible lambda; no confirmation |
| Final 901 artifact manifest | `artifacts/phase8_final_lambda_tradeoff_78a85ac/phase8_final_server_sha256.txt` | 247 entries, all locally verified | Server-to-local artifact integrity |
| Frozen comparator pilot | `artifacts/phase8_recovery_execution_6f489b1/` | release-locked source/config/matrix/authorization inside directory | Baseline, concat, equal-lambda, lc10, and lambda=0.01 frozen comparisons |
| Frozen full auxiliary-penalty summary | `frozen_evidence/full_aux_jacobian_penalty.json` | `a3697feff2f4d6c309fc34b1aadd4cde04508ab317fc0aa532ce603b1f42b3af` | Controlled graph/coefficient mitigation; legacy `pred_loss` is total objective, not pure MSE |
| P0 semantic audit seeds 0--4 | `frozen_evidence/p0/` | hashes listed in package manifest | Partial/total and filtered/raw-chain score disagreement only |
| Stage 1a official aggregate | `frozen_evidence/stage1a/stage1a_aggregate_go_no_go.json` | `57c0eadace1dd67c33514314cd17374c394444b4653c6c4ff0d16fad6ceccf67` | CP semantic pass, performance/novelty fail, no Stage 1b |
| P1 bounded postmortem package | `frozen_evidence/phase7_stage1a_bounded_failure_analysis_v1.zip` | `cf98e6a017d1e4d1419c171cafd9302bd393dd180cbf93355e35b264c187bd3e` | Inconclusive postmortem; A3 uninterpretable |
| Frozen v2.4 manuscript package | external archival package, not overwritten | `97121b3f7340e51645aa71675216e87958154023a1afd29d57f5df42c36bb638` | Archival fallback only, not submission-ready after Phase 8 corrections |

## Final release locks

- Source commit: `78a85acd513fddde1744283c68f17e731692ba2e`.
- Source-manifest SHA256:
  `7ceb3f8bfb1ad17a3d270c3de3b05bd3a9e027bceb90eb95c1057c0cec35691d`.
- Lambda matrix SHA256:
  `98371964c3d3468f5f6db21cf7b6c6bf1aba28e72a06d41e270cec1883abeff1`.
- Lambda config SHA256:
  `6f9176cc9615dc070a58f0efdf5a4b44cf74659a2bc8dafdff7c2f03debd0d5b`.
- Jacobian schedule SHA256:
  `aa2bccc22dee24d1021d5b5542be23ff0f4e5b00b7d1e316f853f6d4b9feefb2`.
- Authorization SHA256:
  `00338a2ceda62e85e5633456fec323120b0755c13b82e29f08debcf230cda075`.

## Independent aggregation check

The server aggregate and the local re-aggregate deserialize to identical JSON
objects at every key and leaf. Their file hashes differ only because Python text
mode wrote LF on Linux and CRLF on Windows. Every gate, metric, selected-lambda
field, and method decision is identical.

## P0 audit file hashes

- seed 0: `74382059164e8a10368a1cd32f7f52b1dcd5caa40c0d5a41fc9a2f9a7c1e020f`
- seed 1: `0bab5b05fd64196a4acd03b82331e933ad7f62e79c1d2a84128e1d3c8bb99d5e`
- seed 2: `aa7fd84a16989bb9e5a22d4a30e84159cf97a84eb2bae5949cf138723e9a4cdf`
- seed 3: `d12bc3a15f224f9a4b2efc4f3a4fd14e27428ce2e637f98611ad8f111dda2e8e`
- seed 4: `d2d86fb0d56d2f5712a46e7562bfd95a446c89550a01f20370ae8330d1522ce9`
