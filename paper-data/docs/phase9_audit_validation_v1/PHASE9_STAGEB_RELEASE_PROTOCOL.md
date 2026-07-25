# Phase 9 Stage B Release Protocol

## Authorization basis

Stage A completed 56/56 CUDA records under commit
`7d73125ece4e98962a237a8dd1adb1ae119ada50` and returned
`UNLOCK_STAGE_B`.

- Stage A archived decision raw-byte SHA256 (Windows CRLF):
  `92786263afd97a299667968f0d73bdad3ee4fa02b503d3441b24e1b0fa63d384`
- Stage B execution-lock decision canonical-text SHA256 (LF):
  `68cff029f6d192260abce0567de5f46bb9de73bf5355b61e64dfa923f8406166`
- Stage A archive SHA256:
  `1e77e8f58c4759fa49f5f65016290ae60be18b556e03be8674c49c2ca60d5f99`
- Independent reaggregation reproduced the decision JSON byte for byte. The
  execution lock canonicalizes CRLF to LF before hashing so the same frozen
  JSON is accepted on Windows and Linux without weakening content validation.
- The user authorized autonomous progression after local scientific and
  execution-integrity review. This replaces the earlier procedural requirement
  for another external-advisor round; it does not alter any scientific gate.

## Frozen Stage B matrix

`PHASE9_AUDIT_STAGEB_AUTHORIZED_MATRIX.csv` contains exactly the 36 Stage B
rows already present in the prospective matrix. It differs from those sealed
rows only in `execution_authorized`, which changes from `false` to `true`.

- NetSim subjects: 16, 0, 30, and 10.
- MoCap segments: run `[728,1228)` and salsa `[3000,3500)`.
- Methods: baseline, Mamba concat, and causal TCN concat.
- Replicates: two pre-registered seeds per data unit and method.
- Training, checkpoint, score, horizon, intervention, and gate definitions are
  unchanged from Stage A.

## Execution order

1. Run the full CPU test suite from a clean release commit.
2. Deploy that exact commit, token, Stage A decision, and authorized matrix to
   AutoDL.
3. Verify every dataset SHA256 before loading arrays.
4. Run two fresh 20-iteration, three-method smoke roots using NetSim16 rep1.
5. Run `validate_phase9_stageb_smoke.py`.
6. Start the 36-record formal root only if the validator exits zero.
7. Run `aggregate_phase9_audit_stageb.py` from the same clean release.
8. Freeze and mirror the complete artifact before any manuscript use.

Smoke outputs are infrastructure diagnostics and are not scientific evidence.

## Confirmation rule

The Stage A gates remain unchanged, except each qualifying Stage B data unit
requires both 2/2 training replicates to satisfy the directional per-run
condition. Both Mamba and TCN must pass all architecture, horizon, and
integrity gates.

Only `CONFIRMED_AUDIT_GENERALITY` permits the bounded claim recorded in the
pre-registration. It does not establish improved graph recovery, causal ground
truth for MoCap, full-prefix attribution, or a successful repair method.
