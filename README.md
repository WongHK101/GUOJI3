# Jacobian Coverage Audits for Reliable Neural Granger Causality

This repository supports a reliability study of graph knowledge extracted by
Jacobian-regularized neural Granger-causality models. The current work is an
**audit and score-semantics project**, not an ISTF-Mamba performance paper.

The central question is whether the derivative reported as a Granger graph
covers the predictive routes, source-variable coordinates, and temporal
support actually used by the model. Auxiliary inputs and transformed histories
can improve prediction while bypassing or changing the Jacobian object used as
graph knowledge. The project therefore separates:

- score-route completeness;
- penalty-route completeness;
- score--penalty alignment;
- source-coordinate validity;
- attribution-horizon validity.

These checks diagnose the declared graph object. They do not certify causal
identifiability, causal sufficiency, or absence of hidden confounding.

## Current Status

Status date: **2026-08-10**.

The latest formal source is available on GitHub, but it is intentionally kept
on a frozen Phase 9 branch rather than silently replacing historical release
commits on the default branch.

| Role | Repository / branch | Immutable reference |
| --- | --- | --- |
| Current code and manuscript-support assets | [`GUOJI3`, `phase9/stageb-manuscript-support-v1`](https://github.com/WongHK101/GUOJI3/tree/phase9/stageb-manuscript-support-v1) | `ff2079fa1346c092c0e50990fd846f9291d99af0` |
| Phase 9 Stage A execution release | `GUOJI3` | `7d73125ece4e98962a237a8dd1adb1ae119ada50` |
| Phase 9 Stage B execution release | `GUOJI3` | `0397e8af27c4f396d7713b129e0d7307da732681` |
| Current KBS manuscript | [`KBS`, `phase9/stageb-manuscript-integration-v1`](https://github.com/WongHK101/KBS/tree/phase9/stageb-manuscript-integration-v1) | `d63bc31709d0fe16e9ca8860f139639f8353f223` |

The default `main` branch remains the project index and historical integration
branch. For scientific reproduction, use the exact commit associated with the
artifact or protocol being inspected.

## Scientific Position

### Supported by the current evidence

- An auxiliary predictive route omitted from an x-only Jacobian score and
  penalty creates a structural route-coverage vulnerability.
- Partial derivatives and total raw-chain derivatives can represent different
  graph objects.
- Controlled concat studies reproduce prediction--knowledge decoupling,
  coefficient-fidelity loss, and mitigation after expanding auxiliary-route
  coverage.
- Preregistered Phase 9 validation confirms bounded omitted-route and
  partial-versus-total score signatures across Mamba and causal-TCN
  preprocessors and across held-out NetSim and motion-capture data units.
- A separately frozen known-graph Lorenz-96 study shows that the audit signal
  is not restricted to a weak-baseline operating point.

### Explicit boundaries

- The preregistered Stage 1a coordinate-preserving depthwise repair passed its
  semantic checks but failed performance and novelty gates. Stage 1b was not
  started.
- Phase 8 coverage-aligned full-prefix regularization exposed a
  graph--prediction trade-off; no tested strength passed the joint gate.
- Legacy cross-channel ISTF-Mamba results may illustrate filtered-coordinate
  versus raw-chain score disagreement, but they are not valid evidence of
  graph-recovery superiority or Mamba effectiveness.
- Phase 9 Stage B supports bounded audit generality only. It does not establish
  universal architecture coverage, full-prefix completeness, or improved
  causal-graph recovery.

## Use the Frozen Source

Clone the repository and check out the current manuscript-support snapshot:

```bash
git clone https://github.com/WongHK101/GUOJI3.git
cd GUOJI3
git checkout ff2079fa1346c092c0e50990fd846f9291d99af0
git status --porcelain
```

The last command must return no output before a release-locked reproduction.
Do not substitute the current default branch for an execution-release commit.

## Repository Map at the Phase 9 Snapshot

```text
src/
  jacobian_coverage_audit.py       Machine-readable audit schema and profiles
  phase9_audit_generalization.py  Raw-chain and intervention audit utilities
  phase8_coverage.py              Partial/total derivative evaluation support
  mamba_jrngc_pilot.py            Legacy baseline and concat model definitions
  repaired_istf.py                Repaired-method and metric adapters

experiments/
  phase9_audit_stagea.py          Release-locked RTX 4090 Stage A runner
  aggregate_phase9_audit_stagea.py
  phase9_audit_stageb.py          Release-locked AutoDL Stage B runner
  aggregate_phase9_audit_stageb.py

configs/
  phase9_audit_stagea_v1.json
  phase9_audit_stageb_v1.json

paper-data/docs/
  phase9_audit_validation_v1/     Preregistration, matrices, gates, protocol
  phase9_stageb_manuscript/       Claim/evidence and theory/tool contracts

tests/
  test_jacobian_coverage_audit.py
  test_phase9_audit_generalization.py
  test_phase9_audit_stagea.py
  test_phase9_audit_stageb.py
```

The complete evidence and writing indexes are maintained in:

- [`paper-data/README-data.md`](https://github.com/WongHK101/GUOJI3/blob/phase9/stageb-manuscript-support-v1/paper-data/README-data.md);
- [`paper-data/README-paper.md`](https://github.com/WongHK101/GUOJI3/blob/phase9/stageb-manuscript-support-v1/paper-data/README-paper.md).

## Environment and Local Checks

The release code targets Python 3.10 and uses PyTorch, NumPy, SciPy,
scikit-learn, and pytest. CUDA is required only for the formal GPU protocols.
A minimal CPU environment is:

```bash
conda create -n jrngc_audit python=3.10
conda activate jrngc_audit
pip install torch numpy scipy scikit-learn pytest
```

Run the core audit and Phase 9 protocol tests from the repository root:

```bash
python -m pytest -q \
  tests/test_jacobian_coverage_audit.py \
  tests/test_phase9_audit_generalization.py \
  tests/test_phase9_audit_stagea.py \
  tests/test_phase9_audit_stageb.py
```

Generate the small non-evidentiary audit-format example with:

```bash
python paper-data/docs/phase9_stageb_manuscript/example_audit/generate_example.py
```

The generated example is an interface fixture, not a scientific result.

## Formal Reproduction

The formal Phase 9 runners require more than a command line: they enforce exact
source commits, release tokens, frozen matrices, dataset SHA256 values,
deterministic settings, and prior-stage decisions. Do not construct replacement
tokens or matrices for convenience.

Use these documents in order:

1. [`PHASE9_AUDIT_VALIDATION_PREREGISTRATION.md`](https://github.com/WongHK101/GUOJI3/blob/phase9/stageb-manuscript-support-v1/paper-data/docs/phase9_audit_validation_v1/PHASE9_AUDIT_VALIDATION_PREREGISTRATION.md)
2. [`PHASE9_AUDIT_CLAIM_GATE_MATRIX.md`](https://github.com/WongHK101/GUOJI3/blob/phase9/stageb-manuscript-support-v1/paper-data/docs/phase9_audit_validation_v1/PHASE9_AUDIT_CLAIM_GATE_MATRIX.md)
3. [`PHASE9_STAGEB_RELEASE_PROTOCOL.md`](https://github.com/WongHK101/GUOJI3/blob/phase9/stageb-manuscript-support-v1/paper-data/docs/phase9_audit_validation_v1/PHASE9_STAGEB_RELEASE_PROTOCOL.md)
4. [`EVIDENCE_TRACEABILITY.md`](https://github.com/WongHK101/GUOJI3/blob/phase9/stageb-manuscript-support-v1/paper-data/docs/phase9_stageb_manuscript/EVIDENCE_TRACEABILITY.md)

Formal datasets, checkpoints, and result roots are intentionally not stored in
Git. Their immutable archive paths and hashes are recorded in the evidence
index and traceability documents.

## Manuscript

The active manuscript is **Jacobian Coverage Audits for Reliable Neural
Granger Causality** in the separate
[`WongHK101/KBS`](https://github.com/WongHK101/KBS) repository. The current
English source, synchronized Chinese review edition, figures, appendix, and QA
assets are on branch `phase9/stageb-manuscript-integration-v1`.

The manuscript treats the audit framework as the contribution and reports
repair attempts as boundaries. It does not claim that ISTF, Mamba, TCN, or
full-prefix regularization universally improves graph recovery.

## Historical Material

The previous ISTF-Mamba performance-oriented README is preserved at
[`docs/LEGACY_ISTF_README.md`](docs/LEGACY_ISTF_README.md) for provenance. It
is not the current project description. Other historical result tables and
scripts remain useful for traceability but must be interpreted through the
current evidence boundaries.

## Maintenance Rules

- Do not overwrite frozen result roots or release manifests.
- Do not reinterpret development runs as confirmatory evidence.
- Keep score coordinates, target/source orientation, diagonal policy, and
  attribution horizon explicit in every derived result.
- Update `paper-data/README-data.md` when evidence is added or deprecated.
- Update `paper-data/README-paper.md` after each manuscript-writing session.
- Keep manuscript changes in the KBS repository and code changes in GUOJI3.
