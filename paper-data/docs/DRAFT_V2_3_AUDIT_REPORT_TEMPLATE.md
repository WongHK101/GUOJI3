# Claim-Specific Jacobian Coverage Audit Report Template v2.3

## Purpose and boundary

Complete one report for each graph-score claim. The output is a diagnostic
audit profile, not a causal-identifiability certificate or a model-wide
guarantee.

## 1. Declared graph object

| Field | Required entry |
| --- | --- |
| Graph-score claim | State the directed Granger-predictive dependency interpretation. |
| Intended target variables | [Specify.] |
| Intended source variables | [Specify.] |
| Conditioning variables and exemptions | [Specify and justify.] |
| Edge orientation | [For example, `score[target, source]` means `source -> target`.] |

## 2. Five-part coverage declaration

\[
C=(V_{score},V_{penalty},P_{pred},M_{coord},H_{attr}).
\]

| Component | Required entry |
| --- | --- |
| `V_score` | Exact variables or coordinates and the derivative aggregated as the graph score. |
| `V_penalty` | Exact variables or coordinates, derivative, reduction, and weight used by regularization. |
| `P_pred` | Architecture-declared predictive route classes. Do not claim enumeration of every activated neural path. |
| `M_coord` | Mapping from every score column to an original source variable. |
| `H_attr` | Retained attribution horizon, transformation support, and truncation rule. |

## 3. Route ledger

For each architecture-declared route class, independently record score status,
penalty status, and exemption status. These are not mutually exclusive: a
route may be both scored and penalized.

| Route class | Score status | Penalty status | Exemption status | Evidence / rationale |
| --- | --- | --- | --- | --- |
| `[route]` | scored / not-scored / unknown | penalized / unpenalized / unknown | exempt / not-exempt / not-applicable | `[artifact, equation, or architecture declaration]` |

## 4. Five audit dimensions

Record each dimension as `PASS`, `FAIL`, `UNKNOWN`, or `NOT APPLICABLE` and
provide the evidence used for adjudication.

| Dimension | Audit question | Status | Evidence and unresolved items |
| --- | --- | --- | --- |
| Score-route completeness | Are all predictive route classes whose effects are interpreted as graph knowledge included in score attribution? | [status] | [evidence] |
| Penalty-route completeness | Are all route classes capable of carrying predictive information covered by regularization or explicitly declared exempt? | [status] | [evidence] |
| Score-penalty alignment | Do score and penalty cover compatible variables, coordinates, route classes, and horizons? | [status] | [evidence] |
| Coordinate validity | Does each score column map to one original source variable under the declared graph object? | [status] | [evidence] |
| Horizon validity | Does the attribution horizon cover support introduced by filtering, smoothing, transformations, or memory? | [status] | [evidence] |

## 5. Claim-specific audit profile

Construct a set containing every applicable flag:

- `CLAIM-COVERED`: use only when all applicable declared dimensions pass and
  no required status is unknown.
- `PARTIALLY COVERED`: route or alignment coverage is incomplete. This flag
  may coexist with `COORDINATE-AMBIGUOUS` and `HORIZON-TRUNCATED`.
- `COORDINATE-AMBIGUOUS`: the source-variable mapping is invalid or ambiguous.
- `HORIZON-TRUNCATED`: relevant or insufficiently audited support lies beyond
  the retained attribution horizon.
- `UNASSESSED`: report for every dimension that cannot be adjudicated from the
  available declaration, semantics, or provenance.

Do not force a single-label precedence rule. Report all applicable failure and
unassessed flags.

`Audit profile = { [flags] }`

## 6. Evidence and provenance

| Field | Required entry |
| --- | --- |
| Artifact paths and hashes | [Specify.] |
| Score-extraction code and version | [Specify.] |
| Architecture and configuration | [Specify.] |
| Seeds / replication unit | [Specify.] |
| Metric aggregation | [Specify.] |
| Remaining unknowns | [Specify.] |
| Permitted claim | [Specify.] |
| Forbidden interpretation | [Specify.] |
