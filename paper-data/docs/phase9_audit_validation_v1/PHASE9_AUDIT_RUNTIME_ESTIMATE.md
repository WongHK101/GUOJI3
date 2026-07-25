# Phase 9 Audit Validation Runtime Estimate

## Empirical RTX 4090 anchors

Development means per 1,000-iteration run:

| Data scale | Baseline | Mamba concat | TCN concat |
| --- | ---: | ---: | ---: |
| NetSim, `d=15,T=200` | 26 s | 45 s | 29 s |
| MoCap, `d=34,T=600` | 54 s | 83 s | 57 s |

Stage A primary training/evaluation estimate:

- four NetSim units x three seeds x approximately 100 s per method triplet:
  about 1,200 s;
- two MoCap units x three seeds x approximately 194 s per method triplet:
  about 1,164 s;
- total 54 primary records: about 0.66 sequential GPU-hours;
- two determinism duplicates: about 0.03 GPU-hours;
- H=32/64/128 checkpoint evaluation, manifests, and aggregation:
  conservatively 0.5 GPU-hours.

Expected end-to-end Stage A wall time is 1.5--2.5 hours. The frozen hard stop is
5 RTX 4090 GPU-hours. Exceeding the cap stops execution; the protocol may not
silently reduce methods, windows, horizons, or seeds.

Stage B contains 36 primary records and is expected to require below two
AutoDL GPU-hours including validation overhead. AutoDL remains off until Stage
A passes and the advisor authorizes Stage B.

These estimates are operational planning values, not scientific evidence.
