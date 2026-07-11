# Stratified Nominal-Plus-History Estimator

## Objective retained

For raw lag `h`, define

```text
F_h(theta) = 1 / (d_out d_src K)
             * sum_j sum_i mean_{u in U_h} |J[u,j,i,h]|.

R_lag(theta) = sum_{h=1}^{H_max} F_h(theta).
```

Track B retains this full lag-balanced objective, `K=1`, `H_max=499`, and
`lambda=0.01`.

## Frozen strata

```text
nominal: h=1
B1: 2..32    (size 31)
B2: 33..128  (size 96)
B3: 129..499 (size 371)
```

The historical stratum follows the deterministic cycle `B1,B2,B3`. Within
the selected stratum, one lag is sampled uniformly by a frozen balanced
permutation. The nominal and historical lags each receive an independently
sampled eligible target window. Two distinct outputs and all raw sources are
used for both lag evaluations.

For one sampled window and output set `O`:

```text
Fhat_h = 1 / (|O| d_src K) * sum_{j in O} sum_i |J[u,j,i,h]|.
```

At a step assigned to stratum `B_s`:

```text
Rhat_t = Fhat_1 + 3 |B_s| Fhat_h.
```

Across a complete three-step cycle:

```text
E[Rhat_t]
  = F_1 + 1/3 * sum_s 3 |B_s| E_{h uniform in B_s}[F_h]
  = F_1 + sum_{h=2}^{H_max} F_h
  = R_lag.
```

The formal importance weights are `93`, `288`, and `1113`. The estimator is
not claimed to equal the exact objective at any individual step.

## Exact-reference validation

The reduced fixture partitions its historical support into three nonempty
contiguous strata and uses 1,536 draws, exactly 512 complete cycles. Required
gates remain:

- relative objective error at most 0.05;
- parameter-gradient cosine at least 0.95;
- finite, nonzero predictor and preprocessor gradients;
- exact stratum frequency;
- balanced within-stratum lag frequency;
- exact `3|B_s|` importance weights;
- deterministic schedule reproduction.
