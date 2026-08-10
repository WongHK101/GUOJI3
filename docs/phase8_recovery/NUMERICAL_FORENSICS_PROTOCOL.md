# Non-Evidentiary Numerical Forensics

The bounded audit profiles lags
`1,2,4,8,16,32,64,128,256,384,499` at:

- regenerated initialization;
- the frozen stopped-preflight iteration-20 checkpoint;
- the frozen stopped-preflight iteration-100 checkpoint.

For each state and lag, it uses the common final target window, all eight
outputs, and all raw sources. It records:

- blockwise absolute total raw-chain Jacobian mass;
- normalized mass;
- predictor and preprocessor gradient norms;
- exact-zero, subnormal, or normal status;
- eligible-window count;
- native-dtype and float64-accumulated gradient norms.

Both float32 and float64 model/data paths are evaluated when supported. The
only diagnostic labels are structural initialization zero, float32 underflow,
numerically nonzero but negligible, unresolved, and not-zero. Negligibility
is preregistered as at most `1e-12` of the same-state lag-1 magnitude.

The report is not used to shorten the horizon, select a stratum, tune lambda,
or support performance claims.
