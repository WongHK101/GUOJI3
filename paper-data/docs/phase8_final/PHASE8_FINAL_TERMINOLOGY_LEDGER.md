# Phase 8 Final Terminology Ledger

| Canonical term | First-use definition / notation | Do not substitute |
|---|---|---|
| Jacobian coverage audit | Five-dimensional diagnostic framework for score routes, penalty routes, alignment, coordinates, and horizon | coverage certificate |
| architecture-declared predictive route classes | `P_pred` in the coverage declaration | exact enumeration of every effective neural path |
| baseline JRNGC | Legacy baseline using raw-history prediction and its declared Jacobian score | raw JRNGC unless explicitly contrasting coordinate domains |
| concat x-only | Conditioned predictor with an x-only score and x-only Jacobian penalty | coverage-complete concat |
| partial nominal score | Partial derivative with respect to raw X, aggregated over declared Granger lag K | total graph score |
| total nominal raw-chain score | Total derivative with respect to original raw X, aggregated over nominal lag K | full-history direct graph score |
| reliable-support raw-history attribution | Secondary maximum over pre-specified reliable historical support | direct Granger graph score |
| coverage-aligned full-prefix raw-chain regularization | Broad raw-chain penalty covering full attributed history | matching score-and-penalty regularization |
| fixed-target pure prediction MSE | MSE against the original unperturbed raw target, excluding Jacobian penalties | training objective or legacy `pred_loss` |
| best total objective | Minimum checked prediction MSE plus applicable Jacobian penalty | pure prediction loss |
| data-seed analysis unit | Mean of two model seeds within one data seed | six independent model-seed observations |
| graph--prediction trade-off | Bounded trade-off across the tested regularization strengths | efficient-set or method-effectiveness claim |
