# Phase 9 Literature and KBS Benchmark Audit

Search date: 2026-07-22

Workflow: targeted multi-source search using publisher/journal pages, official
conference proceedings, PubMed Central, and four local published KBS PDFs. This
is a decision-oriented benchmark audit, not a systematic review or a claim about
formal KBS acceptance requirements.

## Journal fit

The official KBS scope asks for original and innovative AI research, balanced
theory and practical study, and development or implementation of models,
methods, systems, and software that support knowledge-based prediction and
decision making:

- <https://www.sciencedirect.com/journal/knowledge-based-systems>

The present paper fits only if the extracted Granger graph is framed as a
knowledge object whose reliability depends on route, coordinate, and horizon
coverage. A manuscript framed as an unsuccessful ISTF performance method would
fit less well.

## Recent KBS comparison set

| Paper | Verified experimental pattern | Implication for this project |
|---|---|---|
| Hui et al., *Causal relationship analysis of high-dimensional time series based on quantile factor model*, KBS 284 (2024), 111263, <https://www.sciencedirect.com/science/article/pii/S0950705123010122> | Linear simulation systems at different dimensions, 100 Monte Carlo implementations in reported VAR experiments, two real datasets, and four classical/baseline comparisons | A KBS causal-time-series paper normally does not rely on one controlled construction alone |
| Chen et al., *Causal structure learning for high-dimensional non-stationary time series*, KBS 2024, DOI <https://doi.org/10.1016/j.knosys.2024.111868> | Ten simulation datasets, two real datasets, four advanced baselines, module ablations, and parameter-sensitivity analysis | Nonstationarity claims are supported across data regimes and by explicit ablation |
| Sun et al., *PCAC: Causal discovery from low-dimensional small-scale time series*, KBS 2025, <https://www.sciencedirect.com/science/article/pii/S0950705125011761> | Lorenz-96 plus NetSim, DREAM3, Sachs, and air-quality cases; five representative baselines; ablation and statistical testing | External known-graph and real-case breadth is a practical comparison point for the planned audit paper |
| Tao et al., *Active differentiable structure learning for clinical causal discovery*, KBS 327 (2025), 114145, <https://www.sciencedirect.com/science/article/abs/pii/S0950705125011864> | Four synthetic datasets, two public breast-cancer datasets, one collected clinical dataset, three continuous structure-learning baselines, ablation and sensitivity analysis | A knowledge-reliability claim is materially stronger when synthetic behavior and practical relevance are both shown |

Local full texts used to verify design details:

- `E:/GUOJI/KBSpaper/1-s2.0-S0950705123010122-main.pdf`
- `E:/GUOJI/KBSpaper/1-s2.0-S0950705124005021-main.pdf`
- `E:/GUOJI/KBSpaper/1-s2.0-S0950705125011761-main.pdf`
- `E:/GUOJI/KBSpaper/1-s2.0-S0950705125011864-main.pdf`

These four papers are a small purposive sample. Their dataset or baseline counts
are not formal minimums. They nevertheless show that the current synthetic-only
active mainline is empirically narrower than nearby accepted work.

## Neural Granger and temporal causal-discovery standards

| Paper | Relevant standard | Project implication |
|---|---|---|
| Zhou et al., *Jacobian Regularizer-based Neural Granger Causality*, ICML 2024, <https://icml.cc/virtual/2024/poster/34544> | A single-model Jacobian method is justified by extensive graph-recovery experiments, scalability, and comparison with state of the art | Criticism of JRNGC extensions must be equally precise about the architecture and graph-score object |
| Tank et al., *Neural Granger Causality*, TPAMI 44(8), 2022, <https://pmc.ncbi.nlm.nih.gov/articles/PMC9739174/> | Linear VAR and nonlinear Lorenz simulations, five DREAM3 networks, and human motion capture; structured penalties are tied directly to the graph interpretation | DREAM3 and MoCap are defensible external anchors, but MoCap has no complete edge ground truth |
| Gong et al., *Rhino: Deep Causal Temporal Relationship Learning with History-dependent Noise*, ICLR 2023, <https://openreview.net/forum?id=i_1rbq8yFWC> | Identifiability assumptions, extensive synthetic evaluation, real benchmarks, and misspecification ablations | The Phase 9 theory must expose assumptions and boundary cases, not only an existence counterexample |
| Cheng et al., *CUTS+: High-Dimensional Causal Discovery from Irregular Time-Series*, AAAI 2024, <https://ojs.aaai.org/index.php/AAAI/article/view/29034> | Simulated, quasi-real, and real datasets; scalability and irregular-sampling evaluation | External validation should test whether coverage auditing survives non-ideal temporal data, even if irregular sampling is not a primary claim |

## Reliability-audit standards

| Paper | Relevant standard | Project implication |
|---|---|---|
| Adebayo et al., *Sanity Checks for Saliency Maps*, NeurIPS 2018, <https://proceedings.neurips.cc/paper_files/paper/2018/hash/294a8ed24b1ad22ec2e7efea049b8737-Abstract.html> | An actionable methodology, negative and positive controls, multiple explanation methods/architectures/datasets, and supporting theory | The coverage framework must produce executable tests whose outcomes distinguish known-valid and known-invalid fixtures |
| Lapuschkin et al., *Unmasking Clever Hans predictors and assessing what machines really learn*, Nature Communications 10, 1096 (2019), <https://www.nature.com/articles/s41467-019-08987-4> | Multiple domains, direct interventions, and a semi-automated analysis show that predictive metrics can miss invalid strategies | Prediction--knowledge decoupling is publishable as a reliability result only if the diagnostic generalizes beyond one hand-built example |

## Evidence-based acceptance implications

The comparison set supports five conclusions.

1. **Scope fit is necessary but insufficient.** The paper's knowledge-reliability
   framing fits KBS, but current external evidence is too narrow for a safe
   submission.
2. **A negative repair result is not fatal.** Reliability papers can be strong
   without a successful new predictor, provided the audit is executable,
   discriminative, and broad.
3. **Checklist-only novelty is vulnerable.** Chain-rule decomposition and labels
   must be backed by software, positive/negative controls, and prospective
   validation.
4. **Real data have a different role from known-graph benchmarks.** DREAM3 or
   NetSim can evaluate direct graph recovery; MoCap can demonstrate audit and
   stability but cannot certify true causal edges.
5. **A fixed mitigation would materially reduce reviewer risk.** Replicating
   full-route regularization in another regime is sufficient; a new adaptive
   filter is not required.

## Literature gaps to close before manuscript revision

- attribution faithfulness and randomization/sanity-test literature;
- causal-representation and source-coordinate identifiability literature;
- temporal attribution horizon and stateful-model explanation literature;
- software/model-card style audit reporting;
- recent KBS work on trustworthy knowledge extraction and causal inference.

References added later must be verified against publisher or proceedings
metadata. Low-reputation venues are not needed to support the mainline.

