# Paper Writing Framework — ISTF-Mamba (TNNLS)

**Last updated:** 2026-06-12 (Phase 5 charting complete — Figures 2/3/4/S1 generated)
**Status:** Phase 5 root-cause synthetic experiments archived. Figures 2/3/4/S1 drafted (SVG+PDF+PNG). Paper narrative locked per v4 validation report. Writing phase pending.
**Target venue:** IEEE Trans. Neural Networks and Learning Systems (TNNLS)
**Main .tex:** `IEEE-Transactions-LaTeX2e-templates-and-instructions/istf_jrngc.tex`

### Manuscript Tracker (2026-06-12) — Root-Cause Results Integrated

| Part | Status |
|------|--------|
| Abstract | final candidate |
| Sections 1–11 | final candidate |
| Figure 2 (root-cause main) | **drafted (2026-06-12)** |
| Figure 3 (checkpoint dynamics) | **drafted (2026-06-12)** |
| Figure 4 (CausalTime boundary) | **drafted (2026-06-12)** |
| Figure S1 (negative controls) | **drafted (2026-06-12)** |
| Appendix A–D | final candidate |
| LaTeX structure | passed |
| Full manuscript | **needs root-cause section + figures** |
| Submission hygiene audit | **passed (2026-05-15)** |
| Fig. 1 (architecture) | **inserted (2026-05-15)** |
| NOTEARS sentence fix | **fixed (2026-05-15)** |
| Phase 5 results | **archived v4 (2026-06-11)** |

---

## 1. Paper Title

> **Shortcut Learning in Jacobian-Regularized Granger Causality: Diagnosis and Repair via Input-Space Temporal Filtering**

## 2. Core Narrative

The paper has a **three-act structure**, NOT a "Mamba beats baseline" story:

### Act I — Diagnosis (Sections 3 + 5)
We identify a **structural shortcut learning** vulnerability in JRNGC: when auxiliary temporal channels (concat, FiLM) are added for non-stationarity, gradient descent routes predictions through unpenalized side channels, suppressing the Jacobian entries used for GC discovery. We prove existence theoretically and validate with controlled diagnostic experiments.

### Act II — Repair (Section 4)
We propose **Input-Space Temporal Filtering (ISTF)** — all temporal processing confined to the original d-dimensional input space, eliminating unpenalized side channels. Three principles: (i) input-space confinement, (ii) near-identity initialization, (iii) orthogonality regularization. Instantiated with Mamba (selective SSM) and TCN (temporal convolution).

### Act III — Operating Boundary (Sections 6-8)
ISTF is NOT a universal booster. We characterize **when** it helps, when it is neutral, and when it degrades:
- **Helps**: Real-world non-stationary with sufficient T/d (CT_medical: 5/7 metrics Holm-significant; F1 NOT significant)
- **Neutral in near-ceiling regimes**: Lorenz_F40 (chaotic nonlinear, all p>0.3)
- **Degrades on stationary linear VAR**: VAR_d50 (AUPRC and F1 significantly worse; AUROC directional only, adj p=0.0586)
- **Degenerate**: NSVAR_d10 (too few effective pairs for valid Wilcoxon on most metrics; ceiling effect)
- **Near-identity does NOT guarantee no degradation**: VAR_d50 is stationary linear but shows significant negative results

---

## 3. Paper Structure (9 Sections)

### Section 1: Introduction (~1.5 pages)
- Granger causality background + neural GC
- The problem: JRNGC's Jacobian penalty is the causal discovery mechanism, but auxiliary channels bypass it
- **Shortcut learning** as the formal framing
- Three contributions: Diagnosis, Repair, Empirical validation with operating boundary
- **Critical positioning sentence**: "We do NOT claim ISTF as a universal SOTA causal discovery method. It is a structural repair for JRNGC with well-characterized operating boundaries."

### Section 2: Related Work (~1 page)
- Neural Granger causality (cMLP, cLSTM, eSRU, JRNGC)
- Temporal conditioning architectures (concat, FiLM, attention) and their information asymmetry
- Shortcut learning literature (Geirhos et al.) — our shortcut is *structural*, not dataset-driven
- SSMs and Mamba — our use as pre-filter in input space is novel

### Section 3: Shortcut Learning in JRNGC (~2 pages)
- **3.1 Background**: JRNGC formulation — prediction MLP, Jacobian penalty, GC graph recovery
- **3.2 The Auxiliary-Channel Shortcut**:
  - Formal definition of concat and FiLM architectures
  - Theorem 1: Existence of zero-Jacobian, low-loss solutions under concat
  - Theorem 2: Gradient coupling asymmetry
  - Mechanism: gradient descent stays in the low-Jacobian basin
- **3.3 Implications**: This is structural, not data-dependent.

### Section 4: Input-Space Temporal Filtering (~2 pages)
- **4.1 Design Principles**: input-space confinement, near-identity initialization, orthogonality regularization
- **4.2 Filter Instantiations**: ISTF-Mamba, ISTF-TCN
- **4.3 Theoretical Guarantees**: Theorems 3-5

### Section 5: Diagnosis Experiments (~1.5 pages)
- Structural diagnostics: d_cond sweep, coefficient recovery, mask/shuffle, auxiliary penalty
- These are architecture-level diagnostics — stable across v1→v2

### Section 6: Benchmark Evaluation (~2 pages) — **ACTIVE DRAFTING with frozen v2 data**
- **6.1 Experimental Setup**:
  - 11 datasets: 4 inferential (≥5 paired seeds) + 7 descriptive-only
  - Up to 6 methods, with missing entries where a method was not run
  - 7 metrics: AUROC, AUPRC, SHD_topk, nSHD_topk, MCC_topk, F1, Acc
  - Top-k threshold at k=|E_true| for SHD family metrics
  - All baseline/mamba metrics from same GC score matrix (two-stage pipeline)
  - Wilcoxon signed-rank test (paired by seed), Holm-Bonferroni correction (m=4)
- **6.2 Main Results**: Inferential subset only in main text; full table in appendix
- **6.3 Statistical Significance**: Per expert-reviewed Holm results
- **6.4 Descriptive Results**: 7 datasets, no inference drawn

### Section 7: Operating Regime Analysis (~1.5 pages) — **ACTIVE DRAFTING with frozen v2 data**
- **7.1 Factorial Operating Boundary**: 2×2 factorial, three-layer model, directional trends only (no comparison Holm-significant at n=10)
- **7.2 Filter Selection Heuristic**: Practical guidance, not formal test
- **7.3 Design Principle Verification**: Near-identity ablation, orthogonality ablation, filter type comparison

### Section 8: Discussion (~1 page) — **ACTIVE DRAFTING with frozen v2 data**
- Correct per-dataset conclusions (see Section 4 below)
- Limitations
- Open questions

### Section 9: Conclusion (~0.5 pages)

### Appendices
- **Appendix A**: Full 11-dataset × up-to-6-method result tables
- **Appendix B**: Statistical tests with raw p, Holm adjusted p, family size, and rank
- **Appendix C**: Dataset eligibility classification (4 inferential + 7 descriptive)
- **Appendix D**: Score manifest and metric reproducibility
- **Appendix E**: Factorial full data
- **Appendix F**: Lag/top-k/self-edge evaluation protocol (DRAFT COMPLETE; integrated for appendix)

---

## 4. Key Messages Per Dataset (v2 Data, Post-P0 Cleanup)

### 4.1 Inferential (n ≥ 5 paired seeds, Holm-corrected, m=4)

| Dataset | Direction | Significant (Holm) | Main-text message |
|---------|-----------|-------------------|-------------------|
| CT_medical | Mamba > Baseline | AUROC, AUPRC, SHD, nSHD, MCC (5/7) | "Statistically reliable but modest absolute improvement on real non-stationary medical data. F1 is NOT Holm-significant (adj p=0.111)." |
| Lorenz_F40 | Near-neutral | 0/7 | "Approximately neutral on chaotic nonlinear dynamics with sufficient T/d. Near-identity initialization limits degradation in this near-ceiling regime." |
| VAR_d50 | Mamba < Baseline | AUPRC, F1 (2/7, negative) | "ISTF-Mamba significantly underperforms baseline on stationary linear VAR for AUPRC and F1. AUROC is directional only (adj p=0.0586, NOT significant). Near-identity initialization does NOT guarantee no degradation in score ranking." |
| NSVAR_d10 | Degenerate/Ceiling | N/A (p=NaN) | "5 paired seeds but most metrics have too few non-zero paired differences for valid Wilcoxon: topology metrics n_eff=0; AUROC/AUPRC n_eff=4; F1 n_eff=2. Included in correction family conservatively but no inferential claims made. Ceiling-like neutral behavior, not simply 'non-significant'." |

### 4.2 Descriptive (1-3 seeds, no inference)

| Dataset | Seeds | Message |
|---------|-------|---------|
| fMRI_d15 | 3 | PCMCI+ better matches conditional-dependence structure; ISTF not suited for this regime |
| DREAM3 (d10,d50,d100) | 3 | Severely sample-limited (T=21); all methods near chance. TCN topology metrics unavailable (legacy limitation). |
| CT_pm25 | 1 | **Descriptive-only, mixed**: Mamba worsens AUROC relative to baseline (0.468→0.410) and AUPRC is near-identical. Topology metrics show mixed or modest improvement. Not a clear positive case. |
| CT_traffic | 1 | **Descriptive-only**: ISTF-Mamba AUROC below baseline (0.408→0.389); PCMCI+ is descriptively stronger (0.494). |
| NSVAR_d50 | 3 | Mamba neutral to slightly worse; insufficient seeds for inference. Non-stationary VAR medium-scale. |

---

## 5. Critical Wording Constraints

### Use:
- "structural safeguard" / "deployment framework" / "operating boundary"
- "prevents collapse" / "preserves causal signal"
- "directional trend" / "effect size" / "confidence interval" (for non-significant results)
- "statistically reliable but modest" (for CT_medical)
- "near-neutral in near-ceiling regimes" (NOT "stationary neutral")
- "ceiling-like neutral / degenerate" (for NSVAR_d10)
- "near-identity initialization limits degradation, but does not guarantee no degradation in score ranking"

### AVOID:
- "universal improvement" / "SOTA" / "outperforms"
- "statistically significant" (without Holm qualification)
- "proves" / "demonstrates conclusively" / "confirms"
- "mamba is better/worse" (use "ISTF-Mamba shows directional trend")
- "significant" when only raw p < 0.05 but Holm adj p > 0.05
- "stationary neutral" / "near-identity ensures no harm" (VAR_d50 is a counterexample)
- "CT_pm25 / CT_traffic positive" (both are mixed/negative on AUROC)

---

## 6. Key Data References

| Purpose | File | Location |
|---------|------|----------|
| All benchmark metrics | migrated_all_v2.json | `paper-data/canonical_results/` |
| GC score matrices (208 files) | *_gc.npy, *_gt.npy | `results/scores/` (not in paper-data/) |
| Score manifest (sha256) | manifest.json | `paper-data/scores_manifest/` |
| Statistical tests (Holm) | statistical_tests.csv/.tex | `paper-data/tables/` |
| Dataset eligibility | dataset_eligibility.csv/.tex | `paper-data/tables/` |
| 7-metric appendix tables | appendix_*.csv/.tex | `paper-data/tables/` |
| Factorial canonical | factorial_ablation_canonical.json | `paper-data/factorial/` |
| Factorial diagnostics | diagnostics_*.json | `paper-data/factorial/` |

---

## KBS Route-B Audit Pivot Override (2026-07-09)

Current writing direction for the KBS manuscript has pivoted away from an ISTF performance-method paper. The active planning direction is:

> Jacobian coverage audits for reliable neural Granger causality.

This route treats the paper as a mechanism/audit contribution about prediction-knowledge decoupling in Jacobian-regularized neural Granger causality. It does not claim that ISTF-Mamba, CP-depthwise, Mamba, TCN, or any learned filter is a successful benchmark method.

New planning assets:

- `paper-data/docs/kbs_jacobian_coverage_claim_evidence_matrix_2026-07-09.md`
- `paper-data/docs/kbs_jacobian_coverage_manuscript_skeleton_2026-07-09.md`
- `E:\GUOJI\elsarticle\istf_kbs_coverage_audit_skeleton.tex`

Locked claim boundaries:

- Use `Jacobian coverage audit framework`, not `coverage certificate`.
- Use Stage 1a as semantic-pass/performance-fail boundary evidence.
- CP-depthwise remains disqualified from Stage 1b and KBS main-method status.
- Legacy ISTF-Mamba can be used only as historical shortcut/score-semantics diagnostic, not as graph-recovery evidence.
- P1 A3 wording is limited to: `A3 was uninterpretable and did not pass because gradient_replay_alignment_valid=false`.
- Seeds 4 and 5 are not strict future confirmatory candidates; seeds 6-8 require sealed-boundary audit before any use, with fresh seeds preferred.

Preferred KBS rebuild strategy:

- Do not patch the old ISTF-performance narrative line by line.
- Rebuild from a new skeleton and migrate only valid equations, diagnostics, figures, tables, and literature positioning.
- Keep the canonical submitted draft `E:\GUOJI\elsarticle\istf_kbs.tex` unchanged until the advisor approves the new skeleton migration.

## 7. Writing Progress Tracker

### KBS Status Override (2026-07-05)
- Fig. 1 v9 is now promoted into the active KBS manuscript at `E:\GUOJI\elsarticle\istf_kbs.tex`.
- Introduction has been updated with causal-discovery assumption citations.
- Related Work reference expansion pass 2 is complete: 33 cited/used bibliography entries covering neural/time-series GC, shortcut learning, state-space/TCN filtering, temporal causal discovery, and KBS related work.
- Current KBS compile status: 26 pages, 0 LaTeX errors, 0 undefined references/citations, 0 missing figures, 0 Overfull hbox.
- Narrative unification pass is complete for Abstract, Results, Discussion, and Limitations. It preserved frozen figures and all numeric/statistical claims.
- Language-unification polishing is complete across the full manuscript. The pass tightened long sentences, reduced rhetorical/over-strong phrasing, and preserved all numbers, statistical language, figure assets, and claim boundaries.
- Reviewer-style KBS pre-submission audit is complete and archived at `paper-data/kbs_reviewer_audit_2026-07-05.md`.
- Writing-strategy constraint: audit-identified risks are for internal triage. The manuscript should not automatically expose every weakness in defensive prose; only selective framing edits that improve the main argument and prevent overclaim should be added.
- Selective framing edits are complete in the active KBS manuscript: the Introduction/Conclusion now foreground the graph as the extracted knowledge object, FiLM wording is bounded as analogous risk, CT\_medical is framed as operating-regime evidence, and the main-text CT\_medical side-channel self-disclosure was removed.
- Current KBS compile status after selective framing: 26 pages, 0 LaTeX errors, 0 undefined references/citations, 0 missing figures, 0 Overfull hbox.
- Reference expansion is complete as of 2026-07-05: 45 bibitems, selected from verified high-quality/relevant sources (ACM CSUR, Proceedings of the IEEE, ICML, ICLR, NeurIPS, AAAI, PMLR/CLeaR, and recent KBS). The expansion supports the existing ISTF/JRNGC narrative and does not change experiments, figures, numerical results, statistical claims, or conclusion strength.
- Current KBS compile status after reference expansion: 27 pages, 0 LaTeX errors, 0 undefined references/citations, 0 missing figures, 0 Overfull hbox.
- Next writing focus: final internal consistency pass across Abstract, Introduction, Related Work, Discussion, Limitations, and Conclusion after the expanded literature positioning.
- Method-pivot status as of 2026-07-07: KBS text remains frozen while P0 repaired-method readiness is audited locally. P0.3b locks D2 `s0=0.075`, primary checkpoint `500`, CP-depthwise as the only repaired main candidate, FixedFIR3 as matched baseline, FixedEMA as full-H reference, and RawChainMamba as limited diagnostic. Development seed `data_seed=0` remains non-evidentiary and must not be cited as paper performance.
- Execution-readiness status as of 2026-07-07: P0.3c adds the frozen Stage 1a runner, aggregation script, config validation, exact run manifest, resume/atomic-write checks, and a Stage 1-scale CPU smoke. These are execution artifacts only; do not revise manuscript claims until approved Stage 1a/1b results exist.

| Section | Status | Notes |
|---------|--------|-------|
| 1. Introduction | DRAFT (v1, 3-seed data) | CLEARED for rewrite — update with v2 data, corrected narrative |
| 2. Related Work | DRAFT (v1) | CLEARED for minor updates |
| 3. Shortcut Learning (Theory) | DRAFT (v1) | CLEARED — theorems are solid, minor polish only |
| 4. ISTF Design + Theory | DRAFT (v1) | CLEARED — Theorem numbering per expert; filter drift certificate framing |
| 5. Diagnosis Experiments | DRAFT (v1) | CLEARED — structural diagnostics stable across v1→v2 |
| 6. Benchmark Evaluation | **ACTIVE DRAFTING** | Full pre-freeze approved; drafting with frozen v2 data |
| 7. Operating Regime Analysis | **ACTIVE DRAFTING** | Full pre-freeze approved; drafting with frozen v2 data |
| 8. Discussion | **ACTIVE DRAFTING** | Full pre-freeze approved; drafting with frozen v2 data |
| 9. Conclusion | DRAFT (v1) | Minor updates; finalize after Section 6-8 |
| Appendix A (tables) | GENERATED | 11 datasets × up to 6 methods; note TCN DREAM3 topology gaps |
| Appendix B (statistical) | GENERATED | 28 test rows, m=4, updated post-PlanA removal |
| Appendix C (eligibility) | GENERATED | 4 inferential + 7 descriptive |
| Appendix D (score manifest) | DRAFT | Deterministic claim weakened per expert |
| Appendix E (factorial full) | DRAFT | Compressed version ready; full in posthoc summary |
| Appendix F (protocol) | DRAFT COMPLETE (v3) | Full pre-freeze approved; top-k index corrected, summary Granger graph naming; ready for appendix |

### Writing Priority Order (per expert instruction)
1. **P0 fixes** (PlanA removal, old doc deprecation, audit update) — DONE
2. **P1 fixes** (README corrections, encoding, registry, claims) — DONE
3. **Appendix F** — DRAFT COMPLETE; integrated for appendix
4. **Section 1-5** — ACTIVE DRAFTING (full pre-freeze approved)
5. **Section 6-8** — ACTIVE DRAFTING with frozen v2 data (full pre-freeze approved)
6. **Figures** — generate as sections are written
7. **Expert final pre-freeze audit** — APPROVED 2026-05-13; full pre-freeze granted
8. **Final claim audit** — conduct before manuscript submission

---

## 8. Figures to Generate

### KBS Figure Status Override (2026-07-05)
- Fig. 1 active manuscript asset: `E:\GUOJI\elsarticle\figures\fig1_istf_architecture_v9.pdf`.
- Fig. 2, Fig. 3, Fig. 4, Fig. 5, Fig. 6, and Fig. A.1 remain frozen from the passed Phase 6 figure/format audit unless the user explicitly reopens them.

| Figure | Content | Status |
|--------|---------|--------|
| Fig 1 | ISTF architecture diagram: shortcut vs repair | **UPDATED v4** (2026-06-23; independent x/c inputs) |
| Fig 2 | d_cond sweep, mask/shuffle, coefficient fidelity | **UPDATED v2** (2026-06-23; shared legend, readable log ticks, single-axis fidelity) |
| Fig 3 | Root-cause synthetic main results | **UPDATED v2** (2026-06-23; point/error grid, no hatch-heavy bars) |
| Fig 4 | JRNGC checkpoint dynamics | **UPDATED v2** (2026-06-23; loss-F1 trajectory, no dual y-axis) |
| Fig 5 | Main benchmark: AUROC bar chart (4 inferential) | **UPDATED v2** (2026-06-23; publication labels, caption-only significance key) |
| Fig 6 | CausalTime operating boundary | **UPDATED v2** (2026-06-23; two-panel boundary plot, no caveated/hatch layout) |
| Fig A.1 | Root-cause EMA negative controls | **UPDATED v2** (2026-06-23; null p99 + time-shuffle controls) |

---

## 9. Expert Review Log

| Date | Reviewer | Key Feedback | Action Taken |
|------|----------|-------------|--------------|
| 2026-05-09 | External Expert | Full paper audit (~5000 words); approved narrative direction | P0 fixes applied (mask_x triple intervention, Theorem 4 rename, abstract toned down) |
| 2026-05-12 09:00 | External Expert | Found metrics inconsistency — publication-fatal | Rebuilt via two-stage pipeline; migrated_all_v2.json |
| 2026-05-12 16:30 | External Expert | Approved Phase 2 v2; 5 pre-freeze conditions | All 5 conditions completed |
| 2026-05-12 19:00 | External Expert | **Paper-data audit**: found NSVAR_d50/PlanA SHA duplicates, old doc contamination, encoding issues, inaccurate README claims | PlanA removed, old doc deprecated, audit fixed, READMEs corrected |
| 2026-05-12 | External Expert | **Framework approved with conditions**: Section 1-5 can proceed; Section 6-8 blocked pending P0/P1 + Appendix F | COMPLETED (P0/P1 done, Appendix F done) |
| 2026-05-12 | External Expert | **Round 2 audit**: appendix_all_metrics.tex + appendix_score_manifest.tex fail (old counts, PlanA residual). Main canonical data passes. Requested 7-file patch. | 7-FILE PATCH SUBMITTED |
| 2026-05-12 | External Expert | **Round 3 audit**: 7-file patch passed except 3 doc issues (README-data 12→11, README-paper state update, Appendix F top-k index). Requested 3-file micro-patch. | MICRO-PATCH SUBMITTED |
| 2026-05-13 | External Expert | **Full pre-freeze approved**: all 3 micro-patch issues resolved. Data frozen. Section 1-5 immediate writing; Section 6-8 full drafting with frozen v2 data. | APPROVED — ACTIVE DRAFTING |
| 2026-05-14 | External Expert | **Full manuscript audit**: ~20 P0/P1 issues found — duplicate bibliography, missing figure floats, undefined refs, ~18 claim-strength edits across Sections 7-11 | ALL FIXED — cleanup patch submitted |
| 2026-05-15 | External Expert | **Full manuscript final candidate approved**: `istf_jrngc(8).tex` passed all audits. Sections 1-11 + Appendix A-D final candidate. | APPROVED — Next: submission packaging |
| 2026-05-15 | External Expert | **Submission hygiene checklist** (6 phases): compile, citation audit, figure placement, notation, TNNLS formatting, packaging. | ALL 5/6 PASSED; packaging pending user confirmation |
| 2026-05-15 | External Expert | **Submission package audit**: reviewed `submission.zip`; identified NOTEARS "systematically underperform" must be fixed; recommended inserting Fig. 1 architecture diagram; confirmed LaTeX numbering is correct (Fig. 1 = diagnostics, Fig. 2 = benchmark before insertion). | BOTH FIXED; submission package finalized (562 KB, 3 figures, 15 pages) |
| 2026-06-23 | External Expert | **KBS Figure Consistency Pass v2 failed**: Table 1 still single-column; Fig. 1 needed v4 independent input schematic; Fig. 2 legend/log-scale/fidelity-axis issues; Fig. 5 label/significance-note cleanup; v3 ZIP must be self-contained and clean-directory compilable. | COMPLETED — `kbs_figure_consistency_v3.zip` generated; clean ZIP extraction 3-pass compile passed with 0 errors, 0 undefined refs, 0 missing figures, and no Table 1 `68.92006pt` overfull. |
| 2026-06-23 | User-directed figure upgrade | User authorized immediate top-journal figure rework beyond the v3 consistency checklist. | COMPLETED for Fig. 3, Fig. 4, Fig. 6, and Fig. A.1; captions/body updated; appendix figure/table numbering normalized to A.1/B.1 style; KBS manuscript 3-pass compile passed. |

---

## 10. Current Execution Status

### P0: DONE
- [x] NSVAR_d50_PlanA removed (duplicate SHA — data-path bug)
- [x] paper_experiment_results.md deprecated
- [x] migrated_all_v2.json top-level audit updated

### P1: COMPLETE (2026-05-12 final patch)
- [x] manifest.json UTF-8 encoding fixed
- [x] score_registry.json metadata backfilled from manifest
- [x] README-data.md corrected (datasets, methods, size, known limitations)
- [x] README-paper.md corrected (CT_pm25, CT_traffic, stationary neutral, NSVAR_d10)
- [x] Deterministic check claim weakened in appendix_score_manifest.tex
- [x] appendix_all_metrics.tex regenerated from 176-entry canonical v2 (PlanA removed, CT_pm25 values corrected, UTF-8, $\\pm$ LaTeX)
- [x] appendix_score_manifest.tex counts corrected (104/11/208/2.0 GB)
- [x] migrated_all_v2.json audit metadata supplemented (generation_scripts, schema_version, cleanup_script, generated_at)
- [x] **Appendix F: lag/top-k/self-edge evaluation protocol** — DRAFT COMPLETE (v3 micro-patch: top-k indexing corrected, ascending sort + [-k] clarified)

### P0/P1 Manuscript Cleanup: COMPLETE (2026-05-14)
- [x] Duplicate `\begin{thebibliography}` removed
- [x] Figure 2 and Figure 3 float environments inserted
- [x] Undefined refs and `amssymb` fixed
- [x] ~20 claim-strength/wording edits applied across Sections 7-11
- [x] Cross-reference audit passed

### P1 Textual Micro-Cleanup: COMPLETE (2026-05-15)
- [x] Section 6: "optimal (k)" → "top-k binarization"
- [x] Section 7.1: "repair is inactive" → "no statistically detectable change"
- [x] Sections 6.4/8.2: "occupies the regime" → "clearest positive evidence in CT\_medical"
- [x] Section 8.1: "establishing" → "characterizing"
- [x] Section 8.3: "extends JRNGC's applicability" → "provides a structurally safer route"
- [x] Sections 8.3/9: "elimination/eliminates" → "removal/removes"
- [x] Section 8.4: "most harmful" → "expected to be most relevant"
- [x] Section 7.4: T/d heuristic with caveat
- [x] LaTeX compile: 0 errors, 0 warnings
- [x] Grep audit: all old phrases confirmed zero residual

### Next Steps
1. Continue the P0 Jacobian semantics audit before preparing any submission/source package.
2. Decide whether ISTF should remain a cross-channel Mamba repair, move to a coordinate-preserving/depthwise repair, or be reframed as a Jacobian measurement audit.
3. Do not launch full benchmark reruns until the raw-input chain-rule scoring and coordinate-preservation question is resolved.
4. Preserve frozen manuscript figures/results for now; future manuscript edits should wait for the P0 method decision.

P0 audit note: see `paper-data/p0_jacobian_semantics_audit_2026-07-06.md`.

P0 method decision memo: see `paper-data/p0_method_decision_memo_2026-07-06.md`.

P0 minimal GPU benchmark plan: see `paper-data/p0_depthwise_gpu_benchmark_plan_2026-07-06.md`.

P0 external advisor review request: see `paper-data/p0_advisor_review_request_2026-07-06.md`.

P0.3 nonlinear-ground-truth closure (2026-07-07):
- Advisor blocker: old nonlinear transition used `std(pred)`, creating cross-output support expansion; old NS+Nonlinear metrics are invalidated.
- Local fix: coordinate-wise fixed-scale nonlinear transition plus transition-Jacobian support audit.
- Local output: `results_kbs/p0_repaired_local/20260707_034141/`.
- Status: `STAGE0_TRAJECTORY_COMPLETE_WITH_REFERENCE_LIMITATIONS`; blocking gates passed, with EMA and RawChainMamba retained only as reference/diagnostic for nominal-lag interpretation.
- Stage 1 preregistration draft: `paper-data/p0_3_stage1_preregistration_plan_2026-07-07.md`.

P0.3d execution-integrity patch (2026-07-07):
- Advisor blocker: P0.3c runner still used method-offset predictor initialization, aggregation could compute on incomplete subsets, frozen config was not byte-locked, and semantic gates were not part of formal aggregation.
- Local fix: paired predictor seed for all formal methods; separate RawChainMamba filter seed; strict approved config SHA whitelist; completeness-first aggregation; semantic-aware go/no-go; deterministic GPU settings; GPU infrastructure smoke config/commands.
- Local output: `results_kbs/p0_3d_execution_integrity/20260707_155436/`.
- Verification: `py_compile` passed; `tests/test_stage1a_execution.py` passed all 10 tests; formal plan-only has 100 runs; GPU-smoke plan-only has 5 runs.
- Writing status: no KBS manuscript edits. P0.3d remains execution infrastructure only and should not be used as performance evidence.
- Next gate: external advisor reviews `phase7_p0_3d_execution_integrity_v1.zip`; if approved, run limited GPU infrastructure smoke before full Stage 1a.

P0.3e release-lock patch (2026-07-07):
- Advisor blocker: release package/report commit mismatch, runner lacked approved code commit/source-tree guard, aggregator could accept modified config thresholds, and GPU smoke lacked a single hard validator.
- Local fix: added release source manifest, ignored approved code commit artifact, runner code/worktree guard, aggregator frozen-config/root-snapshot guard, and unified two-root GPU smoke validator.
- Verification: `py_compile` passed; `tests/test_stage1a_execution.py` passed all 12 tests; source manifest parity passed.
- Writing status: no KBS manuscript edits. P0.3e remains release infrastructure only and should not be used as performance evidence.
- Next gate: external advisor reviews `phase7_p0_3e_release_lock_v1.zip`; if approved, execute two 5-method GPU smoke roots and `validate_gpu_infrastructure_smoke.py` before full Stage 1a.

Stage 1a 901 official GPU run (2026-07-07):
- User clarified that formal experiments must run on the unified 901 environment; local partial Stage 1a output is non-official and must not be used.
- 901 release checkout: `/root/autodl-tmp/GUOJI/stage1a_release_65e6ae9`, commit `65e6ae9afef552c84d8211a9d6e9aa70db48c276`.
- GPU infrastructure smoke passed with validator `passed=true`, including both smoke roots and CP duplicate.
- Frozen Stage 1a completed 100/100 runs on 901 with `failed_count=0`; all formal methods passed completeness and semantic gates.
- Aggregation output: `/root/autodl-tmp/GUOJI/stage1a_release_65e6ae9/results_kbs/stage1a_65e6ae9_901/stage1a_aggregate_go_no_go.json`.
- Decision: `final_go=false` and `performance_go=false`; CP-depthwise did not meet the preregistered effect-size gates and Stage 1b should not start without advisor review.
- Writing status: no KBS manuscript edits. Do not present Stage 1a as positive evidence for the repaired method.

P1 bounded failure analysis (2026-07-08):
- Scope respected: no GPU, no Stage 1b, no seeds 4-8, no new method training, and no KBS manuscript edits.
- Output root: `/root/autodl-tmp/GUOJI/stage1a_release_65e6ae9/results_kbs/stage1a_bounded_failure_analysis/20260708_030924/`.
- Advisor package: `E:\GUOJI\kbs_review_packages\phase7_stage1a_bounded_failure_analysis_v1.zip`, SHA256 `cf98e6a017d1e4d1419c171cafd9302bd393dd180cbf93355e35b264c187bd3e`.
- Integrity status: official Stage 1a artifact inventory and post-analysis SHA256 manifest both passed; frozen aggregation recomputation matched the official aggregator.
- Strict decision: `INCONCLUSIVE_BOUNDED_POSTMORTEM`. A1 learned-CP-vs-identity equivalence passed, but A2 failed and A3 was not interpretable because no aligned data_seed=0 development artifact existed; B also failed because the preregistered nontrivial-filter-movement thresholds were not met.
- Writing status: continue freezing KBS text. Current CP-depthwise remains disqualified from Stage 1b and must not be restored as the KBS main method based on this P1 analysis.

KBS Route-B Jacobian coverage direction v1.1 (2026-07-10):
- Status: documentation-only correction package for advisor review; no experiments, no GPU, no seeds 4-8 inspection, no new method training, and no replacement of canonical `E:\GUOJI\elsarticle\istf_kbs.tex`.
- Corrected claim-evidence matrix: `paper-data/docs/kbs_jacobian_coverage_claim_evidence_matrix_v1_1_2026-07-10.md`.
- Corrected manuscript skeleton: `paper-data/docs/kbs_jacobian_coverage_manuscript_skeleton_v1_1_2026-07-10.md`.
- Changelog: `paper-data/docs/kbs_jacobian_coverage_v1_to_v1_1_changelog_2026-07-10.md`.
- Full-draft execution plan: `paper-data/docs/FULL_DRAFT_EXECUTION_PLAN.md`.
- Companion LaTeX skeleton: `E:\GUOJI\elsarticle\istf_kbs_coverage_audit_skeleton_v1_1.tex`.
- Key correction: the audit framework now separates score-route completeness, penalty-route completeness, score-penalty alignment, and coordinate/horizon validity. Score coverage and penalty coverage are not interchangeable.
- Active title direction: `Jacobian Coverage Audits for Reliable Neural Granger Causality`.
- Section 2 title locked for the skeleton as `Background and Related Work`.
- Stage 1a remains a main-text boundary table: semantic gates passed, CP performance gate failed, FixedFIR3 novelty gate failed, EMA full-H dominance no-go did not trigger, and Stage 1b eligibility failed.
- Legacy ISTF-Mamba is restricted to one score-semantics failure diagnostic claim: filtered-coordinate scores can diverge from original-input raw-chain attribution when cross-channel transformations destroy source-variable identity.
- Old CausalTime benchmark tables and operating-regime claims are removed from the active mainline.

### Route-B Full Draft v1: COMPLETE PENDING GPT REVIEW (2026-07-10)
- [x] Separate LaTeX draft created: E:\GUOJI\elsarticle\istf_kbs_jacobian_coverage_draft_v1.tex.
- [x] Working title: Jacobian Coverage Audits for Reliable Neural Granger Causality.
- [x] Eight-section route: Introduction; Background and Related Work; Structural Shortcut in Auxiliary Predictive Pathways; Jacobian Coverage Audit Framework; Diagnostic Evidence for Coverage Failures; Semantic Repair as a Boundary Case; Discussion; Limitations and Conclusion.
- [x] Draft uses only frozen controlled concat diagnostics, the limited P0 score-semantics diagnostic, official Stage 1a, and bounded P1 evidence.
- [x] Root-cause synthetic, old CausalTime, old operating-regime claims, and all ISTF/Mamba performance claims are excluded from the active mainline.
- [x] CP-depthwise is reported as a semantic-pass/performance-fail boundary case, not as a successful method.
- [x] Draft evidence documents:
  - paper-data/docs/kbs_jacobian_coverage_claim_evidence_matrix_v1_2_draft_2026-07-10.md
  - paper-data/docs/DRAFT_V1_CHANGELOG.md
  - paper-data/docs/DRAFT_V1_OPEN_ISSUES.md
  - paper-data/docs/DRAFT_V1_EVIDENCE_TRACEABILITY.md
- [x] Current compile status: 3-pass PDF build completed; visual review found no overlap or clipped table/figure content. Underfull warnings remain in dense tables and narrow columns.

### Next Step
1. Send the dedicated draft v1 review ZIP to GPT for a scientific and claim-boundary review.
2. Do not replace istf_kbs.tex, run training, start Stage 1b, or inspect seeds 4-8 before GPT approves the Route-B draft.
- Root-cause synthetic diagnostics are downgraded to `Pending semantics/provenance audit; appendix candidate`.
# CURRENT ROUTE-B KBS STATUS (2026-07-10)

The active KBS writing artifact is the separate, frozen-evidence manuscript
`E:\GUOJI\elsarticle\istf_kbs_jacobian_coverage_draft_v2.tex`, titled
*Jacobian Coverage Audits for Reliable Neural Granger Causality*. It is a
reliability/audit manuscript, not an ISTF performance-method manuscript, and
does not replace canonical `E:\GUOJI\elsarticle\istf_kbs.tex`.

Route-B boundaries: no new training, GPU, Stage 1b, CP revival, inspection of
seeds 4--8 model-training/model-performance outputs, canonical-manuscript
replacement, historical CausalTime performance narrative, operating-regime
claims, root-cause synthetic main-text claims, or legacy ISTF-Mamba performance
claims. Legacy Mamba is a filtered-coordinate versus raw-chain score-semantics
diagnostic only. Stage 1a is a semantic-pass/performance-fail boundary table;
P1 is `INCONCLUSIVE_BOUNDED_POSTMORTEM`, and A3 may only be described as
uninterpretable because `gradient_replay_alignment_valid=false`.

Active V2 evidence docs: `paper-data/docs/kbs_jacobian_coverage_claim_evidence_matrix_v2_2026-07-10.md`,
`DRAFT_V2_EVIDENCE_TRACEABILITY.md`, `DIAGNOSTIC_REPLICATION_AUDIT.md`,
`BIBLIOGRAPHY_AUDIT.md`, and `DRAFT_V2_OPEN_ISSUES.md`.

# CURRENT ROUTE-B KBS STATUS: V2.1 (2026-07-10)

The active review artifact is now the separate frozen-evidence manuscript
`E:\GUOJI\elsarticle\istf_kbs_jacobian_coverage_draft_v2_1.tex`. It remains a
documentation-only Route-B draft and does not replace canonical
`E:\GUOJI\elsarticle\istf_kbs.tex`.

V2.1 corrects the shortcut proposition so that the controlled shortcut requires
the auxiliary route to be omitted from both the x-only graph score and its
corresponding Jacobian penalty. It keeps score-only incompleteness,
penalty-only incompleteness, coordinate ambiguity, and horizon truncation as
separate audit outcomes. Table 1 is explicitly controlled-diagnostic and
claim-specific. The main manuscript contains no historical benchmark,
operating-regime, or legacy-Mamba performance claims.

The documentation set is `paper-data/docs/*V2_1*`, including the claim-evidence
matrix, evidence traceability register, replication audit, official-source
bibliography audit, bibliography correction changelog, final audit report,
open issues, and evidence-extract index. The local final three-pass build is
14 pages with 0 errors, 0 undefined references/citations, 0 missing files, and
0 Overfull hboxes. The remaining decision is GPT review of the separate V2.1
package; no experiment, GPU, Stage 1b, CP revival, or seeds 4--8 inspection is
permitted before that review closes.

# CURRENT ROUTE-B KBS STATUS: V2.2 (2026-07-10)

The active review artifact is the separate frozen-evidence manuscript
`E:\GUOJI\elsarticle\istf_kbs_jacobian_coverage_draft_v2_2.tex`, titled
*Auditing Jacobian Coverage in Neural Granger Causality*. It does not replace
canonical `E:\GUOJI\elsarticle\istf_kbs.tex`.

V2.2 is a focused scientific-writing pass. Proposition 3.1 is restricted to
the structural possibility that a predictor can use an auxiliary route omitted
from both the x-only graph score and the corresponding x-only Jacobian penalty,
while frozen diagnostics separately establish empirical graph-ranking and
coefficient-fidelity degradation. Section 4 is the constructive center: it
starts from the declared graph object and keeps score-route completeness,
penalty-route completeness, score--penalty alignment, coordinate validity, and
horizon validity distinct. Table 1 retains its V2.1 five-column geometry.

The five-seed `|J_c|/|J_x| = 3.689 +/- 0.750` value is retained only as a
post-hoc sensitivity-mass route-usage diagnostic. It is not a graph score,
conditional-Granger estimate, benchmark result, performance result, or
confirmatory result. Stage 1a and P1 claim boundaries are unchanged. The local
three-pass build is 14 pages with 0 errors, 0 undefined references/citations,
0 missing figures, and 0 Overfull hboxes/vboxes.

Review documents are `paper-data/docs/*V2_2*`. No new experiment, GPU, Stage
1b, CP revival, inspection of seeds 4--8 model-training/model-performance
arrays, or canonical-manuscript replacement is permitted by this drafting
pass.

# CURRENT ROUTE-B KBS STATUS: V2.3 SCIENCE LOCK (2026-07-10)

The active review artifact is the separate manuscript
`E:\GUOJI\elsarticle\istf_kbs_jacobian_coverage_draft_v2_3.tex`, titled
*Auditing Jacobian Coverage in Neural Granger Causality*. Canonical
`E:\GUOJI\elsarticle\istf_kbs.tex` remains unchanged.

V2.3 adds a bounded objective-level non-certification remark: under the
controlled shortcut assumptions, an auxiliary-only predictor has `R_x=0`, so a
low x-only regularized objective does not itself certify that the x-only score
represents information used for prediction. This is an existence statement,
not an optimizer-selection or empirical-degradation claim.

The framework is science-locked with five separate dimensions: score-route
completeness, penalty-route completeness, score-penalty alignment, coordinate
validity, and horizon validity. Its output is a claim-specific set of all
applicable flags; `CLAIM-COVERED` requires every applicable dimension to pass
with no required unknown. Score, penalty, and exemption status are recorded
independently for each architecture-declared predictive route class.

The exact P1 boundary is: P1 did not inspect or use Stage 1b model-training or
model-performance outputs for seeds 4--8. Stage 1a tables and gate outcomes are
unchanged from V2.2. Root-cause, CausalTime performance, Stage 1b, legacy-Mamba
performance, and P1 A3 raw values remain excluded.

The 14-page three-pass build has 0 errors, 0 undefined references/citations, 0
missing figures, and 0 overfull boxes. All pages passed visual inspection.
Further structural drafting is stopped; the next action is a full adversarial
paper review and journal-strategy decision.

# CURRENT ROUTE-B KBS STATUS: V2.4 SUBMISSION CANDIDATE (2026-07-10)

The separate manuscript is now
`E:\GUOJI\elsarticle\istf_kbs_jacobian_coverage_submission_candidate_v2_4.tex`.
It retains the accepted Route-B thesis and does not replace canonical
`istf_kbs.tex`.

Critical corrections completed:

- Table 1 defines the controlled concat implementation from source:
  `c_u=g_phi(X_{0:u})`, causal full-prefix support, exact predictor tensor,
  detach behavior, partial x-only score/penalty, full-penalty coordinates and
  the still-partial inherited graph score.
- The abstract marks the capacity and coefficient diagnostics as two single
  runs before giving their numerical values.
- The frozen post-hoc ratio is called a mean absolute Jacobian-magnitude ratio,
  with explicit coordinate-scale and reparameterization limitations.
- Every frozen full auxiliary-penalty variant is reported in an appendix table;
  `full_lc_10` is one exploratory setting, not an optimum.
- Related Work now distinguishes the graph-score audit from attribution axioms,
  saliency sanity checks, faithfulness evaluation and explanation
  regularization using officially verified primary sources.
- Main-text internal protocol language is reduced; the boundary study remains a
  go/no-go table, and the postmortem remains inconclusive.
- Submission metadata placeholders and journal-strategy records are present.

The local three-pass build is 17 pages with 0 errors, 0 undefined
references/citations, 0 missing files and 0 overfull boxes. Visual inspection
found no clipping, overlap or float-order error after the Appendix B/C barrier
fix. The next permitted action is GPT submission-readiness review of the v2.4
ZIP. No new experiment, GPU, Stage 1b, canonical replacement, or seeds 4--8
model-performance inspection is permitted.

# CURRENT ROUTE-B KBS STATUS: PHASE 8 FINAL INDEPENDENT REVISION (2026-07-12)

The active review manuscript is the independent file
`E:\GUOJI\elsarticle_phase8_final\istf_kbs_jacobian_coverage_phase8_final.tex`.
Canonical `istf_kbs.tex` and the frozen v2.4 archive remain unchanged.

The manuscript title is *Jacobian Coverage Audits for Reliable Neural Granger
Causality*. Its locked evidence line is:

1. capacity-based prediction--knowledge decoupling replicated in 5/5 paired
   runs;
2. partial graph and lag-1 coefficient degradation replicated in 5/5 paired
   runs;
3. corrected fixed-target auxiliary dominance did not replicate (0/5), with raw
   history more prediction-critical;
4. post-hoc total raw-chain scoring did not restore graph or coefficient
   fidelity;
5. coverage-aligned full-prefix regularization produced a bounded
   graph--prediction trade-off across the tested strengths, but no lambda passed
   the complete pilot-go rule;
6. held-out confirmation was not executed and repair development is closed.

The final independent PDF has 16 pages. Three-pass compilation reports zero
errors, undefined references/citations, missing figures, and overfull boxes.
All pages were rendered and visually inspected. Four new figures replace the
single-run panels and show seed-level data plus population SD. Stage 1a remains
a supporting historical boundary table, and P1 A3 is described only as
uninterpretable because `gradient_replay_alignment_valid=false`.

The final review documents, claim matrix, terminology ledger, limitations,
journal strategy, traceability, and reproduction commands are under
`paper-data/docs/phase8_final/`. No further repair tuning is authorized.

# CURRENT KBS STATUS: SUBMISSION CANDIDATE V1 (2026-07-12)

The terminal Phase 8 artifact passed GPT scientific review. The active review
manuscript is now the independent file
`E:\GUOJI\elsarticle_phase8_final\istf_kbs_jacobian_coverage_submission_candidate_v1.tex`.
Canonical `istf_kbs.tex`, the Phase 8 terminal TeX, and the frozen v2.4 archive
remain unchanged.

The final title is *Auditing Jacobian Coverage in Neural Granger Causality*.
The submission revision makes the following corrections:

1. Figure 2d is referenced only as the corrected fixed-target intervention;
   full auxiliary-penalty variants are cited through the appendix table.
2. Penalty-route coverage mitigation is explicitly separated from the still
   partial score-route coverage.
3. Post-hoc total raw-chain scoring is bounded to the controlled coefficient
   study and is not generalized universally.
4. The final repair evidence is described as a bounded graph--prediction
   trade-off across tested strengths; no efficient-set analysis is claimed.
5. Phase 8 is the central empirical boundary evidence. The full Stage 1a table
   and detailed P1 discussion are in the appendix, while the main text retains
   only the historical gate summary.

The 16-page three-pass build has 0 errors, 0 undefined references/citations, 0
missing figures, and 0 overfull boxes. All pages passed visual inspection after
an appendix float-order correction. The log retains 65 underfull hbox and 8
underfull vbox warnings, all visually audited and reported. Author identity,
affiliation, funding, CRediT roles, conflicts, and public archive URLs remain
explicit author-controlled placeholders.

# CURRENT KBS STATUS: AUTHOR METADATA READY V1 (2026-07-12)

GPT accepted the submission candidate scientifically and authorized only a final
author-metadata readiness correction. The manuscript source remains
`istf_kbs_jacobian_coverage_submission_candidate_v1.tex`; its author-ready commit
is `1b50e8a577995b02485764fead959bfa10ae159d`, with package-guide branch head
`ffe74087d17c6925aaf099eedb268efb3091c09b`.

The rendered `Appendix Appendix C` duplication is fixed, all five declaration
paragraph headings render with one period, and the AI-use statement now includes
software implementation and test development. Instructional square brackets in
the reusable audit template were replaced by italic `Entry:` prompts. The only
remaining PDF placeholders are author-controlled identity, affiliation, funding,
conflict, CRediT, repository, and archive fields; one additional bracketed object
is mathematical concatenation rather than a placeholder.

The final author-ready build remains 16 pages with 0 errors, undefined
references/citations, missing figures, or overfull boxes. It reports 64 underfull
hbox and 8 underfull vbox warnings. Rendered-text grep and whole-document visual
inspection both pass. No scientific content, experiment, method, threshold, or
conclusion changed.

# CURRENT KBS STATUS: VISUAL AND LAYOUT REWORK V1 (2026-07-13)

The active visual-review branch is `phase8/kbs-visual-rework-v1` in both the
method and independent manuscript repositories. It preserves the author-ready
scientific content and all frozen numerical evidence.

The four main figures were redrawn from the same frozen artifacts after a
page-by-page comparison with four local published KBS papers. The adopted
patterns are a dominant horizontal predictive flow for the framework figure,
short panel titles, restrained method-consistent colors, thin paired-seed
traces, compact local/shared legends, and a heatmap-like gate summary. Figure 1
now separates predictive routes, the coverage declaration, and the
claim-specific workflow; Figure 3 replaces the previous derivative cards with
an explicit coordinate and horizon schematic. Figure captions were shortened
without changing their evidentiary boundaries.

The appendix float layout was also audited. Removing the unnecessary barrier
after the Stage 1a wide table reduced the manuscript from 16 to 15 pages and
eliminated one large sparse appendix page while preserving table order. A
bottom-wide-float experiment increased the page count and whitespace and was
rejected. The retained build has 0 errors, undefined references/citations,
missing figures, overfull hboxes, overfull vboxes, or float-too-large warnings.
All 15 pages were rendered and visually inspected at publication scale.

No experiment, metric, seed, claim strength, author placeholder, or canonical
manuscript was changed. The next review step is visual approval of the 15-page
PDF and four standalone figures before any author metadata is filled.

# CURRENT KBS STATUS: VISUAL AND LAYOUT REWORK V2 (2026-07-13)

The active review branches are `phase8/kbs-visual-rework-v2` in the method and
independent manuscript repositories. The review source is
`istf_kbs_jacobian_coverage_submission_candidate_v2.tex`; v1 and the canonical
manuscript remain unchanged.

The revision addresses three presentation findings without changing scientific
content. First, a source-verified controlled-concat architecture figure now
anchors page 4 and separates causal prefix processing, the raw-target prediction
path, and three derivative objects. Second, a reusable audit-workflow figure
anchors page 8, with the replicated quantitative evidence following on page 9.
Pages 4--9 therefore alternate figures, tables, and prose instead of presenting
five consecutive text-only pages. Third, the Appendix B/C wide tables use
position-controlled two-column strips, while the longer Appendix D template is
flushed as a complete top-of-page float. The previous page-12/page-13 blank
regions and delayed isolated-table page are absent.

All six figures use a restrained grayscale, blue-gray, and muted-brick palette.
Large saturated status blocks were replaced by white cells, outlines, marker
shapes, and direct labels. The final direct three-pass build is 16 pages with
0 LaTeX errors, 0 undefined references/citations, 0 missing figures, and 0
overfull hboxes/vboxes. Every page and all six standalone PNGs were visually
inspected at publication scale. No experiment, metric, claim boundary, seed,
author placeholder, or frozen artifact changed.

# CURRENT KBS STATUS: VISUAL NARRATIVE REWORK V3 (2026-07-14)

The active review branches are `phase8/kbs-visual-narrative-rework-v3` in the
method and independent manuscript repositories. The review source is
`istf_kbs_jacobian_coverage_submission_candidate_v3.tex`; v2, the canonical
`istf_kbs.tex`, and every frozen artifact remain unchanged.

Figures 1--3 were rebuilt after a documented comparison with four local KBS
papers, JRNGC, and CUTS. They now use scientific glyphs, time-series windows,
Jacobian tensors, graph objects, and audit signatures rather than text-only
cards. All panels are white with thin gray boundaries. Figures 4--6 use the same
frozen values with a restrained blue/brick/gray palette and shape-based method
encoding. High-saturation pixels occupy at most 0.004 percent of any standalone
PNG.

The page sequence is now Figure 2 on page 4, Figure 3 on page 5, a wide audit
profile table on page 6, and Figures 4--6 on pages 7--9. Appendix D contains a
reusable audit record and exact score-aggregation definition. Pages 12--13 are
dense, balanced, and free of blank/float-only pages. The retained three-pass
build is 13 pages with 0 errors, undefined references/citations, missing files,
overfull boxes, float-too-large warnings, or unprocessed floats.

No GPU, experiment, new seed, metric recalculation, claim-threshold change, or
legacy performance revival occurred. The next step is clean-directory package
verification and external visual review before author metadata is completed.

# CURRENT KBS STATUS: TOP-JOURNAL NARRATIVE REWORK V4 (2026-07-14)

The active review branches are `phase8/kbs-top-journal-narrative-v4` in the
method and independent manuscript repositories. The review source is
`istf_kbs_jacobian_coverage_submission_candidate_v4.tex`; v3, canonical
`istf_kbs.tex`, and every frozen artifact remain unchanged.

V4 replaces the three sequential conceptual diagrams with one integrated
route-resolved figure, advances the first quantitative evidence to Figure 2,
and retains four main figures total. The figure design was audited against
Tao et al. KBS 2025 Fig. 1, Chen et al. KBS 2024 Fig. 1, JRNGC ICML 2024
Fig. 1, and CUTS ICLR 2023 Fig. 1. The exact design map and no-copy boundary
are documented under `paper-data/docs/`.

The manuscript argument is now mechanism -> coverage declaration -> controlled
evidence -> training-time frontier. Internal development language, P1 gradient
replay details, repeated no-go disclaimers, and speculative optimization
accounts were removed. Material evidence boundaries remain explicit, including
the legacy total-objective meaning of `pred_loss`, the post-hoc score result,
the graph--prediction frontier, and the distinction between score semantics and
empirical advantage.

The retained 12-page three-pass build has 0 LaTeX errors, 0 undefined
references/citations, 0 missing figures, and 0 overfull hboxes. Figure 1 anchors
page 2; pages 4--8 alternate full-width tables, quantitative figures, and prose.
Standard `table*` floats replace unstable `strip` blocks, eliminating the page-9
duplicate-text defect. Page 13 is eliminated; page 12 contains the balanced tail
of the 46-item bibliography.

No author metadata, GPU work, experiment, new seed, metric recalculation, or
canonical manuscript edit occurred. The next step is clean-package verification
and external review before deciding whether v4 replaces v3 as the author-ready
baseline.

# CURRENT KBS STATUS: SYNCHRONIZED CHINESE REVIEW MIRROR (2026-07-14)

A complete Chinese proofreading mirror of the v4 manuscript is now maintained
in the independent manuscript repository:

- source: `istf_kbs_jacobian_coverage_submission_candidate_v4_zh.tex`;
- PDF: `istf_kbs_jacobian_coverage_submission_candidate_v4_zh.pdf`;
- QA and maintenance tools: `qa/chinese_review_v4/`.

The Chinese mirror uses a simple 16-page A4 one-column layout. It translates
all manuscript and appendix prose, captions, tables, and author-facing
placeholders, while preserving the English-master formulas, labels, citation
keys, numerical values, figures, audit labels, and evidence boundaries. Figure
interiors and bibliography entries remain English to avoid divergence from the
submission assets.

The build and synchronization checks pass with zero LaTeX errors, undefined
references/citations, warnings, or overfull hboxes. The English v4 source is the
sole submission master. Future scientific edits must be applied to the English
version first and synchronized to the Chinese mirror in the same writing
session; the Chinese file must not become an independent source of claims.

# CURRENT KBS STATUS: PHASE 9 ACCEPTANCE-READINESS PLANNING (2026-07-22)

The user approved a prospective strengthening program after a targeted review
of recent KBS causal-discovery papers and top-venue reliability-audit work. The
planning branch is `phase9/kbs-acceptance-readiness-v1`; v4 English/Chinese
manuscripts and all Phase 8 frozen artifacts remain unchanged.

The proposed plan defines two internal states. `KBS_SUBMISSION_READY` requires
theory, an executable audit, prospective cross-architecture controls, external
known-graph and real-case validation, and reproducibility gates.
`KBS_STRONG_READY` additionally requires the claim-specific diagnostics to show
prospective validity and the fixed full-auxiliary-penalty mitigation to
generalize without unacceptable pure-MSE cost. Neither state guarantees an
editorial outcome.

The exact documentation is under
`paper-data/docs/phase9_kbs_readiness/`. The plan uses fresh Phase 9 seed
namespaces, keeps Phase 7 seeds 4--8 closed, does not revive CP-depthwise or the
failed Phase 8 repair, and treats DREAM3 as a known-graph in-silico benchmark and
MoCap as a real no-ground-truth audit case. Implementation, run-matrix release,
GPU work, and manuscript revision require a separate advisor approval.

# CURRENT KBS STATUS: PHASE 9 RTX 4090 BOUNDED ROUND 1 (2026-07-25)

The first bounded development round is complete on branch
`phase9/4090-bounded-enhancement-v1`. The English v4 manuscript, synchronized
Chinese review mirror, canonical `istf_kbs.tex`, and all frozen artifacts remain
unchanged.

Two method lines are closed:

1. A constrained adaptive FIR produced a development-only AUROC gain in the
   stationary nonlinear D2 cell at 2,000 iterations, but neither the static nor
   contextual version produced stable non-stationary gains. The contextual
   candidate was below baseline in both bounded 2,000-iteration
   non-stationary cells. This does not support reviving filtering as the paper's
   positive method contribution.
2. Prediction-guarded full-prefix regularization lowered pure prediction MSE
   but materially reduced graph recovery. Protecting only the historical term
   remained close to the standard repair and did not establish a better
   frontier. This does not close the Phase 8 repair trade-off.

The audit-generalization line remains scientifically promising but is not yet
paper evidence. Across development-only NetSim and MoCap slices, both Mamba and
an active causal TCN auxiliary route showed nontrivial missing-route mass,
partial-versus-total score disagreement, positive fixed-target `mask_c`
effects, and high cross-source coordinate entropy. The same-window
H=32/64/128 audit left nominal-lag scores unchanged and found that H=64 captured
nearly all measured H=128 attribution mass, while attribution beyond H=128
remains unassessed.

Manuscript constraints:

- Do not add the adaptive/contextual FIR or gradient-projection results as
  positive method evidence.
- Do not describe NetSim as a successful graph-recovery benchmark; its
  development baseline was only random-to-moderate.
- MoCap supports only route-use and score-semantics diagnostics because no
  accepted direct graph ground truth is available.
- Do not claim architecture universality from this round. The defensible
  prospective target is a bounded cross-architecture result after an
  advisor-approved protocol and fresh formal runs.
- Preserve the current `HORIZON-TRUNCATED` label because no derivative mass
  beyond H=128 was computed.

The authoritative development summary is
`docs/phase9_4090_round1/PHASE9_4090_ROUND1_REPORT.md`. Before any manuscript
revision, freeze a small prospective audit-only protocol with held-out
subjects/segments and method initializations; retain the current rows as
development data only.

# CURRENT KBS STATUS: AUDIT-GENERALITY PREREGISTRATION V1 (2026-07-25)

A planning-only two-stage validation is frozen under
`paper-data/docs/phase9_audit_validation_v1/`. The manuscript and Chinese
review mirror remain unchanged.

Stage A would use four hash-selected held-out NetSim subjects, two
nonoverlapping MoCap segments, baseline JRNGC, Mamba concat, causal TCN concat,
and three fresh training replicates. It is an internal RTX 4090 go/no-go stage,
not manuscript confirmation evidence. Stage B uses four separately sealed
NetSim subjects, the same MoCap segments, two fresh replicates, and AutoDL only
after Stage A and advisor approval.

The preregistered claim target is narrow: reproducible omitted-route
attribution, fixed-target auxiliary-route use, partial-versus-total nominal
score disagreement, source-coordinate mixing, and bounded temporal support
across two auxiliary architectures and two data domains. It does not reopen
the stopped filtering/repair methods and cannot support superiority, universal
architecture validity, full-prefix completeness, or graph truth on MoCap.

No manuscript change is allowed until both stages pass. If only one
architecture passes, the result is explicitly `PARTIAL_GENERALITY`; if H=64
fails the H=128 adequacy gate, execution stops rather than silently changing
the horizon.

# CURRENT KBS STATUS: STAGE A PASSED, STAGE B RELEASE PENDING (2026-07-26)

Stage A returned `UNLOCK_STAGE_B` after 56/56 clean CUDA records and an
independent byte-identical reaggregation. Both Mamba and causal TCN passed the
frozen generality and horizon gates across all six data units. This is a clear
positive go/no-go result for the audit-generality direction, but it remains
non-confirmatory by design.

The manuscript and Chinese mirror remain frozen. Stage A must not be presented
as final manuscript evidence. The NetSim baseline known-graph operating point
was weak (median AUROC 0.514), so the current positive result concerns route
coverage, fixed-target route use, coordinate mixing, and partial-versus-total
score disagreement rather than improved graph recovery.

Stage B retains the preselected subjects 16, 0, 30, and 10, the two frozen
MoCap segments, the same three methods, and two fresh replicates. No method,
threshold, horizon, intervention, or training field changed after Stage A was
observed. If Stage B returns `CONFIRMED_AUDIT_GENERALITY`, the strongest
eligible claim remains the bounded statement in the preregistration; it does
not establish a successful repair method, full-prefix attribution, or causal
ground truth for MoCap.

# CURRENT KBS STATUS: STAGE B CONFIRMED (2026-07-26)

Stage B completed 36/36 frozen AutoDL records and returned
`CONFIRMED_AUDIT_GENERALITY`. The result passed the release lock, smoke gate,
semantic checks, deterministic metadata checks, artifact-integrity audit, and
independent local reaggregation. It is the first confirmatory evidence eligible
for the Route-B manuscript mainline.

Allowed central empirical claim:

> Under a preregistered bounded raw-chain audit, omitted-route attribution,
> fixed-target auxiliary-route use, and partial-versus-total score disagreement
> were reproduced across held-out NetSim subjects, nonoverlapping MoCap
> segments, and two causal auxiliary preprocessors.

Required boundaries:

- Present this as audit-framework generality, not graph-performance
  superiority or a successful repair.
- State that the held-out NetSim baseline operating point was weak (median
  AUROC 0.495); graph-performance success was not a gate and was not obtained.
- Use MoCap only for route-use and score-semantics evidence because it has no
  accepted direct graph ground truth.
- Do not claim attribution completeness beyond H=128.
- Keep the failed CP-depthwise and Phase 8 repair lines closed.
- Do not revive legacy ISTF-Mamba benchmark or operating-regime claims.
- Do not present the two architectures or two domains as universal coverage.

The canonical English manuscript and Chinese review mirror remain unchanged.
The next writing session should first update the claim-evidence matrix and
traceability table, then revise the audit-generality Results/Discussion
sections from the frozen Stage A/Stage B artifacts.

# CURRENT KBS STATUS: ROUTE-B SUBMISSION CANDIDATE V5 (2026-07-26)

A new independent manuscript worktree now contains the first complete Route-B
submission candidate:

- English source:
  `E:\GUOJI\elsarticle_phase9_stageb\istf_kbs_jacobian_coverage_submission_candidate_v5.tex`;
- English PDF:
  `E:\GUOJI\elsarticle_phase9_stageb\istf_kbs_jacobian_coverage_submission_candidate_v5.pdf`;
- Chinese review source/PDF:
  `istf_kbs_jacobian_coverage_submission_candidate_v5_zh.tex/.pdf`.

The v5 title is:

> Jacobian Coverage Audits for Reliable Neural Granger Causality

The paper is no longer an ISTF performance paper. Its argument is:

1. a declared graph derivative can omit an architecture-declared predictive
   route;
2. score routes, penalty routes, coordinate identity, and attribution horizon
   must be audited independently;
3. controlled Phase 8 diagnostics reproduce prediction--knowledge decoupling;
4. preregistered Stage B confirmation reproduces bounded audit signatures
   across Mamba/TCN preprocessors and NetSim/MoCap held-out units;
5. semantic repair remains a boundary because CP-depthwise failed performance
   and novelty gates, while full-prefix regularization traced a steep
   graph--prediction frontier.

The English manuscript has eight first-level sections, four main figures, one
appendix diagnostic figure, and four bounded theoretical propositions. It
retains the following hard boundaries:

- Stage B confirms audit signatures, not graph-recovery performance;
- MoCap has no accepted direct graph ground truth;
- H=64/H=128 stability does not assess mass before H=128;
- the full-prefix repair is not a successful method contribution;
- legacy ISTF-Mamba is a score-semantics diagnostic only;
- the framework does not establish causal identifiability.

All five figures are stored in independent self-contained directories with
source data, backend-specific editable source, PDF/SVG/PNG exports, validation
JSON, and complete SHA256 manifests. Figure 1 remains an editable Draw.io
mechanism schematic. Figures 2--4 and Appendix Fig. E.1 are generated directly
from frozen CSV/JSON inputs by Python/matplotlib; their superseded Draw.io
versions are archival only. The restrained white/charcoal, muted-blue,
muted-rust, and muted-teal visual system replaces the prior saturated
multi-color presentation.

Current English build status:

- 14 pages;
- 0 LaTeX errors;
- 0 undefined references/citations;
- 0 missing figures;
- 0 overfull hboxes;
- all five included figure PDFs are single-page vector exports.

The prior large blank areas on pages 12--13 are removed. Quantitative visual
anchors now appear on pages 6, 8, and 10, with the conceptual framework on page
3 and the score-semantics appendix figure on page 13. The Chinese mirror is a
19-page single-column proofreading aid; its formatting is intentionally not a
submission constraint.

The v4 English/Chinese sources and canonical `istf_kbs.tex` remain frozen.
V5 must pass final self-review, clean-package verification, commit/push, and
author metadata replacement before it can become the submission master.

# MANUSCRIPT REPOSITORY ORGANIZATION (2026-07-26)

The latest Route-B candidate has passed its existing self-review and is now the
canonical manuscript in the Phase 9 manuscript worktree:

- English source/PDF:
  `E:\GUOJI\elsarticle_phase9_stageb\istf_kbs.tex/.pdf`;
- Chinese review source/PDF:
  `E:\GUOJI\elsarticle_phase9_stageb\review\chinese\istf_kbs_zh.tex/.pdf`;
- integrated English/Chinese appendix sources:
  `E:\GUOJI\elsarticle_phase9_stageb\supplement\appendix_en.tex` and
  `appendix_zh.tex`;
- Appendix Fig. E.1 assets:
  `E:\GUOJI\elsarticle_phase9_stageb\figures\phase9_submission\figE1_score_semantics\`.

Superseded manuscript, figure, source-data, table, planning, and QA versions
are grouped under `E:\GUOJI\elsarticle_phase9_stageb\archive\` and are no
longer active build inputs. The canonical build entry point is
`scripts/build_manuscripts.ps1`. The organization pass changed paths and build
structure only; it did not change scientific values, claims, or frozen
evidence. The subsequent figure-number/backend correction replaced all four
quantitative Draw.io figures with Python/matplotlib outputs generated from the
same frozen data, renamed the integrated score-semantics asset to Fig. E.1,
and synchronized the Chinese appendix order. The corresponding current
manuscript commit is `9b90823a715729dd396f1cd40d70148c0c9cc590` on
`phase9/stageb-manuscript-integration-v1`.

# FIGURE 1 AUTHOR REVISION INTEGRATION (2026-09-04)

The author's subsequent Draw.io revision is now the canonical Figure 1. It
adds a dashed raw-history boundary, rounded direct/auxiliary blocks, a pale
amber predictor field, and revised spacing while retaining the shared-neuron
predictor and the same route-coverage argument. The exact author-saved source
is archived as `provenance/user_revision_20260904_142256.drawio`.

A validation-only repair converted the `route_aux` edge from mixed attached
and absolute endpoint metadata to equivalent absolute endpoints; it does not
alter the rendered geometry. Draw.io validation reports zero errors/warnings.
Standalone, grayscale, 900-pixel, English page-3, and Chinese page-3 renders
show no clipping or overlap, and the outer-layout whitespace audit is 17.43%.
The unchanged caption remains semantically aligned. English and Chinese clean
builds remain 18/23 pages with no errors, undefined references/citations,
missing figures, or overfull boxes. The integrated manuscript/figure revision
is committed and pushed as
`dd2a6f30e99df73cc00d829316720c7ef4f8a142` on
`phase9/stageb-manuscript-integration-v1`.

# NUMERIC AUDIT STATUS (2026-07-31)

The current English manuscript completed a full frozen-evidence numeric audit.
The 16-sheet master workbook is:

`E:\GUOJI\outputs\kbs_numeric_audit_20260731\KBS_NUMERIC_MASTER_AUDIT_20260731.xlsx`

All 141 hand-mapped manuscript/configuration claims agree with their
authoritative sources: 43 exactly and 98 at the manuscript's declared display
precision. There are zero substantive mismatches and zero workbook formula
errors. The audit also inventories the complete Stage B formal run ledger,
Stage 1a data-seed units, repair frontier, and authoritative scalar JSON
fields.

This audit does not expand the permitted narrative. Full-auxiliary
`pred_loss`, legacy ISTF-Mamba, Stage 1a, Stage B, P1 A3, and the Phase 8
repair frontier retain their previously frozen semantic boundaries. Future
manuscript number changes should be made from the authoritative source first
and then rechecked with
`E:\GUOJI\elsarticle_phase9_stageb\qa\current\numeric_audit\`.

# FIGURE 1 FINAL REDESIGN STATUS (2026-09-04)

The first three-panel redraw was rejected during author visual review and is
superseded. The active KBS Figure 1 is now a Draw.io-native, single-scene
mechanism diagram informed by a visual audit of 10 related top-venue papers.
It follows the continuous visual chain raw history -> direct/auxiliary
operators -> neural predictor -> Jacobian decomposition -> graph-readout
mismatch. Explanatory prose and the former audit ledger were removed from the
artwork. The redesign changes presentation only and introduces no empirical
number or new scientific result.

The canonical figure directory is:

`E:\GUOJI\elsarticle_phase9_stageb\figures\phase9_submission\fig1_route_coverage_framework\`

It contains the editable `.drawio` source, PDF/SVG/PNG exports, design and
source manifests, validation JSON, grayscale/thumbnail checks, and English and
Chinese manuscript-page renders. A second author-directed refinement enlarged
the raw-history and direct/auxiliary blocks, removed the ambiguous black route
junction, and replaced the generic neural-node icon with an explicit
`W_X/W_C -> Sigma -> sigma -> W_o` predictor chain. All connector segments are
now horizontal or vertical. The palette is restricted to muted teal, muted
rose, charcoal, and neutral gray. Tight geometric cropping leaves approximately
14.5% outer-layout whitespace, with no background recoloring used to disguise
empty space. The active English and Chinese manuscripts remain 18 and 23 pages,
respectively, and both passed clean three-pass builds and the
submission-closure audit after the replacement.

# FIGURE 1 AUTHOR-LAYOUT TYPOGRAPHY PASS (2026-09-04)

The author subsequently adjusted the active Draw.io layout manually. That
saved geometry is now the canonical Figure 1 source and supersedes the exact
layout coordinates described in the preceding redesign note. The follow-up
pass changed typography only: labels below the manuscript-scale readability
floor were raised to 18--22 Draw.io font units while titles retained their
existing hierarchy. No scientific content, caption, frozen evidence, module
position, or visible route geometry changed.

The figure directory now includes `scripts/export_existing_drawio.mjs`, which
validates and exports the manually maintained `.drawio` without reconstructing
its geometry. The prior programmatic builder is retained for provenance but is
not the routine export path. English page 3, Chinese page 3, a 900-pixel
thumbnail, grayscale render, and vector PDF were rechecked with no clipping or
overlap. Draw.io validation has zero errors/warnings, and the English/Chinese
three-pass builds remain 18/23 pages with zero errors, undefined references,
missing figures, or overfull boxes. Portal submission copies remain unchanged
until author visual approval.

# FIGURE 1 NEURAL-PREDICTOR GLYPH (2026-09-04)

Following the author's request to make the predictor read more clearly as a
neural model, the central `f_theta` region now retains `W_X` and `W_C` as
route-specific maps into two compact shared-neuron layers and an output map
`W_o`. The three displayed nodes per layer are schematic and are not a claim
about the implemented layer width. The outer author-adjusted composition,
Jacobian decomposition, route-readout contrast, and restrained palette remain
unchanged.

All directional flow arrows remain horizontal or vertical. Only the 18 thin,
arrowless neural synapses are diagonal. The exact prior source is archived as
`provenance/pre_neural_predictor_layout.drawio`, and the redesign scope and QA
are recorded in `provenance/neural_predictor_redesign.json`. English and
Chinese captions were synchronized, and clean three-pass builds remain 18 and
23 pages with no errors, undefined references/citations, missing figures, or
overfull boxes. No scientific result changed; portal submission copies remain
pending author approval. The manuscript/figure revision is committed and
pushed as `5ee1ee02fe5d52a259ec6fe75d8b6fd7f9658d86` on
`phase9/stageb-manuscript-integration-v1`.

# FIGURE 1 FINAL AUTHOR REFINEMENT AND SUBMISSION SYNC (2026-09-04)

The author's latest saved Draw.io revision is now the canonical Figure 1. The
exact source was archived as
`provenance/user_revision_20260904_144902.drawio` with SHA256
`8c85bad3819d558577ec3f2b8c31126d1dab33ed2143a3c0b1dfd9d861ea09c1`;
the canonical source has the same hash, so no geometry or style was altered by
Codex during this integration. Draw.io validation reports 186 cells, 152
vertices, 32 edges, zero errors/warnings, and no raster or reference-overlay
content. The final 3200 x 1174 export has 18.64% outer-layout whitespace under
the frozen 2%-fuzz content-bound audit.

English and Chinese three-pass builds remain 18 and 23 pages, respectively,
with zero errors, undefined references/citations, missing figures, overfull
boxes, or rerun warnings. The manuscript update and asset-index refresh were
committed and pushed as `8c7a4d3d760ac315b662011905a142c4b41e8f66` and
`f5c24eda2a4d118797466ed68a8af05e278ca4b7` on
`phase9/stageb-manuscript-integration-v1`.

The accepted files were synchronized to
`E:\GUOJI\投稿系统提交材料\`. The upload-facing LaTeX package is
`08_LaTeX源文件_LaTeX_Source\KBS_LaTeX_Source_kbs-submission-v3.zip`
(SHA256 `4dc75bbfc9df2a06123a0c499707058f7d5f82414f2c1b794827d23bb6f922a2`).
It contains 141 files, passes its 140-entry internal manifest, and compiles to
the same 18-page English manuscript in a clean extraction. The superseded v2
ZIP was moved to the internal, non-upload archive. Author metadata remains the
only intentionally pending manuscript field; no scientific value, claim, or
caption changed in this pass.
