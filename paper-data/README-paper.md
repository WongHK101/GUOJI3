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

## 7. Writing Progress Tracker

### KBS Status Override (2026-07-05)
- Fig. 1 v9 is now promoted into the active KBS manuscript at `E:\GUOJI\elsarticle\istf_kbs.tex`.
- Introduction has been updated with causal-discovery assumption citations.
- Related Work reference expansion pass 2 is complete: 33 cited/used bibliography entries covering neural/time-series GC, shortcut learning, state-space/TCN filtering, temporal causal discovery, and KBS related work.
- Current KBS compile status: 26 pages, 0 LaTeX errors, 0 undefined references/citations, 0 missing figures, 0 Overfull hbox.
- Narrative unification pass is complete for Abstract, Results, Discussion, and Limitations. It preserved frozen figures and all numeric/statistical claims.
- Next writing focus: language-unification polishing across the full manuscript while preserving numbers, statistical language, figure assets, and claim boundaries.

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
1. Run language-unification polishing across the full KBS manuscript.
2. Preserve all frozen figure assets, statistical/numeric claims, and claim-strength boundaries.
3. Run reviewer-style pre-submission audit after language polishing.
4. Generate the next advisor-facing review package only after the internal pre-submission audit is clean.
