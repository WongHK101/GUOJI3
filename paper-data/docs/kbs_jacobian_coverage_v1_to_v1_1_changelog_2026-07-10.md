# KBS Jacobian Coverage Direction v1 to v1.1 Changelog

Date: 2026-07-10

This changelog records every documentation-only correction requested by GPT for v1.1.

## Corrections

1. Replaced the incorrect single audit question, "Do all predictive paths enter either the score or the penalty?", with four separate audit dimensions:
   - score-route completeness;
   - penalty-route completeness;
   - score-penalty alignment;
   - coordinate and horizon validity.
2. Kept the five-part declaration `C = (V_score, V_penalty, P_pred, M_coord, H_attr)`.
3. Redefined `P_pred` as architecture-declared predictive route classes, not an exact enumeration of every effective neural-network path.
4. Added the compact audit output taxonomy:
   - `COVERED`;
   - `PARTIALLY COVERED`;
   - `COORDINATE-AMBIGUOUS`;
   - `HORIZON-TRUNCATED`;
   - `UNASSESSED`.
5. Explicitly stated that these labels are diagnostic labels, not mathematical guarantees or certificates.
6. Corrected evidence tiering so root-cause synthetic experiments are no longer A-tier main-text core. They are now classified as: "Pending semantics/provenance audit; appendix candidate."
7. Added exact artifact paths, score type, model architecture, audit label, and permitted claim for:
   - d_cond sweep;
   - mask/shuffle;
   - coefficient recovery;
   - full auxiliary-Jacobian penalty;
   - root-cause synthetic;
   - legacy ISTF-Mamba filtered-coordinate/raw-chain comparison;
   - Stage 1a;
   - P1.
8. Rebounded legacy ISTF-Mamba to a score-semantics failure diagnostic only.
9. Added the exact permitted legacy ISTF-Mamba claim: "Filtered-coordinate scores can diverge from original-input raw-chain attribution when cross-channel transformations destroy source-variable identity."
10. Added explicit forbidden uses for legacy ISTF-Mamba:
    - graph-recovery performance;
    - benchmark superiority;
    - ISTF effectiveness;
    - Mamba effectiveness;
    - operating-regime claims.
11. Removed old CausalTime benchmark tables and operating-regime claims from the active mainline.
12. Added the limitation-only use of legacy CausalTime material: legacy benchmark results cannot validate the new raw-variable score-semantics claims.
13. Renamed Section 2 to "Background and Related Work".
14. Added required Section 2 coverage:
    - neural Granger causality;
    - Jacobian-based graph scoring;
    - temporal and auxiliary conditioning;
    - shortcut learning;
    - attribution and score reliability;
    - knowledge-extraction auditing.
15. Corrected abstract wording from "expanding Jacobian coverage can mitigate the failure" to "expanding Jacobian coverage mitigated the failure in controlled concat diagnostics."
16. Replaced "raw JRNGC" with "baseline JRNGC" where the raw-input scoring contrast is not explicit.
17. Kept Stage 1a as a main-text boundary table.
18. Required Stage 1a framing:
    - semantic gates passed;
    - performance gate failed;
    - FixedFIR3 novelty gate failed;
    - EMA full-H dominance no-go did not trigger;
    - Stage 1b eligibility failed.
19. Restricted P1 A3 wording to: "A3 was uninterpretable and did not pass because `gradient_replay_alignment_valid=false`."
20. Removed any mention of A3 raw-branch values, tendencies, potential pass status, or conflict-pattern counts.
21. Preserved the instruction that v1.1 is documentation-only and does not replace canonical `istf_kbs.tex`.
22. Preserved prohibitions:
    - no new experiments;
    - no Stage 1b;
    - no seeds 4-8 output inspection;
    - no new method training;
    - no canonical manuscript replacement.
