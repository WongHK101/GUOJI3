# Full Draft Execution Plan - KBS Jacobian Coverage Audit Paper

Date: 2026-07-10

Scope: this plan describes how to write the full Route-B manuscript from the v1.1 skeleton. It does not authorize new experiments, Stage 1b, seeds 4-8 inspection, new method training, or replacement of canonical `istf_kbs.tex` before explicit approval.

## Objective

Build a full KBS draft around the title direction:

> Jacobian Coverage Audits for Reliable Neural Granger Causality

The manuscript is a reliability and audit paper, not an ISTF performance paper.

## Execution Order

1. Freeze the claim-evidence map.
   - Use `kbs_jacobian_coverage_claim_evidence_matrix_v1_1_2026-07-10.md`.
   - Resolve only provenance gaps, especially root-cause synthetic artifacts.
   - Do not infer unsupported claims from old figures or old benchmark tables.

2. Build the new LaTeX draft as a separate file.
   - Start from `istf_kbs_coverage_audit_skeleton_v1_1.tex`.
   - Do not overwrite `istf_kbs.tex` until the new draft passes advisor review.
   - Keep at most eight first-level sections.

3. Draft the technical core first.
   - Section 2: Background and Related Work.
   - Section 3: Structural Shortcut in Auxiliary Predictive Pathways.
   - Section 4: Jacobian Coverage Audit Framework.
   - Section 5: Diagnostic Evidence for Coverage Failures.
   - Section 6: Semantic Repair as a Boundary Case.

4. Draft framing sections last.
   - Abstract.
   - Introduction.
   - Discussion.
   - Limitations and Conclusion.

5. Redraw minimal figures and tables from frozen artifacts only.
   - Fig. 1: coverage mismatch schematic.
   - Fig. 2: controlled concat diagnostics.
   - Fig. 3: score-semantics audit schematic or compact diagnostic panel.
   - Table 1: coverage declaration examples.
   - Table 2: Stage 1a boundary table.
   - Optional Table 3: evidence tiering and allowed claims.

6. Compile and audit.
   - Run 3-pass `pdflatex` in `E:\GUOJI\elsarticle`.
   - Check for LaTeX errors, undefined references, missing figures, and visible table overlap.
   - Do not treat this as a final submission compile until all figures are regenerated.

7. Advisor review package.
   - Include draft TeX/PDF, figures, claim-evidence matrix, changelog, compile log, and advisor-facing memo.
   - Ask GPT to verify claim strength, evidence provenance, and reviewer-risk mitigation.

## Writing Guardrails

- Use "Jacobian coverage audit framework", not "coverage certificate".
- Use "baseline JRNGC", not "raw JRNGC", unless raw-input scoring is explicitly contrasted.
- Use "expanding Jacobian coverage mitigated the failure in controlled concat diagnostics."
- Stage 1a must be a boundary table, not a benchmark leaderboard.
- Legacy ISTF-Mamba may support only the filtered-coordinate/raw-chain score-semantics failure claim.
- P1 A3 wording must be exactly bounded: "A3 was uninterpretable and did not pass because `gradient_replay_alignment_valid=false`."
- Root-cause synthetic diagnostics remain pending semantics/provenance audit and appendix candidate only.
- Old CausalTime benchmark tables and operating-regime claims stay out of the active mainline.

## Required Checks Before Full Draft Review

1. Every main-text empirical claim points to an exact artifact path.
2. Every artifact has a declared score type and model architecture.
3. Each figure caption states whether the evidence is controlled diagnostic, semantic audit, boundary table, or appendix-only historical material.
4. No text claims ISTF, Mamba, TCN, or CP-depthwise benchmark superiority.
5. No text uses seeds 4-8, Stage 1b, or new training outputs.
6. No text uses A3 raw-branch trends.

## Recommended Codex Mode

Use Default mode for the actual full drafting pass, because it will involve editing separate LaTeX files, regenerating manuscript figures from frozen data, compiling PDFs, updating logs, and committing/pushing changes. Use Plan mode only if the next step is another advisor discussion without file edits.
