# Draft v1 Changelog

Date: 2026-07-10

## New Separate Manuscript

- Added a separate full draft: E:\GUOJI\elsarticle\istf_kbs_jacobian_coverage_draft_v1.tex.
- Added a compiled draft PDF and compile log without replacing E:\GUOJI\elsarticle\istf_kbs.tex.
- Rebuilt the argument around Jacobian coverage auditing and causal graph reliability.

## Narrative Migration

- Replaced the prior ISTF-performance framing with the working title Jacobian Coverage Audits for Reliable Neural Granger Causality.
- Retained the auxiliary-route proposition after narrowing it to an existence and diagnostic statement.
- Added the score-semantics taxonomy: x-only, auxiliary-coordinate, filtered-coordinate, raw-chain, nominal-lag, and full-H.
- Added the five-part coverage declaration and four separate audit dimensions.
- Added diagnostic labels COVERED, PARTIALLY COVERED, COORDINATE-AMBIGUOUS, HORIZON-TRUNCATED, and UNASSESSED as labels rather than guarantees.

## Evidence Changes

- Reorganized the d_cond sweep, mask/shuffle, coefficient-recovery, and full auxiliary-penalty results around coverage failures.
- Added only a limited legacy ISTF-Mamba score-semantics panel. It is explicitly excluded from all performance, benchmark, operating-regime, and effectiveness claims.
- Added the official Stage 1a aggregate as a boundary table rather than a method leaderboard.
- Added the bounded P1 result with its required restriction: A3 was uninterpretable and did not pass because gradient_replay_alignment_valid=false.

## Removed From the Active Mainline

- Old CausalTime benchmark tables and operating-regime narrative.
- All ISTF performance-superiority and deployment-framework claims.
- Root-cause synthetic figures and claims pending a separate semantics/provenance audit.
- Any assertion that the CP-depthwise semantic repair was a successful performance method.

## New Draft Assets

- Fig. 1: conceptual coverage mismatch schematic.
- Fig. 2: score-semantics audit schematic plus frozen P0 diagnostic values.
- Fig. 3: frozen controlled concat diagnostics.
- Table 1: coverage declaration examples.
- Table 2: Stage 1a cell-level boundary summary.
- Table 3: Stage 1a gate interpretation.
