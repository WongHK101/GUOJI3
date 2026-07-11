# Phase 8 GPU Execution Changelog

This release supersedes the CPU-preflight-only release `9a54cd7` for the
authorized preflight, replication, and pilot execution task.

## Scientific-protocol corrections

- Capacity and coefficient replication gates now use the frozen legacy
  partial/raw-X score objects. Total raw-chain scores remain secondary
  semantic diagnostics.
- Repair-pilot gates use total nominal-lag raw-chain scores and total lag-1
  coefficient fidelity exclusively.
- The new no-auxiliary matched control now has an executable raw-target,
  raw-chain training adapter with a deterministic cyclic schedule. It remains
  non-historical and cannot enter graph-recovery gates.
- The 135-row run matrix is unchanged at SHA256
  `cc82b4283dfb28f5180891f1d0716d868bc10ddad53e64dbf174b9a36a04ac1a`.

## Execution infrastructure

- Added a release-locked, one-record CUDA runner.
- Added stage orchestration with durable ledgers and hard prerequisites.
- Added five-record GPU preflight validation, including the repeated first 20
  repair iterations, scale traces, runtime, full-attribution time, and VRAM.
- Added seed-preserving replication and pilot aggregators.
- Confirmation remains rejected by both record and stage runners.

## Replication classification

The approved planning package contained paired directional claims but no
additional replication-effect magnitude thresholds. Before any GPU output,
the implementation froze the following classification:

- `REPLICATED`: the joint preregistered direction holds in at least 4/5 pairs;
- `PARTIAL_REPLICATION`: it holds in exactly 3/5 pairs;
- `NON_REPLICATION`: it holds in fewer than 3/5 pairs.

All effect sizes and both partial/total semantic tracks are retained. No
post-result score or threshold selection is permitted.
