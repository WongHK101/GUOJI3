# Remaining Risks

1. Importance weighting can still yield high-variance historical gradients.
2. B3 may remain dominated by zeros or subnormal float32 values.
3. The revised benchmark may pass aggregate reachability while the repair
   remains noncompetitive in the pilot.
4. Full exact attribution remains the main evaluation-time cost.
5. Track A diagnostics may fail replication independently of Track B.
6. Confirmation remains sealed regardless of the pilot outcome in this task.
