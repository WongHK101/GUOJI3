# Remaining Phase 8 Execution Risks

1. Full-prefix second-order autograd may fail the frozen one-hour runtime or
   80% VRAM gate on the approved GPU.
2. Full raw-chain attribution may fail the 30-minute evaluation gate.
3. The frozen `H_max/K` regularizer scale may impede pure-MSE learning; lambda
   cannot be tuned in this task.
4. Correct fixed-target diagnostics may not reproduce legacy total-objective
   trends.
5. Total raw-chain scoring may reveal semantic information without making the
   repair competitive.
6. Confirmation remains sealed regardless of pilot outcome and requires a new
   GPT-reviewed release token.
