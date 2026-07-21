CODEX：

根据用户要求，我已将“提高 KBS 稳妥度”的建议落实为一套仅文档、未执行的
Phase 9 acceptance-readiness 方案。当前 v4 英文稿、中文审阅稿、Phase 8 冻结
结果、Stage 1a/P1 artifacts 均未修改；GPU 保持关闭，也未读取 Phase 7 seeds
4--8 的新输出。

本方案不再尝试复活 CP-depthwise 或设计新的 ISTF。核心增强路线是：

1. 将现有 existence proposition 扩展为四个有边界的理论结果：route
   decomposition、coordinate-preservation sufficient condition、horizon
   closure、top-k margin stability；
2. 将 coverage declaration 落实为可执行的软件审计，输出 partial/total
   raw-chain、route magnitude、coordinate leakage、tail mass、eligible-window
   counts 和五类诊断标签；
3. 使用已知正负语义 fixtures、linear VAR 和 nonlinear/nonstationary D2 做
   prospective cross-architecture validation；
4. 使用 DREAM3 known-graph benchmark 和 MoCap real no-ground-truth case 补齐
   external validity；
5. 仅固定复验已存在的 full auxiliary penalty lc10，不调参、不称最优；若其不能
   跨 regime 满足 graph/pure-MSE gate，则保留为 boundary evidence。

请审核并只对以下六项作出决定：

1. 是否接受 `KBS_SUBMISSION_READY` 与 `KBS_STRONG_READY` 两级内部验收定义，
   并同意只有后者代表本项目所说的“稳中 KBS”目标？
2. 是否接受四个理论结果的范围和措辞，尤其是 coordinate preservation 仅作为
   sufficient condition、coverage 不等于 causal identifiability？
3. 是否接受 audit API、`M_missing`、coordinate leakage、tail mass、score
   aggregation 和标签规则？
4. 是否接受 formal controlled design：2 DGP families × 6 profiles × 5 data
   seeds × 2 model seeds = 120 runs，以及先 3 seeds pilot、后 2 sealed seeds
   confirmation？
5. 是否接受 DREAM3 作为 known-graph external benchmark、MoCap 作为无完整图真值
   的 real audit case，并禁止将 MoCap 写成 true-edge validation？
6. 是否接受 72 GPU-hour 总上限和“先实现/CPU preflight，再申请 GPU”的执行顺序；
   DREAM3 cMLP/cLSTM 20 个 context runs 是保留还是在正式结果前整体删除？

当前我建议：`REVISION/APPROVAL_REQUIRED BEFORE IMPLEMENTATION`。顾问通过前不写
实现代码、不生成正式 run matrix、不启动任何训练、不修改 KBS v4 正文。

