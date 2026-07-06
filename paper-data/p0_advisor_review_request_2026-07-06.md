科研助理：

我们重新打开了 KBS 稿件的 P0 方法审计。原因是用户把当前 PDF 给两个 web GPT 会话做预审，二者都较悲观；我在本地核对代码、Jacobian 计算和小规模 smoke 后，认为其中关于 Jacobian score semantics 的核心担忧是合理的，但项目仍可挽救。

请您优先审查下面的技术决策，而不是润色正文。

一、当前本地事实

1. 当前 cross-channel ISTF-Mamba 的图分数是 filtered-coordinate score，即 `dY/dx'`。如果 filter 做跨变量混合，它不必然等价于 raw-input chain-rule score `dY/dx`。
2. 本机 P0 audit 显示该差异不是纯理论问题：
   - ISTF-Mamba current `dY/dx'` vs raw-chain `dY/dx`：mean score corr 0.8316，top-k Jaccard 0.7843，leakage 0.0835。
   - Concat partial `dY/dx` vs total derivative through `z(x)`：mean score corr 0.5064，top-k Jaccard 0.6913，leakage 0.1195。
   - Ordinary depthwise ISTF current vs raw-chain：mean score corr 0.999978，top-k Jaccard 1.0000，leakage 0.0053。
3. 因此，我不建议继续把当前 cross-channel ISTF-Mamba 作为无修改主方法，仅靠文字解释投稿。

二、本机 smoke 结果摘要

Controlled repair smoke，`d=6`, `T=100`, `lag=3`, `max_iter=120`, seeds 0--4：

| Method | Current AUROC | Raw-chain AUROC | Semantic corr | Leakage |
| --- | ---: | ---: | ---: | ---: |
| Baseline JRNGC | 0.6225 | n/a | n/a | n/a |
| ISTF-Mamba | 0.5588 | 0.5768 | 0.8316 | 0.0835 |
| Ordinary depthwise ISTF | 0.6325 | 0.6316 | 1.0000 | 0.0053 |
| Depthwise-gated ISTF | 0.6709 | 0.6720 | 0.9999 | 0.0017 |

Factorial D2 smoke，`d=6`, `T=180`, `lag=3`, `max_iter=120`, seeds 0--2：

| Cell | Baseline | ISTF-Mamba | Ordinary depthwise | Depthwise-gated |
| --- | ---: | ---: | ---: | ---: |
| Stat+Linear | 0.8994 | 0.7635 | 0.8736 | 0.7562 |
| Stat+Nonlinear | 0.8366 | 0.7043 | 0.7808 | 0.7505 |
| NS+Linear | 0.7973 | 0.7865 | 0.8030 | 0.6575 |
| NS+Nonlinear | 0.7405 | 0.7135 | 0.7432 | 0.6058 |

解读：
- ordinary depthwise ISTF 是当前最稳的坐标保持候选；
- depthwise-gated 虽然在 controlled repair smoke 中最好，但 factorial D2 明显退化，因此暂不作为主线；
- cross-channel ISTF-Mamba 在这些 smoke 中没有足够理由继续当 unmodified main method。

三、我的建议

1. 冻结当前 KBS 稿件为 pre-P0 artifact，不进入投稿润色。
2. 将主方法 pivot 到 coordinate-preserving ISTF，优先测试 ordinary depthwise ISTF。
3. cross-channel ISTF-Mamba 后续最多作为 legacy/ablation，不再作为默认主方法。
4. 不在正文中主动暴露“我们原方法有 fatal flaw”这类内部审计语言。投稿叙事应呈现最终修复后的方法：输入空间、无辅助 side channel、坐标保持、raw-variable score semantics 可解释。
5. 在正式改稿前，先做最小 GPU benchmark 验证。

四、拟定最小 GPU benchmark

Stage 1：Controlled factorial benchmark
- Generator：D2 factorial。
- Cells：Stat+Linear, Stat+Nonlinear, NS+Linear, NS+Nonlinear。
- Suggested scale：`d=10`, `T=600`, `lag=3`, `max_iter=2000`, seeds 0--4。
- Methods：baseline JRNGC, ISTF-Mamba legacy, ordinary depthwise ISTF。
- Metrics：summary-max AUROC/AUPRC, top-k SHD/nSHD/MCC, train loss, runtime。
- Semantic check：current vs raw-chain score correlation, top-k Jaccard, leakage on selected windows。

Stage 2：Shortcut diagnostics
- Concat partial vs total-derivative score。
- Concat side-channel intervention。
- ISTF-Mamba and depthwise current-vs-raw-chain semantic alignment。

Stage 3：limited main benchmark probe
- CT-medical, NSVAR_d10, Lorenz_F40, VAR_d50 as the first small set。
- baseline, ISTF-Mamba legacy, ordinary depthwise ISTF。
- 先 3 seeds，若 promising 再扩展到 manuscript protocol。

五、请您审核的问题

1. 是否同意：当前稿件不能只靠 wording 修补，必须先解决 coordinate-preserving/raw-chain score semantics？
2. 是否同意 ordinary depthwise ISTF 是下一轮正式验证的主候选，而 depthwise-gated 暂不推进？
3. 上述最小 GPU benchmark 是否足以决定是否重写 KBS 稿件主方法？
4. 是否还需要在启动 AutoDL/GPU 前补一个本机低成本诊断？
5. 论文最终写法是否应避免主动暴露内部 cross-channel 版本的问题，只呈现 final repaired ISTF 的设计约束和验证？

当前 AutoDL/GPU 尚未启动；所有结果均为本机 CPU smoke，不是正式 benchmark。
