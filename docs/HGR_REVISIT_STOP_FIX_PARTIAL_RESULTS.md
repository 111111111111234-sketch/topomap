# REVISIT 停止链路修复：C-lite 阶段性结果

日期：2026-09-15。

本报告只分析已经完整结束的 `memory-dedup` 与 `memory-cost-gate`。两项均使用
`goat_train_fast_3x2_seed77.json`，覆盖 3 个 scene、每个 scene 2 个 episode、
共 49 个 goal。`intent-persistent` 当前 split 未完整结束，`verified-ablation`
尚未运行，因此不进入本次机制选择。

对照使用同一 manifest 上此前完成的 Route-only。Route-only 来自修复前源码
指纹，但不会执行本次修改的 REVISIT 到达分支；模型在线输出并非确定性，因此
配对差值仍属于开发集单次结果，不能解释成最终统计结论。

## 结果

| 配置 | GOAT SR | GOAT SPL | Snapshot SR | 完成状态 | Step | 路程（m） | VLM 调用 |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| Route-only | 65.31 | 50.78 | 22.45 | 49 completed | 254 | 155.10 | 428 |
| Memory-dedup | 67.35 | 52.85 | 20.41 | 49 completed | 288 | 171.36 | 461 |
| Memory-cost-gate | 57.14 | 40.99 | 24.49 | 47 completed、2 exhausted | 260 | 141.29 | 602 |

Memory-dedup 相对 Route-only 的 GOAT SR 为 `+2.04 pp`，scene-block 95% CI
为 `[-5.88, 12.50] pp`；GOAT SPL 为 `+2.07 pp`，区间为
`[-4.80, 11.50] pp`。两个区间均跨零，当前只能判为成功率基本持平，不能声明
Memory 已提高性能。它同时多用 34 step、16.26 m 和 33 次 VLM 调用，尚未满足
同成功率下的效率改进条件。

Memory-cost-gate 相对 Route-only 的 GOAT SR 为 `-8.16 pp`、GOAT SPL 为
`-9.80 pp`，并使用 602 次 VLM 调用。它执行了 34 次授权重选，最终产生 140 次
EXPLORE 选择；门控节省的路程没有转化为成功率或调用效率。当前证据支持从候选
配置中淘汰这一成本门控实现。

按 goal 类型看，Memory-dedup 的 description SR 为 82.35%，高于 Route-only
的 70.59%；image SR 从 61.54% 降至 53.85%，object SR 同为 63.16%。该差异
样本很小，需等待 Intent/Verified 和独立评测，不据此新增针对 goal 类型的策略。

Snapshot SR 不适合作为本轮主判断：Memory-dedup 有 26/49 个 goal 无可用
Snapshot-ID 映射，Memory-cost-gate 有 25/49 个。主指标继续使用要求合法停止且
满足距离成功的 GOAT SR/SPL；Snapshot SR 只用于实体映射诊断。

## 修复有效性

| 配置 | 旧 GOAT SR | 新 GOAT SR | 旧 → 新完成数 | Step 变化 | 路程变化 | VLM 调用变化 |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| Memory-dedup | 6.12 | 67.35 | 4 → 49 | 539 → 288 | 287.61 → 171.36 | 944 → 461 |
| Memory-cost-gate | 10.20 | 57.14 | 6 → 47 | 477 → 260 | 255.56 → 141.29 | 1100 → 602 |

修复后 Memory-dedup 的 45 次 REVISIT 到达全部提升为 APPROACH；
Memory-cost-gate 的 44 次也全部提升。两项均没有已知空间运动违规，分别审计
288 和 258 条运动事件。旧版主要结束原因 `no_executable_candidate` 从 42/41
次降到 0/2 次，证明此前的主要退化确实来自到达后释放 intent 的停止链路缺陷。

Memory-dedup 记录 34 次历史来源选择、45 次提升和 0 次同 goal 重复证据选择，
说明跨 goal 历史来源确实参与了规划执行，同时 goal-scoped 去重没有再造成循环。

Memory-cost-gate 的两个未完成任务均为 image goal，结束原因为
`selected_source_not_executable`。其中一个在 5 step 后距目标 1.09 m，另一个在
首步没有可用 Snapshot 映射且未移动。这是来源可执行性/选择恢复问题，不是本次
REVISIT 提升再次失效。

## 当前决策

1. REVISIT 停止链路修复保留。
2. Memory-dedup 暂时保留为下一阶段参考，但尚未优于 Route-only。
3. Memory-cost-gate 当前淘汰，不进入最终候选。
4. 完成 Intent-persistent 与 Verified-ablation 后再冻结 C-lite 候选。
5. 最终候选冻结前不启动 D 批次三次重复实验。

机器可读和自动生成的完整报告位于
`artifacts/hgr_rebuild/experiment_plan/revisit_stop_fix_partial.json` 与
`artifacts/hgr_rebuild/experiment_plan/revisit_stop_fix_partial.md`。
