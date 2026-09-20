# REVISIT 停止链路修复后的完整 C-lite 结果

日期：2026-09-16。

本轮 `revisit_stop_fix_v1` 已完成 Memory-dedup、Memory-cost-gate、
Intent-persistent 和 Verified-ablation 的两个 split。每项覆盖同一开发清单中的
3 个 scene、每个 scene 2 个 episode，共 49 个 goal。Route-only 使用同一清单上
此前完成的结果；它不经过本次修改的 REVISIT 到达分支。

在线模型输出并非确定性，且只有 3 个 scene，因此单次点估计不能单独证明性能
提升。以下判断同时依据配对差值、scene-block 置信区间、结束质量和运行成本。

## 一、主结果

| 配置 | GOAT SR | GOAT SPL | Snapshot SR | Distance SR | 状态 | Step | 路程（m） | VLM 调用 |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| Route-only | 65.31 | 50.78 | 22.45 | 65.31 | 49 completed | 254 | 155.10 | 428 |
| Memory-dedup | 67.35 | 52.85 | 20.41 | 67.35 | 49 completed | 288 | 171.36 | 461 |
| Memory-cost-gate | 57.14 | 40.99 | 24.49 | 57.14 | 47 completed、2 exhausted | 260 | 141.29 | 602 |
| Intent-persistent | 61.22 | 44.28 | 26.53 | 63.27 | 47 completed、2 exhausted | 444 | 252.44 | 518 |
| Verified-ablation | 71.43 | 34.38 | 28.57 | 75.51 | 40 completed、8 exhausted、1 error | 629 | 334.94 | 697 |

合法停止与距离成功共同构成 GOAT SR。Intent 的 Distance SR 比 GOAT SR 多一个
成功，是因为一个任务在距目标 0.34 m 时耗尽步数，没有完成停止协议。Verified
的 Distance SR 比 GOAT SR 多两个成功，同样来自未合法停止但最终距离较近的任务。

Snapshot SR 继续作为实体映射诊断。各配置有 19–27 个 goal 无可用 Snapshot-ID
映射，不能用该指标替代 GOAT SR/SPL。

## 二、与 Route-only 的配对结论

| 候选 | GOAT SR 差值（pp） | scene-block 95% CI | GOAT SPL 差值（pp） | scene-block 95% CI |
| --- | ---: | --- | ---: | --- |
| Memory-dedup | +2.04 | [-5.88, 12.50] | +2.07 | [-4.80, 11.50] |
| Memory-cost-gate | -8.16 | [-25.00, 5.88] | -9.80 | [-26.43, 5.11] |
| Intent-persistent | -4.08 | [-12.50, 11.76] | -6.50 | [-23.91, 7.97] |
| Verified-ablation | +6.12 | [-12.50, 23.53] | -16.41 | [-39.95, -3.99] |

Memory-dedup 只多成功 1 个 goal，成功率和 SPL 的区间均跨零，同时增加 13.4%
step、10.5% 路程和 7.7% VLM 调用。它没有达到“成功改善有统计支持”或“成功相同
时成本下降”的候选门槛。

Memory-cost-gate 执行 34 次授权重选并产生 602 次调用。它虽然少走 13.82 m，
但成功率、SPL 和调用成本同时退化，应淘汰当前成本门控实现。

Intent-persistent 将 HGR 选择数从 Memory-dedup 的 152 次降到 97 次，说明持续
intent 确实减少了语义重选；但长路线承诺使运动增至 444 step、252.44 m，最终
GOAT SR 和 SPL 都低于 Route-only。机制生效，但没有转化为导航收益。

Verified 多成功 3 个 goal，但 SR 区间跨零；GOAT SPL 在三个 scene 上全部下降，
95% 区间完全低于零。它使用 99 次终点验证、697 次总调用和 334.94 m 路程，成本
不可接受，因此不能仅凭最高 SR 点估计入选。

## 三、Verified 审计

99 次验证返回 66 次 confirmed、20 次 rejected、13 次 uncertain；其中 34 次要求
新视角，40 个任务最终获得 stop。历史来源需要两视角，所以 confirmed 次数大于
最终停止数。

40 个确认停止中，35 个满足真实距离成功，距离口径确认精度为 87.5%；另有 5 个
确认后仍离目标过远。Snapshot 映射口径中，14 个匹配、12 个不匹配、14 个无法
建立映射。映射缺失使 Snapshot 误确认数不能直接视为验证器精度，但距离结果已经
证明终点验证仍会产生错误停止。

Verified 有 8 个 exhausted 和 1 个 `semantic_selection_failed`。其高 SR 来自较多
探索和验证，不是更高效的拓扑复用。

## 四、记忆和执行正确性

- Memory-dedup：34 次历史来源选择、45 次 REVISIT 提升、0 次同 goal 重复证据。
- Intent-persistent：27 次历史来源选择、45 次 REVISIT 提升、0 次重复证据。
- Verified：38 次历史来源选择、99 次终点验证。
- 四项结果均保留跨 goal 证据；最后一个 goal 开始时仍有 prior-goal evidence。
- 所有记录到的已知空间运动均通过审计：Memory-dedup 288、Memory-cost-gate 258、
  Intent 443、Verified 622 条，违规数均为 0。

因此，Memory 确实维护并使用了跨 goal 历史来源，topo/known-space 路线也实际参与
执行。当前瓶颈已经从“到达后不能停止”转为语义选择质量、长 intent 路线成本和
Verified 的验证效率。

## 五、C-lite 决策

1. 保留 REVISIT → APPROACH 停止链路修复。
2. 保留 Memory-dedup 作为后续研究对照，但不宣称优于 Route-only。
3. 淘汰 Memory-cost-gate、Intent-persistent 和当前 Verified-ablation。
4. 按计划的候选门槛，本轮没有新配置可靠优于 Route-only；当前正式版本继续使用
   Route-only。
5. 若现在进入 D，使用 `--final-candidate route-only`，只在独立清单上评测
   Baseline 与 Route-only；不应把已淘汰的 Verified 当作最终候选。

完整自动汇总位于
`artifacts/hgr_rebuild/experiment_plan/revisit_stop_fix_complete.json` 和
`artifacts/hgr_rebuild/experiment_plan/revisit_stop_fix_complete.md`。
