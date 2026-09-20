# R5 route-only 开发集结果分析

本报告的 R3/R4/R5 均指 B 阶段 route-only，与后续 C-R3/C-R4 区分。
整体设计更新见 [主规格](HGR_DUAL_TOPO_GENERAL_DEVELOPMENT_SPEC.md)，
当前 Phase C 对照见 [三个 smoke 复核](HGR_PHASE_C_THREE_SMOKES_REVIEW.md)。

分析日期：2026-09-09。数据目录均位于 `results/`：

- R5：`exp_dev_goatbench_place_route_only_qwen3vl30b_dashscope_train_r5`
- R3：`exp_dev_goatbench_place_route_only_qwen3vl30b_dashscope_train_r3`
- baseline：`exp_dev_goatbench_hgr_baseline_qwen3vl30b_dashscope_train`

## 完成与总体结果

固定 train development 清单为 `cfg/manifests/goat_train_dev_12x2_seed77.json`。三组有相同的 173 个 subtask ID，覆盖 12 scenes、24 episodes。R5 两个日志均有 All scenes finish；指标无非有限值。

| 指标 | baseline | R3 | R5 |
|---|---:|---:|---:|
| Distance SR | 55.49% (96/173) | 60.69% (105/173) | 64.16% (111/173) |
| Distance SPL | 39.31% | 32.39% | 35.69% |
| Snapshot SR | 23.12% | 24.86% | 27.75% |
| Snapshot SPL | 17.34% | 13.63% | 15.98% |
| 平均 subtask_steps | 6.06 | 8.84 | 7.58 |
| 平均行走距离 m | 4.77 | 6.85 | 6.56 |
| VLM logical calls | 2079 | 2289 | 2207 |
| VLM failures | 0 | 25 | 0 |
| VLM 累计调用耗时 s | 9154.38 | 5181.44 | 10224.64 |

调用耗时为 telemetry 累计值，不是并行作业墙钟时间。R3 有 2389 次 HTTP attempts；R5 为 2207 次，全部成功。

R5 相对 baseline：SR +8.67 pp，SPL -3.62 pp，平均步数 +25.0%，平均距离 +37.6%。配对新增成功 24、丢失成功 9，净增 15。相对 R3：新增成功 22、丢失成功 16，净增 6；SR +3.47 pp、SPL +3.30 pp。

## 任务类型

| 类型 | N | baseline SR / SPL | R3 SR / SPL | R5 SR / SPL |
|---|---:|---:|---:|---:|
| object | 55 | 47.27 / 32.08 | 63.64 / 33.46 | 65.45 / 33.15 |
| description | 55 | 60.00 / 46.20 | 61.82 / 33.45 | 65.45 / 41.90 |
| image | 63 | 58.73 / 39.60 | 57.14 / 30.52 | 61.90 / 32.48 |

以上均为 Distance 百分数。对 baseline 的成功增益主要来自 object（+10 个），description +3、image +2；image 的 SPL 仍低 7.12 pp。

## 执行机制验收

从 R5 全部 episode JSONL 统计：

- 1311 条 step 的 Place audit 均 valid。
- 932 次 route planning：690 same_place、241 next_hop、1 disconnected。
- 121 次 commitment 创建、215 次续行、186 次 waypoint 到达、95 次进入终端段。
- 25 次 commitment 释放：24 次 `hgr_source_mapping_disappeared`、1 次 `subtask_ended_before_terminal_segment`。
- 1 次 waypoint fallback：`00179-MVVzj944atG_1_3` step 16，原因 `waypoint_invalid_in_current_tsdf`，记录边失效并回退到同一 HGR 目标终端段。
- 7 次规划的 execution_reached_place_id 与 observed_place_id 不同，说明执行进度和观测归属分离在闭环中实际触发。
- 在同一 commitment 生命周期内，以 target kind、target ID、source Place、next Place 为键检查相邻 waypoint-arrived 事件，没有连续重复同一已完成边。

因此，R4 已知的连续重复已完成边循环在本次全量 trace 中未复现；该检查不等于排除了所有形式的绕路或循环，也不能证明局部几何路径始终沿边的记录轨迹。

## 效率瓶颈

按 R5 是否至少触发一次 next_hop 划分，再对相同 ID 取 baseline：

| R5 子集 | N | baseline → R5 SR | baseline → R5 SPL | 平均步数 | 平均距离 m |
|---|---:|---:|---:|---:|---:|
| 跨 Place | 86 | 59.30 → 67.44 | 39.99 → 25.69 | 6.73 → 11.23 | 5.84 → 9.74 |
| 未触发 next_hop | 87 | 51.72 → 60.92 | 38.63 → 45.57 | 5.40 → 3.97 | 3.71 → 3.41 |

这是按运行行为事后分组的诊断，不是控制消融。跨 Place 子集集中了 SPL 损失，但尚不能把损失完全归因于 waypoint 或某一种几何设计。

两组都成功的 87 个任务中，baseline → R5 平均距离 3.67 → 5.44 m，SPL 72.44 → 56.24%。因此，成本增加不能仅由 R5 救回更多困难任务解释。

典型 scene：`00685-ENiCjXWB6aQ` 成功数 13 → 16，但 SPL -23.58 pp；`00062-ACZZiU6BXLz` 成功数 8 → 12，SPL +17.04 pp，说明收益与代价存在明显场景差异。

## 失败与后续优先级

62 个 Distance 失败中，61 个 final_choice 为 snapshot，1 个为 frontier；26 个失败任务仅执行 1–4 步。不能仅凭 final_choice 判断为误识别，但结果提示应检查目标选择、停止位置和最终验证，而不只看超时。

建议下一轮优先：

1. 逐条追踪 24 次 source mapping 消失，区分 snapshot 合并/重索引、对象消失和合法目标失效，检查稳定目标身份是否需要修复。
2. 对两组都成功但 R5 明显变长的轨迹，分解 next-hop、终端段、重新选目标的距离；重点审查 `00685-ENiCjXWB6aQ`，用控制消融确认额外路径来源。
3. 对短步失败核查 snapshot/object 与最终距离，评估第二层目标验证的缺口。
4. 保留 R5 进度分离机制，先验证上述具体问题，再进入完整 goal-navigation 与独立场景测试。

## 结论边界

本次结果支持“已知执行循环未复现，成功率提升，但路线效率仍未超过 baseline”。不能宣称完整双层方法完成或泛化提升。

以 scene 为 cluster，12 个 scene 有放回采样 10000 次，随机种子 5，每次对抽中 scene 的全部 subtask 计算加权均值差，percentile 95% CI 为：

| 对照 | SR 差 pp | SPL 差 pp |
|---|---:|---:|
| R5 − baseline | [2.50, 14.97] | [-9.97, 3.21] |
| R5 − R3 | [-2.94, 10.06] | [-2.67, 10.02] |

这些区间只反映该开发集的场景抽样波动，不包含重复 API 运行方差。R3 的 25 次模型失败、历史运行条件和代码版本差异使比较不能作为严格单变量因果实验。当前工作区有未提交改动，不能用当前 HEAD 代替各历史运行的源码快照。该清单已反复用于开发，不属于独立验证集。
