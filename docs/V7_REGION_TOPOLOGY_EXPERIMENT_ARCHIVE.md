# V7 Region Topology 实验归档

更新日期：2026-09-03。本文记录 HGR 上 V7 双层动态拓扑方向的已完成
实现与开发集实验；它不是论文的最终 benchmark 结果。

## 1. 研究目标与边界

目标是在不训练、不替换 HGR 底层导航器的前提下，把原本扁平的
Snapshot/Frontier 候选选择改成层级决策：先确定值得探索的稳定区域，
再在区域内部选择局部目标。

保留的 HGR 能力：TSDF、frontier 提取、Snapshot、目标检测、局部
Pathfinder/碰撞检查、VLM、语义假设图。

新增高层状态：

1. 持久层：`VISITED` anchor、`OBSERVED` Snapshot、`FRONTIER` 及其
   `ANCHORED_TO` 等边构成 episode 内稳定拓扑。
2. 动态层：针对当前 GOAT 子任务更新 relevance、confidence、uncertainty、
   freshness、路径代价与 typed-edge support。
3. Region：以稳定 `VISITED` anchor 为 Region ID；snapshot 经 capture
   anchor、frontier 经 `ANCHORED_TO` 自动归属，无场景名或类别特例。

高层正常流程为：

```text
HGR 更新 TSDF / Snapshot / Frontier
  -> 动态层评估候选
  -> DIRECT 或严格授权 REVISIT；否则 Region EXPLORE
  -> Region 内局部选择
  -> HGR TSDF + Pathfinder 执行
```

历史 Snapshot 不能绕过既有 strict revisit gate。Frontier commitment 使用
V7.2-B 的几何重绑定，因此 TSDF 重提取后仍可持续追踪同一物理目标。

## 2. 数据与比较规则

开发快速集固定为 3 scene x 2 episode，共 49 subtask：

- 配置的 `start_ratio=0.00`、`end_ratio=0.09`、`split=1,2`；
- 所有下表 fast 结果均为同一 49 subtask，可横向比较；
- smoke 为 8 subtask，只用于行为和错误检查，不能用于版本定稿；
- 完整开发集为当前固定 train manifest 的 173 subtask（部分实验尚未重跑）。

指标采用 Distance SR 和 Distance SPL；SPL 越高代表同等成功下路径越短。

## 3. 已完成版本与结果

### 3.1 同一 fast 集（49 subtask）

| 版本 | 主要机制 | SR | SPL | 平均步数 | 结论 |
| --- | --- | ---: | ---: | ---: | --- |
| V7 revised selected gate | 保守 selected-ID gate | 81.63% | 67.18% | 4.82 | 当前绝对指标最高，但不是 Region topo 主体 |
| V7 dynamic active | 动态目标图 | 75.51% | 59.13% | 4.10 | 强基线 |
| V7.1-C strict | strict revisit gate | 77.55% | 57.82% | 6.00 | 稳定强基线 |
| **V7.4 Region Topology** | Region 筛选 + Region 内 VLM 局部选择 | **79.59%** | **55.16%** | 7.96 | 当前 Region topo 主版本 |
| V7.2-B | frontier commitment + geometric rebind | 71.43% | 49.48% | 8.82 | 低于 V7.4 |
| V7.5 Region evidence | snapshot 选中后强制同区 frontier | 71.43% | 46.02% | 7.94 | 弃用 |
| V7.4.1 frontier context | snapshot 仅上下文、只允许 frontier 动作 | 尚未跑 fast | 尚未跑 fast | 尚未跑 fast | smoke 已失败，停止 |

### 3.2 完整开发集（173 subtask）

| 版本 | SR | SPL | 平均步数 | 结论 |
| --- | ---: | ---: | ---: | --- |
| V6 hybrid | 61.27% | 37.19% | - | 当前 HGR hybrid 参照 |
| V7.1-C strict | **65.90%** | **45.26%** | 7.21 | 已完整验证的最佳版本 |
| V7 revised selected gate | 63.01% | 43.96% | 5.90 | 次优 |
| V7 dynamic active | 61.85% | 41.86% | 7.13 | 低于 strict |

尚未对 V7.4 进行完整 173-subtask 重跑，因此不能声称 V7.4 在完整开发集
优于 V7.1-C strict。

## 4. Region 变体诊断

### V7.4：保留

运行目录：
`results/exp_devfast_goatbench_goal_topomap_v7_4_region_topology_qwen3vl30b_dashscope_train_3x2/`

- 85 次 Region 决策均形成非空 Region；
- 34 次实际区域局部 frontier 选择；
- 51 次 `region_query_no_local_frontier` 回退，主要是 VLM 选择了不能按
  strict gate 执行的 snapshot；
- 尽管存在安全回退，SR/SPL 是所有 Region 版本中最高。

结论：V7.4 是目前最合理的 Region topo 实验版本。回退应视为严格历史
证据约束的安全机制，不应通过强制转换来消除。

### V7.5：弃用

运行目录：
`results/exp_devfast_goatbench_goal_topomap_v7_5_region_evidence_qwen3vl30b_dashscope_train_3x2/`

规则：区域 VLM 选择 snapshot 后，将其作为证据并强制执行同 Region 的
frontier。

- Region 局部 frontier 选择升至 56/82；其中 7 次来自 snapshot evidence；
- 回退降至 29 次；
- 但 SR/SPL 明显降至 71.43%/46.02%。

结论：更多 topo 控制并不等价于更好的导航；该强制规则会造成错误承诺。

### V7.4.1：弃用

运行目录：
`results/exp_dev_goatbench_goal_topomap_v7_4_1_frontier_context_qwen3vl30b_dashscope_train_smoke/`

规则：snapshot 仅作视觉上下文，VLM 只能选择 Region 内 frontier；Region
筛选也只保留含可达 frontier 的 Region。

Smoke（8 subtask）：SR 62.50%，SPL 37.53%，平均 11.88 步；19 次 Region
决策均走局部 frontier，0 次回退。

结论：完全禁止 snapshot 动作导致过度探索，明显劣化；不跑 fast/完整集。

## 5. 当前代码与配置

主要实现：

- `src/persistent_belief_topology.py`：持久 anchor 与拓扑边；
- `src/dynamic_goal_topology.py`：目标相关动态层；
- `src/region_topology.py`：Region 聚合与排序；
- `src/dual_layer_topology.py`：高层 DIRECT/REVISIT/EXPLORE 与 frontier
  commitment；
- `run_goatbench_evaluation.py`：运行时决策、trace、配置门控；
- `src/query_vlm_goatbench.py`、`src/eval_utils_gpt_goatbench.py`：可选的
  frontier-only prompt（仅用于 V7.4.1 消融）。

关键配置：

| 配置 | 用途 | 状态 |
| --- | --- | --- |
| `eval_goatbench_goal_topomap_v7_4_region_topology_...` | V7.4 Region 主版本 | 保留 |
| `eval_goatbench_goal_topomap_v7_5_region_evidence_...` | snapshot 强制延续 | 弃用消融 |
| `eval_goatbench_goal_topomap_v7_4_1_frontier_context_...` | frontier-only 上下文 | 弃用消融 |

所有行为均由配置显式启用，旧 V6/V7 配置不受影响。

## 6. 当前推荐与下一步

1. 不再叠加新的 Region 规则；V7.5 和 V7.4.1 已提供反例。
2. V7.4 作为双层 Region topo 主版本保留。
3. V7.1-C strict 作为完整开发集上的可靠强基线。
4. 只有在冻结 V7.4 后，才运行其完整开发集、独立 internal validation 和
   官方 GOAT-Bench split；不要用最终 split 继续调规则。
5. 论文中将 Region 设计概括为“稳定区域拓扑 + 动态目标拓扑 + 层级局部
   选择”；geometric rebind 与 strict gate 作为执行安全约束/消融，而不是
   额外主贡献。

## 7. 验证状态

截至本归档，V7.4.1 加入后的完整单元测试为 155 项，通过。测试覆盖：

- Region anchor 聚合及 snapshot-only Region 排除；
- Region 内 snapshot-to-frontier 消融映射；
- frontier-only prompt 拒绝 snapshot 返回；
- 旧 ActiveTopo/GoalTopo 候选映射兼容性；
- 旧 V7.4、V7.5 配置开关隔离。
