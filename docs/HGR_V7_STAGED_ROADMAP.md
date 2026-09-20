# HGR V7 分阶段实施路线图

> 本文档是 V7 后续开发与实验的唯一执行顺序。  
> `HGR_V7_MODIFICATION_PLAN.md` 说明完整方法，`HGR_V7_1_DESIGN.md` 说明 metric execution
> 细节；若旧文档中的阶段顺序与本文冲突，以本文为准。

## 1. 总原则

论文中的最终方法只有一个 V7：

```text
HGR perception and mapping
        ↓
Persistent Scene Topology
        ↓
Goal-conditioned Topology Overlay
Evidence / Belief / Cost / Validity
        ↓
Conservative Reuse Gate
        ↓
Frozen C-strict Direct-first Executor
```

开发时逐步启用模块，每一步只回答一个研究问题。Object、Description、Image 始终共用同一
动作空间、metric cost、gate 和 commitment；三类只允许使用不同的 evidence encoder，不允许
复制三套导航逻辑或设置 goal-type-specific 阈值。

共同约束：

1. training-free，不训练新模型；
2. 不改变 GOAT success 条件、最大步数、检测模型和底层 TSDF planner；
3. 不使用 GT、最终 success 或 benchmark 结果参与在线决策；
4. 每个行为变化必须有独立配置和独立结果目录；
5. 前一阶段通过验收后才能进入下一阶段；
6. 某阶段失败时回退，不用后续复杂模块掩盖问题。

## 2. 数据使用规则

### Train-dev

```text
manifest: cfg/manifests/goat_train_dev_12x2_seed77.json
规模:     12 scenes × 2 episodes = 24 episodes，当前为 173 subtasks
用途:     Debug、阈值选择、消融、失败分析
```

- split 1：实现后的首次运行和有限阈值选择；
- split 2：确认方向，不能根据单个 episode 增加新规则；
- 两个 split 都属于开发集，不能作为最终 benchmark。

### Internal validation

从 GOAT train 另外固定至少 12 个不重叠 scene，每个 2 episodes。最终 V7 配置和参数现已冻结，
生成 manifest 后一次性运行，不再根据该结果修改方法或参数。

已封存为 `cfg/manifests/goat_train_internal_val_12x2_seed77.json`：12 个场景、24 episodes、
170 subtasks，与开发集 scene 重叠为 0。具体 hash、通过条件和两套配置见
`docs/HGR_V7_INTERNAL_VALIDATION_PROTOCOL.md`。

### Final benchmark

完整 V7 和所有参数冻结后，才运行：

```text
val_seen
val_seen_synonyms
val_unseen
```

官方 split 不用于修改代码、阈值或 prompt。

## 3. 阶段总览

| 阶段 | 唯一新增变化 | 是否改变导航 | 核心问题 |
|---|---|---:|---|
| V7.0 | 指标、manifest、trace | 否 | 结果是否可信、可配对、可解释？ |
| V7.1-A | 统一 MetricAction shadow view | 否 | 三类能否无损映射到统一动作层？ |
| V7.1-B | Pathfinder direct-first | 是 | V6 的 SPL 损失是否主要来自历史绕路？ |
| V7.1-C | Conservative reuse gate | 是 | 什么时候值得让历史信息介入？ |
| V7-Dynamic-shadow | 动态证据图诊断 | 否 | 历史失败消融，保留复现 |
| V7-Dynamic-active/R2 | 动态置信调分 | 是 | 均未超过 C-strict，不进入最终方法 |
| V7-Final Overlay | 四变量 goal-conditioned projection | 否（行为等价 C-strict） | 明确双层方法边界、统一三类动作并提供可审计因果 trace |
| V7-Full | 冻结、验证、正式评测 | 否 | 收益能否泛化并形成论文结论？ |

当前状态（2026-09-01）：V7.0、V7.1-A、V7.1-B 和 V7.1-C 已完成；C-strict 在完整
173-subtask train-dev 上相对 V6 的 Distance SR/SPL 分别提高 4.62/8.07 pp。V7.1-D 失败；旧
V7-Compact 在完整开发集上相对 C-strict 的 Distance SR/SPL 分别下降 6.36/7.59 pp，也只保留
为失败消融。失败原因无法归因，因为它同时改变置信度和 route commitment。

DynamicGoalTopology 第一版 active 在完整 173-subtask train-dev 上相对 C-strict 的
Distance SR/SPL 下降 4.05/3.40 pp，不能采用；trace 显示 1001 次负证据但 typed-edge 仅 3 次
真正改变分数。R2 恢复 3 个独立缺失视角门槛后，快速集相对 C-strict 的 Distance SR/SPL
仍下降 6.12/1.25 pp，且传播改变 0 个动作。因此最终方法改为无动态调分的轻量 projection，
直接复用 C-strict 的不可变动作分数，并将论文级状态收缩为 Evidence、Belief、Cost、Validity。

最终配置：

```text
cfg/eval_goatbench_goal_topomap_v7_final_qwen3vl_dashscope_train.yaml
```

动态模块、R2 配置和 commitment 代码只作为失败消融保留。

最终 Overlay 的 1×1 smoke 已完成：8 subtasks、51 个 overlay/gate 对齐事件、1337 个有效候选、
0 个非有限指标、0 次 VLM failure；动态失败模块事件为 0。当前状态进入代码/参数冻结和不重叠
internal validation，不再根据 smoke 分数调参。

快速迭代可使用固定 `cfg/manifests/goat_train_fast_3x2_seed77.json`（3 场景 × 2 episode，
49 个 subtask；Image/Description/Object = 13/17/19）。快速集只用于版本筛选，所有对比方法
必须使用同一 manifest；方法冻结后仍需回到完整 12 场景 × 2 episode 和官方 benchmark。

## 4. V7.0：评测与诊断基础

### 状态

代码已完成；需要运行 V7.0 获取新的 route 和 subtask diagnostics。

### 输出

- finite-safe SR/SPL 和 `metric_validity.json`；
- 每 subtask 路径、帧数、步数、VLM 调用和最终候选；
- selected REVISIT 的 capture-anchor direct distance、topology distance 和 detour ratio；
- 固定 24-episode manifest；
- baseline/V6/V7 paired summary。

### 进入 V7.1-A 的条件

- 24 episodes 均来自固定 manifest；
- 全部聚合指标有限；
- route trace 能读取且字段完整；
- V7.0 与 V6 的行为配置完全一致。

V7.0 不根据分数决定算法方向；它只建立可靠测量。

## 5. V7.1-A：统一 MetricAction shadow view

### 唯一变化

新增统一动作表示，但不把其输出用于导航：

```text
Object evidence      ┐
Description evidence ├─> DIRECT / REVISIT / EXPLORE MetricAction
Image evidence       ┘
```

对每个动作只做 shadow 计算和 trace：

- stable topo ID 和原始 source mapping；
- `[0,1]` relevance；
- direct geodesic；
- safe-topology cost；
- effective metric cost；
- reachable、budget ratio、route mode。

实际候选、prompt、VLM 选择和导航仍使用 V7.0/V6 路径。

### 为什么先做 shadow

这是统一三类任务的接口验证，也是后续 gate 阈值的数据来源。直接一边重构动作层一边改变导航，
出现差异时无法判断是 mapping bug 还是方法效果。

### 验收

- 三类任务都能生成同构 MetricAction；
- source index/Snapshot/object ID 100% 可逆映射；
- shadow view 不改变 VLM logical calls；
- 使用固定 mock response 时，V7.1-A 与 V7.0 的动作序列完全一致；
- 无非有限 JSON 字段，UNREACHABLE 使用显式状态而不是 NaN。

该阶段只需 smoke + split 1；确认兼容后不需要为分数跑完整消融。

## 6. V7.1-B：Pathfinder direct-first execution

### 实现状态（2026-08-29）

代码已完成，待第二次真实 smoke 验收。第一次 smoke 成功暴露出 capture anchor 到达后重复选择
同一历史 Snapshot 的零位移循环（139 次 `anchor_reached`）；该结果判定为失败，不用于性能比较。
现已加入一次性 post-anchor 原 HGR 接管，避免引入 V7.1-C gate 的同时释放循环，106 项离线测试
通过。配置：

- 完整固定 train-dev：`cfg/eval_goatbench_goal_topomap_v7_1_b_direct_first_qwen3vl_dashscope_train.yaml`
- 1 scene × 1 episode smoke：`cfg/eval_goatbench_goal_topomap_v7_1_b_direct_first_qwen3vl_dashscope_train_smoke.yaml`
- 修复后独立复测：`cfg/eval_goatbench_goal_topomap_v7_1_b_direct_first_qwen3vl_dashscope_train_smoke_r2.yaml`

V7.1-B 保留 V6 的 Image `projection_only` 和三类原目标选择逻辑。执行器仅在原方法已经选中
历史 Snapshot 后介入，并记录 `direct_first_route_planned`、`direct_first_anchor_reached` 及
明确的 fallback 原因；没有加入 intervention gate、confidence 或新 VLM 调用。

### 唯一变化

保持 V6 已选出的目标不变，只改变 REVISIT 的路线执行：

```text
current -> capture anchor direct Pathfinder path
                     ↓ 不可达
             VISITED-only safe topology
                     ↓ 仍不可达
                 original HGR fallback
```

- direct 查询必须指向固定 capture anchor；
- 不把 OBSERVED、FRONTIER 或 HYPOTHESIS 当作中间 waypoint；
- 每次最多执行原 planner 的一个约 1 米 segment；
- 到达 anchor 不等于 success，必须由当前观测重新形成有效目标。

### 验收

相对 V7.0/V6：

- Distance SR 不下降超过 1 pp；
- Distance SPL 至少提高 1 pp，或 REVISIT 路径长度中位数下降至少 15%；
- direct 可达时不再选择更长历史路线；
- 不增加 VLM logical calls；
- Image、Description、Object 使用同一个 route executor。

若路线明显缩短但整体 SPL 未变化，先检查 REVISIT 覆盖率；不得立即增加 confidence。

## 7. V7.1-C：Conservative reuse gate

### 唯一变化

开始让 shadow MetricAction 影响“历史 REVISIT 是否有资格介入”。不改变 evidence 计算，不增加
confidence。

统一门控只使用：

1. REVISIT 相对当前替代动作的 relevance margin；
2. REVISIT 占剩余路径预算的比例；
3. REVISIT 与最低成本 EXPLORE 的 metric cost ratio；
4. reachable/source mapping 等硬约束。

门控失败必须完整回退该目标类型的原 HGR 输入。三个目标类型使用同一个公式和同一组参数。

### 参数纪律

最多选择三个 gate 参数：

```text
min_relevance_margin
max_budget_fraction
max_revisit_to_explore_ratio
```

- 参数候选在运行前写入小网格；
- split 1 选择，split 2 只确认；
- 禁止 category、scene 或 goal-type-specific 参数；
- 进入 V7.1-D 后冻结 gate。

### 验收

相对 V7.1-B：

- Distance SR 不下降超过 1 pp；
- Distance SPL 不下降，并优先提高；
- 被拒绝的高成本 REVISIT 比例可解释；
- Topo intervention 和 HGR fallback 均非零，避免 gate 恒开或恒关；
- 任一目标类型 Distance SR 下降不超过 2 pp。

## 8. V7.1-D：Short-horizon target commitment

> 历史消融状态：已实现并完成实验，但相对 C-strict 的 Distance SR/SPL 分别下降
> 12.24/7.65 pp，且没有产生有效 KEEP/SWITCH/COMPLETE，因此不进入最终 V7。其“每段后
> 重评估”原则由新的 DynamicGoalTopology 逐步投影保留；执行层继续使用已验证的 C-strict
> direct-first/route commitment，不采用失败的 D 状态机。

### 唯一变化

将 V6 的 route commitment 替换为稳定的 target commitment：

```text
commit topo target + capture anchor
  -> 执行一个 segment
  -> 重新观察和重算 metric cost
  -> KEEP / SWITCH / COMPLETE / CANCEL / PREEMPT
```

- 不保存脆弱的 Snapshot Python 对象；
- mapping 消失只取消旧 Snapshot 的终止资格，不立即放弃 capture anchor；
- 当前强 DIRECT 证据可以抢占；
- 小幅 relevance/cost 波动由统一 hysteresis 抑制；
- 连续无 geodesic progress 才取消；
- 三类任务使用同一个状态机。

### V7.1 总验收

V7.1-D 相对 V6：

- Distance SR 不下降超过 1 pp；
- Distance SPL 至少提高 3 pp；
- 平均帧数至少下降 10%；
- 任一目标类型 Distance SR 下降不超过 2 pp；
- harmful cancellation rate 至少下降 30%；
- VLM logical calls 不增加；
- 每个保持、切换、完成和取消都能从 trace 重建。

不满足时停在表现最好的 V7.1-B/C，不进入 confidence 阶段。

## 9. V7.2-A：Positive goal confidence

> 历史消融，已停止。与负证据/传播组成的 active 动态版本未超过 C-strict；最终 V7 不做
> Beta、freshness 或 uncertainty 调分。

### 唯一变化

在 V7.1 冻结执行层上，把单步 relevance 升级为 goal-specific confidence。三类共享同一个状态
结构和更新接口，但 evidence encoder 不同：

```text
Object:      category/detector/semantic
Description: text-image similarity
Image:       image-image similarity
```

第一阶段只加入正证据：

- 多视角支持；
- 观测质量；
- 位置/方向/source identity 去重；
- 时间衰减；
- uncertainty。

禁止同一 Snapshot 或近似相同视角无限累积置信度。

### 验收

相对 V7.1：

- Distance SR 提高至少 1 pp，或 SR 持平且 SPL 提高至少 2 pp；
- 三类 confidence 均有实际变化且来源可追踪；
- 删除 positive fusion 后结果有可解释差异；
- 任一目标类型下降不超过 2 pp。

## 10. V7.2-B：Conservative negative evidence

> 历史消融，已停止。R2 即使要求 3 个独立缺失视角，快速集仍低于 C-strict，最终方法不使用
> 缺失证据降低静态场景中的目标存在 belief。

### 唯一变化

在正 confidence 已通过后，再加入负证据。只允许使用在线可观察事实：

- 足够搜索覆盖；
- 多个独立视角；
- 合格观测中目标持续缺失；
- 导航到目标区域后仍未确认。

禁止使用 GT、最终 success、单帧 detector absence 或单次 VLM 判断永久 suppress 节点。

### 验收

- repeated REVISIT 和已充分搜索区域回访率下降；
- Distance SR 不下降超过 1 pp；
- false suppression 有明确统计并处于预设上限内；
- 去掉 negative evidence 后能解释差异。

若只有路径下降但 SR 明显下降，移除 negative evidence，保留 V7.2-A。

## 11. V7.3：Typed-edge one-hop propagation

> 历史消融，已停止。R2 中传播没有改变任何动作，不能作为主方法贡献。最终方法只保留具有
> 直接因果作用的 `OBSERVED_AT` 和 `NAVIGABLE`；不进行 message passing。

### 唯一变化

只在 V7.2 可靠后尝试传播：

- 最多 1 hop；
- edge-type aware；
- positive/negative 分开；
- correlated evidence 去重；
- 归一化并裁剪到 `[0,1]`；
- `OBSERVED_AT`、`DEPENDS_ON` 优先，`NAVIGABLE/SPATIAL_NEAR` 只提供弱距离衰减支持。

### 验收

相对 V7.2：

- 至少一个主要 SR/SPL 指标提高 1 pp；
- 其余主要指标下降不超过 1 pp；
- false hotspot、错误区域重复访问不增加；
- 2-hop 不作为默认主方法。

没有稳定独立收益时，最终 V7 不包含 propagation。这是有效的负实验，不影响论文严谨性。

## 12. V7-Full：冻结与最终实验

### 冻结顺序

1. 最终阶段已选择为 C-strict + behavior-preserving Goal Overlay；
2. 完成一次 smoke，验证 overlay trace 与 C-strict gate decision 完全对应；
3. 冻结代码、配置、prompt、阈值和随机种子；
4. 在未使用的 train-validation 上一次性验证；
5. validation 方向成立才进入官方 benchmark；
6. 官方 benchmark 运行后不再修改方法。

### 论文实验矩阵

```text
B0  Original HGR + same Qwen
B1  V6 Hybrid
A0  V7.0 diagnostics-equivalent behavior
A1  Unified MetricAction + direct-first
A2  A1 + conservative gate
A3  Final V7: A2 + goal-conditioned overlay（与 A2 行为等价）
F1  A2 + target commitment（失败消融）
F2  A2 + lightweight confidence（失败消融）
F3  A2 + DynamicGoalTopology / R2（失败消融）
```

必报总体和分目标类型的 SR、SPL、路径长度、帧数、VLM calls，以及 paired subtask 转移：

```text
fail -> success
success -> fail
both success with shorter/longer path
both fail
Topo intervention / HGR fallback
```

## 13. 立即执行顺序

当前只做下面三件事：

```text
1. 运行最终 V7 的 1×1 smoke，检查三类 overlay 与 gate trace
2. 生成与开发集 scene 不重叠的 internal-validation manifest
3. 冻结后运行 internal validation；方向成立才运行官方三个 split
```

不再实现新的 confidence、negative evidence、propagation 或 goal-type-specific 规则，也不提前
运行官方 benchmark。
