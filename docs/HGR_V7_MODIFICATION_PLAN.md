# HGR V7 修改方案：Goal-conditioned Topological Overlay

> 文档状态：最终方法已收敛；V7.1-C strict 为已验证行为，轻量 goal-conditioned overlay 为最终方法表达；所有动态调分与 commitment 仅保留为失败消融  
> 基础版本：Route V6 Hybrid Goal Topo Map  
> 方法性质：training-free，不训练新模型  
> 开发数据：固定 GOAT-Bench train 子集  
> 最终原则：开发阶段不使用官方 benchmark 调参

> 阶段执行顺序：以 `docs/HGR_V7_STAGED_ROADMAP.md` 为唯一实施路线图。本文保留完整方法设计，
> `docs/HGR_V7_1_DESIGN.md` 保留 V7.1 细节；若旧步骤编号与路线图冲突，以路线图为准。

## 0. 最终实施结论（2026-09-01）

最终 V7 不再继续堆叠在线状态变量。论文主线固定为：

```text
Persistent Topological Memory
        +
Goal-conditioned Topology Overlay
        +
Cost-aware Conservative Decision
```

具体实现为：

1. 第一层 `PersistentBeliefTopology` 保存与目标无关的场景节点、稳定身份、可达性和导航边；
2. 第二层不是独立动态图，而是第一层在当前 goal 下的临时投影；
3. Object、Description、Image 共用 DIRECT / REVISIT / EXPLORE 候选空间；
4. 每个候选只公开 Goal Evidence、Belief、Cost、Validity 四组变量；
5. `OBSERVED_AT` 决定历史证据的可执行 anchor，`NAVIGABLE` 决定 cost/reachability/fallback；
6. frozen C-strict gate 直接消费 overlay，决定历史 REVISIT 是否值得介入；
7. Pathfinder/TSDF 只负责“怎么去”，Topo 负责“去哪里、为何去、历史证据能否复用”。

对应最终入口为 `build_goal_conditioned_topology_overlay()` 和
`cfg/eval_goatbench_goal_topomap_v7_final_qwen3vl_dashscope_train.yaml`。该 overlay 复用原始不可变
`MetricAction`，不修改 relevance、成本或候选映射，因此与已经完成完整开发集评测的 C-strict
行为严格等价；新增内容是方法边界、统一接口和可审计 trace，而不是新的调参规则。

没有进入最终方法的实验：

- V7.1-D commitment：相对 C-strict 的 Distance SR/SPL 下降 12.24/7.65 pp；
- 旧 `lightweight_goal_confidence`：完整集下降 6.36/7.59 pp；
- DynamicGoalTopology 第一版：完整集下降 4.05/3.40 pp，负证据远多于有效传播；
- Dynamic R2：快速集 Distance SR/SPL 仍下降 6.12/1.25 pp，传播改变 0 个动作。

这些实现和配置保留用于复现实验，但不能被默认配置继承，也不再作为后续加规则的起点。

最终 Overlay smoke 已通过（`00062-ACZZiU6BXLz` episode 0，8 subtasks）：51 个决策步均同时
生成 MetricAction、Goal Overlay 和 C-strict gate 事件；共投影 1337 个有效候选、0 个无效候选，
其中 DIRECT/REVISIT/EXPLORE 为 72/1122/143。Gate 接受 5 次、拒绝 46 次；lightweight 与
DynamicGoalTopology 事件均为 0，VLM failure 为 0，所有 SR/SPL 指标有限。132 项离线测试通过。
该 smoke 只验证接入正确性，不用于论文性能结论。

不重叠 internal validation 已在查看结果前封存：12 scenes × 2 episodes、170 subtasks，manifest
SHA-256 为 `a96fcce33f26df9876dd22cf58a1dd3d3ead7cef76f2797174f990e132d6696b`。
数据、两套冻结配置、通过条件与执行纪律见 `docs/HGR_V7_INTERNAL_VALIDATION_PROTOCOL.md`。

## 1. 目标与研究问题

V7 不重新构造一套与 HGR 并行的 Belief Planner，也不替换 HGR 的感知、地图和低层导航。
它在 V6 已有持久场景拓扑上解决两个已经被实验确认的问题：

1. 如何把持久 Topo 中的历史 observation 和 frontier 投影到当前目标；
2. 历史证据相对当前 observation/exploration 是否值得支付导航成本。

方法定义：

> 在 HGR 的感知、Snapshot/Frontier 和 TSDF 导航基础上，维护与目标无关的持久 Scene
> Topology；针对当前目标把 DIRECT、REVISIT、EXPLORE 投影成共享的 Evidence、Belief、Cost、
> Validity 候选，并由保守门控决定何时复用历史证据。

核心研究问题：

> 持久场景拓扑经过目标条件投影和成本/有效性约束后，能否在保持或提高 SR 的同时恢复或提高
> SPL，并减少无收益的历史回访？

## 2. V6 实验依据

当前固定 train 开发子集包含 12 个场景、每场景 2 个 episode，共 24 episodes、173 subtasks。

| 指标 | Baseline | V6 | V6 变化 |
|---|---:|---:|---:|
| Snapshot SR | 25.43% | 29.48% | +4.05 pp |
| Distance SR | 60.69% | 61.27% | +0.58 pp |
| Distance SPL | 约 44.40% | 37.19% | -7.21 pp |
| 平均总帧数 | 77.72 | 99.24 | +27.7% |
| 平均保留 Snapshot | 6.64 | 4.94 | -25.7% |

按目标类型的 Distance SR：

| 目标类型 | Baseline | V6 | 变化 |
|---|---:|---:|---:|
| Object | 60.00% | 61.82% | +1.82 pp |
| Description | 63.64% | 67.27% | +3.64 pp |
| Image | 58.73% | 55.56% | -3.17 pp |

V6 trace 表明：

- stable-ID match rate 为 90.43%，稳定身份不是当前第一瓶颈；
- 281 次 Topo 路线规划生成 223 个中间航点，223 个航点全部到达；
- 67 次路线承诺中 43 次完成、24 次取消，取消率约 35.8%；
- 没有 route fallback 和 blocked-target retry；
- 共发生 2437 次 VLM 调用，平均每个 subtask 约 14.1 次。

因此主要问题不是“路线不可执行”，而是“可执行路线不一定值得走”。最终 V7 只保留已经验证的
成本感知保守决策，不再加入未验证的复杂置信度建模。

## 3. 冻结范围

为保持实验归因清晰，V7 第一轮冻结：

- YOLO-World、SAM、ConceptGraph；
- Snapshot 生成与聚类、Frontier 提取；
- HGR Hypothesis Graph 与语义预测；
- TSDF、Habitat Pathfinder；
- Qwen3-VL 模型、endpoint、图像预算和 prompt 基本格式；
- GOAT success distance 和最终成功判定；
- 相机、基础规划器、最大步数和随机种子。

V7 只修改：

1. Scene Topology 保存的导航统计；
2. Goal Overlay 的证据、belief、cost、validity 和候选组织；
3. REVISIT 的路径代价估计与保守干预门控；
4. 直接相关的 trace、指标统计和配置。

## 4. 总体架构

```text
HGR perception
RGB / Depth / Semantic / YOLO / SAM
                |
                v
ConceptGraph / Snapshot / Frontier / Hypothesis
                |
                v
Persistent Scene Topology                 current goal + agent state
goal-independent                          goal-specific, temporary
VISITED / OBSERVED / FRONTIER / HYPOTHESIS          |
NAVIGABLE / OBSERVED_AT / DEPENDS_ON                 |
                |                                    |
                +----------------+-------------------+
                                 v
                 Goal-conditioned Topology Overlay
                   evidence / belief / cost / validity
                                 |
                                 v
                     Conservative Intervention Gate
                       reject -> original HGR order
                       accept -> historical reuse allowed
                                 |
                                 v
                    Qwen semantic choice or DIRECT guard
                                 |
                                 v
                    CURRENT / OBSERVED / FRONTIER
                                 |
                    +------------+------------+
                    |            |            |
                  DIRECT       REVISIT      EXPLORE
                    |            |            |
                    +------------+------------+
                                 v
                   TSDF + Pathfinder metric execution
                                 |
                                 v
                         new observation
```

### 4.1 Scene Topology 与 Goal Topology

Scene Topology 只保存目标无关的环境事实：

- 真实走过的位置与连通关系；
- Snapshot 的拍摄位置、视觉特征、类别和来源；
- Frontier 的稳定身份、位置和视觉特征；
- 导航边长度、验证时间和成功/失败统计。

Goal Topology 是当前 subtask 的临时 overlay，保存：

- Goal Evidence：证据来源和原始 relevance；
- Belief：冻结的 goal relevance，不进行时间衰减或 Beta 更新；
- Cost：当前 metric execution cost；
- Validity：source mapping、reachability 和 route mode。

overlay 不保存正负计数、freshness、uncertainty 或 commitment，也不把 goal-specific 状态写回
Scene Topology。

### 4.2 统一候选与模态证据

统一三类候选：

```text
CURRENT observation   -> DIRECT
historical OBSERVED   -> REVISIT
current FRONTIER      -> EXPLORE
```

DIRECT、REVISIT、EXPLORE 只表示执行类型，不建立三套独立 planner。但允许使用
modality-specific evidence encoder：

- Object：类别匹配、检测置信度、CLIP、HGR semantic；
- Description：text-image similarity、跨视角一致性；
- Image：image-image similarity、跨视角视觉一致性。

证据编码后统一映射到 `[0,1]` belief，再进入同一个保守门控；最终方法没有 goal-type-specific
阈值或三套导航逻辑。

## 5. 分阶段实施计划

每阶段使用独立配置、结果目录和 trace 字段，不覆盖 V6。只有当前阶段通过验收，才进入下一阶段。

### 5.1 V7.0：评测与诊断修复

> 实施状态（2026-08-29）：已完成。V7.0 仅增加评测、manifest 和 trace，导航策略继承
> V6 Hybrid，不改变候选选择或运动控制。

#### 目标

保证后续 SPL、路径成本和 paired comparison 可用，避免算法变化与统计错误混在一起。

#### 修改内容

1. 修复 baseline 与新版本 SPL 非有限值：
   - 失败任务 SPL 固定为 0；
   - 成功但 GT distance 非有限时单独标记；
   - 输出 finite count、invalid count 和处理规则。
2. 为 baseline/V7 统一记录每个 subtask 的：
   - success、SPL、实际路径长度、GT geodesic distance；
   - 总帧数、Snapshot 数、VLM logical calls；
   - 候选 source index 与稳定 topo ID。
3. 增加 route 诊断：
   - 历史拓扑路线长度；
   - Pathfinder 当前直达距离；
   - route detour ratio；
   - commitment 创建、保持、切换、完成和取消原因。
4. 固定 episode manifest，禁止仅依赖动态 ratio 排序重建开发集。

#### 主要文件

- `src/logger_goatbench.py`
- `run_goatbench_evaluation.py`
- `scripts/summarize_belieftopo_traces.py`
- 新增固定开发/验证 manifest 生成和校验脚本

#### 验收条件

- 所有 SR/SPL 均为有限值；
- baseline 与 V6 subtask ID 集合完全一致；
- 173 个开发 subtasks 可逐项 paired comparison；
- trace 可重建每次 REVISIT 的直接距离、Topo 距离和最终结果；
- 指标修复不得改变 agent 行为。

#### 已落地内容

- 固定清单：`cfg/manifests/goat_train_dev_12x2_seed77.json`，12 scenes × 2 episodes；
- 运行配置：`cfg/eval_goatbench_goal_topomap_v7_0_diagnostics_qwen3vl_dashscope_train.yaml`；
- 每个 split 输出 `subtask_metrics_*.json`，聚合输出 `subtask_metrics.json`；
- 聚合输出 `metric_validity.json`，非有限 SPL 明确计数并保守记为 0；
- `scripts/summarize_goatbench_results.py` 支持无 Topo trace 的 baseline 与 V7 配对；
- 旧 baseline 与 V6 已验证 173/173 subtask ID 完全一致；
- 94 个离线单元测试通过，固定 manifest 的 24 个 episode 均通过真实数据校验。

### 5.2 V7.1：Metric-grounded Conservative Reuse

> 详细冻结前设计见：`docs/HGR_V7_1_DESIGN.md`。该设计明确了 metric action、固定 capture
> anchor、direct-first 执行、保守介入门控和短视野 target commitment；最终阈值等待 V7.0
> route diagnostics 后确定。

#### 目标

只解决已经确认的路径效率问题。暂时复用 V6 relevance，不加入新 confidence、negative evidence
或 topology propagation。

#### A. 真实 metric cost

对每个候选计算当前状态下的 Pathfinder geodesic distance：

```text
CURRENT   -> 当前目标 observation point
OBSERVED  -> Snapshot 的固定 capture anchor / executable observation point
FRONTIER  -> 当前 Frontier position
```

Topo hop 只作为连通性和安全恢复信息，不再作为主要成本。不可达候选不进入 Top-K，但不永久删除
Scene Topology 节点。

#### B. 保守干预门控

Goal Topology 只有在自身具有明确意见时才改变 HGR 候选组织。门控至少检查：

- Top-1 relevance 是否达到最低可信条件；
- Top-1 与 Top-2 margin；
- 当前可达性；
- metric cost 是否超过剩余预算；
- 当前 HGR 是否已有低成本可执行方案。

门控不通过时保留原 HGR 候选集合、顺序、prompt 和直接导航。阈值只能在固定 train-dev 调整，
进入内部 validation 前冻结。

#### C. Pathfinder 直达优先

选择历史 OBSERVED 后：

1. 先查询当前位置到 capture anchor 的 Pathfinder 路径；
2. 可以安全直达时不生成历史 VISITED 中间航点；
3. 只有直达无效或不稳定时，才使用 Scene Topology 提供安全连接；
4. 记录 shortcut 和估计节省距离。

#### D. Target Commitment

将 route commitment 改为 semantic/topological target commitment：

```text
commit target
  -> execute one short segment
  -> observe/update
  -> recompute target value
  -> keep unless new target exceeds current + hysteresis margin
```

允许取消的条件：当前精确 Object 检测出现、mapping 消失、目标不可达、新负证据、连续无进展、
新候选显著更好或 subtask 结束。

Object 当前精确检测优先继续保留，作为零延迟 DIRECT guard。

#### 三类统一策略

V7.1 的 Object、Description、Image 共用同一个 `DIRECT / REVISIT / EXPLORE` 动作空间、metric
cost、intervention gate、direct shortcut 和 target commitment。三类只在 evidence adapter 上
不同：类别/文本/图像分别编码成 `[0,1]` relevance，进入动作层后使用同一组阈值和状态机。
Image original-HGR 仅作为安全消融，用于定位 V6 的 Image 负迁移是否来自证据质量，不作为
V7.1 主方法的硬编码分支。

#### 主要文件

- `src/persistent_belief_topology.py`
- `run_goatbench_evaluation.py`
- `src/query_vlm_goatbench.py`
- 新增 `cfg/eval_goatbench_goal_topomap_v7_1_*.yaml`
- 新增 cost gate、shortcut、target commitment 单元测试

#### 验收条件

在当前 173-subtask train-dev 上：

- Distance SR 不低于 V6 超过 1 pp；
- Distance SPL 相对 V6 至少提高 3 pp；
- 平均总帧数相对 V6 至少下降 10%；
- Image Distance SR 不低于 paired baseline 超过 2 pp；
- route cancellation rate 明显低于 35.8%；
- 不增加 VLM HTTP failure；
- 所有新增选择都能由 trace 中的 gate、cost 或 commitment 解释。

未满足时不进入 V7.2。

### 5.3 V7.2：Goal Confidence 与正负证据

> 历史失败消融：第一版和 R2 均未超过 C-strict。以下内容保留为实验设计记录，不进入最终
> 配置，也不再据此增加状态或阈值。

#### 目标

在 V7.1 恢复路径效率后，将单次 similarity/relevance 升级成动态、可解释、goal-specific 的
confidence。

#### 数据结构

建议新增独立 `GoalEvidenceState` overlay：

```python
goal_key
node_id
local_evidence
positive_support
negative_support
search_coverage
observation_count
independent_view_count
association_quality
last_evidence_step
confidence
uncertainty
```

`NAVIGABLE` 边按需补充：

```python
metric_length
success_count
failure_count
last_validated_step
traversability
```

已有字段能表达同一含义时不得重复增加。

#### Local Evidence

第一版只使用少量可归一化信号：

```text
Object:      exact category / detector confidence / CLIP / HGR semantic
Description: text-image CLIP / independent-view consistency
Image:       image-image CLIP / independent-view consistency
```

不同来源先映射到 `[0,1]`，不得把原始 YOLO confidence、CLIP cosine 和 semantic probability
未经校准直接相加。

#### Positive Evidence

重复 Snapshot 或同一位置小角度重复观测不能作为独立证据无限累积。独立视角至少根据位置、朝向、
step 和 Snapshot source identity 去重。

#### Negative Evidence

只能来自 agent 在线可观测事实，禁止使用 GT 和最终评测 success：

```text
negative support += search coverage * observation quality * repeated target absence
```

约束：

- 仅经过不能形成强负证据；
- 必须达到最小多视角覆盖；
- detector absence 只提供保守负证据；
- Description/Image 负证据弱于精确 Object 类别；
- 按 goal key 隔离并随时间衰减；
- 单次误检测或 VLM 判断不得永久 suppress 区域。

#### 输出与验收

trace 必须输出 confidence、uncertainty、evidence 来源、相对上一步变化、gate 结果和原因。

验收条件：

- 相对 V7.1，Distance SR 提高至少 1 pp，或 SR 持平但 Distance SPL 再提高至少 2 pp；
- repeated REVISIT 和充分搜索区域回访率下降；
- confidence 变化均能关联到实际 evidence；
- 删除 negative evidence 后出现可解释差异；
- 任一目标类型下降不超过 2 pp。

### 5.4 V7.3：Typed-edge Topology Propagation

> 历史失败消融：实现曾保留一次 typed-edge 正支持，但 R2 中传播改变 0 个动作，未形成独立
> 因果贡献。最终 V7 不做消息传播，只保留用于 source mapping 和 metric validity 的边。

#### 目标

验证拓扑能否把局部证据合理传递给相邻可执行区域，而不是扩大错误置信度。

#### 传播规则

第一版采用单步、归一化、边类型感知传播：

\[
\widetilde C=(1-\lambda)C+\lambda D^{-1}WC
\]

约束：

- 输出裁剪到 `[0,1]`；
- 默认只传播 1 hop；
- `OBSERVED_AT` 权重最高；
- `DEPENDS_ON` 只在绑定 Frontier/Hypothesis 间传播；
- `NAVIGABLE` 仅提供距离衰减的弱支持；
- `SPATIAL_NEAR` 不跨明显不同区域传播高置信语义；
- correlated evidence 去重；
- positive/negative evidence 分开传播。

#### 消融与验收

比较无传播、只 `OBSERVED_AT`、`OBSERVED_AT + DEPENDS_ON`、全部允许边 1 hop，以及 1/2 hop。

验收要求：相对 V7.2 至少一项主要 SR/SPL 提高 1 pp，其余下降不超过 1 pp，且 false hotspot
和错误区域重复访问不增加。该条件未满足，因此最终方法移除 propagation。

### 5.5 V7.4：模态一致性与 Image 证据消融

> 历史诊断计划：最终 V7 已固定为三类共用同一个静态 overlay 和 strict gate；Image 不建立
> 独立导航器，也不使用动态置信调分。

比较：

```text
Image-A: 完全原 HGR
Image-B: confidence projection-only
Image-C: confidence + cost-aware projection
Image-D: high-confidence、low-cost 条件下允许 REVISIT
```

优先保证 Image SR 不低于 baseline 超过 2 pp。这些分支用于定位 Image evidence adapter 与
统一 gate 的问题；主方法仍保持三类共用同一动作和执行框架。若统一版本不能消除 Image 负迁移，
应修复证据归一化或判定 V7 未通过，而不是在主方法中硬编码另一套 Image 导航器。

## 6. 建议代码结构

### `src/persistent_belief_topology.py`

保留 Scene Topology 节点/边、stable frontier matching、OBSERVED capture anchor、连通性、导航统计
和基础 trace。减少具体 GOAT 目标策略和 subtask confidence 状态。

### `src/metric_goal_execution.py`

负责：

- DIRECT / REVISIT / EXPLORE 的 metric candidate construction；
- Goal Evidence / Belief / Cost / Validity 四变量 overlay；
- conservative intervention gate；
- direct-first route execution。

`src/dynamic_goal_topology.py`、`src/lightweight_goal_confidence.py` 和 target commitment 相关代码
保留为失败消融复现，不由最终配置启用。

### `run_goatbench_evaluation.py`

最终只负责流程编排：

```python
scene_topology.update(...)
task_view = scene_topology.build_goal_task_view(...)
metric_view = build_metric_action_view(task_view, ...)
goal_overlay = build_goal_conditioned_topology_overlay(metric_view)
decision = evaluate_revisit_intervention_gate(goal_overlay, ...)
execute_with_tsdf_and_pathfinder(execution)
goal_topology.update_from_observation(...)
```

### `src/query_vlm_goatbench.py`

负责 Top-K 到原 HGR Snapshot/Frontier 的可逆映射。V7 不把整张 Scene Topology 塞入 prompt。

## 7. 配置设计

每阶段使用独立配置：

```text
cfg/eval_goatbench_goal_topomap_v7_1_metric.yaml
cfg/eval_goatbench_goal_topomap_v7_2_confidence.yaml
cfg/eval_goatbench_goal_topomap_v7_3_propagation.yaml
cfg/eval_goatbench_goal_topomap_v7_4_image.yaml
```

建议结构：

```yaml
active_topology:
  enabled: true
  mode: dynamic_goal_topology

  intervention:
    enabled: true
    min_confidence: null
    min_margin: null
    top_k: 5

  metric_cost:
    use_pathfinder: true
    direct_shortcut: true
    reject_unreachable: true
    normalize_by_remaining_budget: true

  target_commitment:
    enabled: true
    segment_horizon: 1
    hysteresis_margin: null
    cancel_on_live_object: true
    cancel_on_no_progress: true

  confidence:
    enabled: false       # V7.2 开启
    positive_evidence: true
    negative_evidence: true
    multi_view_dedup: true

  propagation:
    enabled: false       # V7.3 开启
    max_hops: 1
    normalized: true
    typed_edges: true

  goal_behavior:
    object: dynamic
    description: dynamic
    image: dynamic
```

`null` 表示实施时必须先定义量纲和开发集选择流程，不能沿用未经解释的经验阈值。

## 8. 测试计划

### 8.1 单元测试

至少覆盖：

- metric cost 与不可达处理；
- direct shortcut；
- HGR fallback 不改变原候选顺序；
- Top-K/source index 可逆映射；
- CURRENT/OBSERVED/FRONTIER 执行分类；
- target commitment 保持、切换和 hysteresis；
- goal switch 后 evidence 隔离；
- independent-view 去重；
- coverage 不充分时不得形成强负证据；
- propagation 归一化、有界、限制 hop；
- 三类目标共享同一 MetricAction、gate 和 commitment；
- Image original-HGR 仅作为诊断消融；
- `active_topology.enabled: false` 与 baseline 行为兼容。

### 8.2 Smoke test

每阶段正式运行前：

1. 1 scene × 1 episode，包含三类目标；
2. 无 API 400、exception、非有限指标；
3. trace 能解释每次选择；
4. 至少出现一次阶段新增行为，否则 smoke 无效。

### 8.3 数据划分

- Train-dev：继续使用当前固定 12 scenes × 2 episodes，只用于开发和少量阈值选择；
- Internal validation：GOAT train 中另外固定至少 12 个不重叠场景，每场景至少 2 episodes；
- Final benchmark：方法冻结后完整运行 `val_seen`、`val_seen_synonyms`、`val_unseen`。

不得根据单个失败任务增加 scene-specific 或 category-specific 规则；validation 运行后不得继续逐
episode 调参。

## 9. 实验矩阵

| 编号 | 方法 | 目的 |
|---|---|---|
| B0 | 原 HGR + 同一 Qwen | 公平 baseline |
| B1 | V6 Hybrid | 当前方法基准 |
| A1 | V7.1 metric execution only | 验证成本与 shortcut |
| A2 | A1 + intervention gate | 验证保守介入 |
| A3 | A2 + Goal-conditioned Overlay | 最终 V7；四变量统一方法表达，行为等价 A2 |
| F1 | A2 + target commitment | 失败消融 |
| F2 | A2 + lightweight confidence | 失败消融 |
| F3 | A2 + DynamicGoalTopology / R2 | 失败消融 |

必报指标：

- Snapshot/Distance SR、SPL，及三种目标分组结果；
- 路径长度、总帧数、subtask steps；
- VLM logical calls、attempts、failure 和耗时；
- DIRECT/REVISIT/EXPLORE 数量；
- REVISIT 成功率、平均距离和 detour ratio；
- commitment 完成、切换和取消率；
- Topo intervention/HGR fallback rate；
- negative evidence 次数；
- propagation 前后排名变化；
- stable-ID match rate。

同一 subtask 必须做 paired analysis：baseline-only、V7-only、both success、both failure、路径差、
候选类型变化和 Topo intervention。不能只比较总体百分比。

## 10. 验收与停止条件

### 阶段标准

V7.1：

- SR 基本不退化；
- Distance SPL 至少比 V6 提高 3 pp；
- 平均帧数至少下降 10%。

V7.2/V7.3：

- 相对前一阶段出现可复现的 SR 或 SPL 增益；
- 单个目标类型下降不超过 2 pp；
- 增益在未参与开发的 train-validation scene 上仍存在。

### 最终论文版本建议标准

相对公平 baseline：

- Distance SR 提高至少 2 pp；
- Distance SPL 不低于 baseline 超过 2 pp；
- Snapshot SR/SPL 至少一项明确改善；
- 任一目标类型不出现超过 2 pp 的系统性负迁移；
- API failure 为 0 或与 baseline 同等；
- 额外帧数和 VLM 调用开销可解释且不过度增长。

停止条件：

- V7.1 无法恢复 SPL：暂停 confidence/propagation，先修 cost 和执行；
- V7.2 只在 train-dev 提升：判定过拟合，不进入最终版本；
- propagation 产生 false hotspot 或目标类型退化：最终移除 propagation；
- Image unified 持续低于原 HGR：修复 evidence normalization/gate；仍不成立则判定统一 V7 失败；
- 需要大量 category-specific 阈值：回退到更简单版本。

## 11. 实施步骤

```text
Step 1   V7.0：固定指标、manifest、trace 和 paired summary
Step 2   V7.1-A：三类统一 MetricAction shadow view，不改变行为
Step 3   V7.1-B：只开启 Pathfinder direct-first
Step 4   V7.1-C：只增加 conservative reuse gate
Step 5   V7.1-D / confidence / negative / propagation：均已实验并判为失败消融
Step 6   将 C-strict 统一动作空间整理为四变量 Goal-conditioned Overlay
Step 7   用离线测试证明 overlay 与 C-strict decision 严格等价
Step 8   运行最终配置 1×1 smoke 并检查 overlay/gate trace
Step 9   冻结最终代码、配置、prompt、阈值和随机种子
Step 10  在未查看的 train-validation 上一次性验证
Step 11  验证通过后运行三个官方 benchmark split
Step 12  汇总主实验、拓扑/缓存消融、效率和失败动态模块
```

## 12. 完成定义

V7 只有同时满足以下条件才算完成：

1. baseline、V6、V7 使用相同数据、模型、感知和底层规划配置；
2. 所有指标有限且可以逐 subtask 配对；
3. Scene Topology 与 goal-specific overlay 职责清楚；
4. Topo 是否介入、为何介入、为何切换目标均可从 trace 重建；
5. SR 提升不以明显 SPL、帧数或某个目标类型退化为代价；
6. 结论在未参与开发的 train-validation scene 上成立；
7. 最终 benchmark 在代码和参数冻结后一次性完整运行；
8. 消融分别说明 History cache、Persistent Topology、C-strict 与 Goal Overlay 的作用；动态
   confidence、negative evidence、propagation 和 target commitment 作为负实验单列。
