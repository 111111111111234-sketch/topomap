# HGR V7.1 设计：Metric-grounded Conservative Reuse

> 状态：设计冻结前草案，等待 V7.0 route diagnostics 后确定少量阈值  
> 基础版本：V7.0 / V6 Hybrid Goal Topo Map  
> 方法性质：training-free；不增加模型训练，不增加新的 VLM 请求  
> 作用范围：Object、Description、Image 共用同一决策与执行框架

> 实施顺序以 `docs/HGR_V7_STAGED_ROADMAP.md` 为准：先做不改变行为的 unified MetricAction
> shadow view，再单独开启 direct-first、gate 和 target commitment。

## 1. V7.1 只解决什么

V7.1 只解决 V6 的路径效率和历史目标执行问题：

1. V6 选择历史 Snapshot 后，优先机械回放 VISITED waypoint，没有优先使用当前时刻的
   Pathfinder 最短路；
2. route commitment 保存脆弱的 Snapshot 对象，mapping 消失后大量取消；
3. commitment 期间不重新比较当前目标与新候选，可能继续执行已经不划算的回访；
4. V6 的相关性排序没有结合绝对 metric cost，远距离历史证据可能带来 SR 小幅提高但 SPL
   明显下降。

V7.1 不实现以下内容：

- 不引入新的 confidence/belief 状态；
- 不融合正负证据；
- 不做 topology propagation；
- 不修改检测器、Snapshot 聚类、HGR prompt 或成功条件；
- 不使用 GT 距离或最终 success 参与在线决策。

## 2. 核心原则

V7.1 使用下面的执行优先级：

```text
当前强 DIRECT 证据（Object/Description/Image）
        ↓ 没有
原 HGR 可执行候选 + V6 goal relevance
        ↓ 历史复用满足保守门控
REVISIT capture anchor
        ↓
Pathfinder 当前直达优先
        ↓ 直达不可用
VISITED safe-topology fallback
        ↓
每执行一个短 segment 后重新观察、重新计算、决定保持或切换
```

Scene Topology 提供稳定身份、历史采集位置和安全恢复连接，但不再把 hop 数当作真实路径成本。

三类任务共享 `DIRECT / REVISIT / EXPLORE`、metric cost、intervention gate、direct shortcut 和
target commitment。唯一允许不同的是输入证据的编码方式：Object 使用类别/检测信息，Description
使用 text-image 相关性，Image 使用 image-image 相关性。证据进入动作层后都必须是 `[0,1]`
relevance，且 V7.1 使用同一组 gate 参数；不得复制三套导航状态机或按目标类型调阈值。

## 3. 数据结构

建议新增 `src/metric_goal_execution.py`，避免把 V7.1 的执行状态继续堆入 runner。

### 3.1 MetricAction

```python
@dataclass(frozen=True)
class MetricAction:
    action_type: DIRECT | REVISIT | EXPLORE
    topo_node_id: str
    source_kind: str
    source_index: int | None
    snapshot_image: str | None
    capture_anchor_id: str | None
    relevance: float
    direct_geodesic_m: float | None
    topology_route_m: float | None
    effective_cost_m: float
    route_mode: CURRENT | DIRECT_PATH | SAFE_TOPO | UNREACHABLE
```

`MetricAction` 是每一步生成的 goal-specific executable view，不写回持久 Scene Topology。

### 3.2 TargetCommitment

```python
@dataclass
class TargetCommitment:
    goal_key: str
    action_type: REVISIT
    topo_node_id: str
    capture_anchor_id: str
    snapshot_image: str | None
    created_step: int
    last_evaluated_step: int
    initial_cost_m: float
    last_cost_m: float
    best_cost_m: float
    relevance: float
    segments_executed: int
    no_progress_segments: int
```

Commitment 不保存 `Snapshot` Python 对象和 object mapping。Snapshot mapping 消失不再自动导致
路线中途取消；agent 可以继续到固定 capture anchor 重新观察，但旧 Snapshot 不再被当作终止成功。

## 4. Metric cost 定义

### 4.1 Capture anchor

OBSERVED 节点必须通过固定 `OBSERVED_AT` 边找到其原始 VISITED capture anchor。禁止使用后续
扫描位置重新锚定历史 Snapshot。

### 4.2 三类动作

```text
DIRECT
  当前观测中的目标；cost = 0 或当前位置到当前 executable point 的 geodesic。

EXPLORE
  cost = Pathfinder(current, frontier.position)。
  当前查询不可达时，本步标记 UNREACHABLE，但不删除持久节点。

REVISIT
  direct = Pathfinder(current, capture_anchor)
  direct 可达：effective_cost = direct，route_mode = DIRECT_PATH
  direct 不可达且 safe topology 可用：effective_cost = topology route length
  两者均不可用：effective_cost = inf，route_mode = UNREACHABLE
```

所有距离统一为米。用于决策的 Pathfinder 查询不允许随机邻近点恢复；邻近点恢复只能在真正执行
导航时由原 planner 使用。

### 4.3 剩余预算

GOAT 当前每个 agent step 最多前进约 1 米，因此：

```text
remaining_budget_m = remaining_steps * planner_step_m
budget_ratio = effective_cost_m / max(remaining_budget_m, eps)
```

该预算只用于拒绝明显不可能完成的远距离回访，不改变 benchmark 的最大步数。

## 5. Conservative intervention gate

V7.1 不直接替代 HGR。门控失败时，必须逐字保留原 HGR 候选顺序、source mapping、prompt 和
导航调用。

### 5.1 无条件规则

1. 当前帧存在通过原 HGR 判定的强目标证据时使用 DIRECT；
2. Object 的精确类别检测是 DIRECT 的确定性证据，Description/Image 则沿用各自原 HGR
   当前观测匹配，不另建导航分支；
3. 没有 finite REVISIT 时走原 HGR；
4. REVISIT 超过剩余预算时走原 HGR；
5. source mapping 无法可逆解析时走原 HGR。

### 5.2 历史复用门控

令：

```text
R = relevance 最高的 reachable REVISIT
E = relevance 最高的 reachable 非历史动作
C_E = 当前 reachable EXPLORE 中最小 metric cost
```

只有全部满足时才允许 Goal Topology 介入：

```text
semantic_clear:
    relevance(R) >= relevance(E) + min_relevance_margin

budget_ok:
    cost(R) <= max_budget_fraction * remaining_budget_m

opportunity_cost_ok:
    cost(R) <= max_revisit_to_explore_ratio * max(C_E, planner_step_m)
```

若当前没有 reachable EXPLORE，只使用 `semantic_clear + budget_ok`，禁止因除零放宽门控。

门控只决定是否允许历史 REVISIT 进入 V6 task view；最终 Snapshot/Frontier 选择仍复用 HGR/V6
现有选择机制，不新增 VLM call。

### 5.3 需要冻结的参数

V7.1 最多允许三个需要数据确定的参数：

```yaml
min_relevance_margin: null
max_budget_fraction: null
max_revisit_to_explore_ratio: null
```

参数选择流程：

1. 用 V7.0 split 1 查看已选 REVISIT 的 detour ratio，并由 V7.1-A 的 shadow metric view
   记录全部候选的 relevance margin 与 cost ratio；
2. 只比较预先声明的小网格，不做 category-specific 或 scene-specific 参数；
3. 在 split 2 只做一次确认，不根据单个 episode 继续调参；
4. 进入不重叠 train-validation 前冻结。

在 V7.0 尚未完成前，不在代码中写入拍脑袋的最终阈值。

## 6. Direct shortcut 与 safe-topology fallback

选择 REVISIT 后：

1. 查询当前位置到固定 capture anchor 的直接 Pathfinder 路径；
2. 路径存在时始终执行 direct shortcut，因为它是当前 navmesh 上的最短可达路径；
3. 直达不可用时，才调用 VISITED-only safe topology；
4. safe topology 只能产生 VISITED waypoint，OBSERVED/FRONTIER/HYPOTHESIS 不能作为中间航点；
5. 两种方式都只执行一个最多 1 米的 segment，然后返回主循环重新观察。

到达 capture anchor 不等于 subtask 成功：

- 若当前观测重新形成有效目标 Snapshot，则交回原 HGR Snapshot 终止逻辑；
- 若没有重新观测到目标，则完成本次 REVISIT，释放 commitment，下一步重新选择；
- 禁止仅凭历史节点或历史类别直接判定成功。

## 7. Target commitment 状态机

### 7.1 创建

REVISIT 通过门控并被实际选择时，创建对 `topo_node_id + capture_anchor_id` 的 commitment。

### 7.2 每段后重评估

每走一个 segment：

1. 更新 RGB-D、Scene、Snapshot、Frontier 和 persistent topology；
2. 重新计算 committed target 的 relevance、direct cost 和 fallback cost；
3. 构建当前最佳 challenger；
4. 根据以下规则 KEEP、SWITCH、COMPLETE 或 CANCEL。

### 7.3 保持与切换

采用可解释的字典序规则，不把不同量纲随意加权成一个 utility：

```text
KEEP：
  committed target 仍可达，且 challenger 的 relevance 优势不足 hysteresis；

SWITCH：
  challenger relevance >= committed relevance + switch_relevance_margin，
  且 challenger 通过相同 budget/opportunity-cost gate；

COST_SWITCH：
  两者 relevance 差位于 tie_margin 内，
  且 challenger 至少节省 min_switch_saving_m，
  并至少节省 min_switch_saving_ratio；
```

当前精确 Object 检测不受 hysteresis 限制，始终立即抢占。

### 7.4 完成与取消

```text
COMPLETE:
  capture_anchor_arrived

CANCEL:
  goal_switch
  subtask_end
  target_and_safe_route_unreachable
  repeated_no_progress
  source_identity_unresolvable

PREEMPT:
  current_exact_object_detected

SWITCH:
  semantic_challenger
  cheaper_tied_challenger
```

`snapshot_mapping_disappeared` 不再单独作为立即取消原因；它只让旧 Snapshot 失去终止资格。

### 7.5 无进展

默认基于 metric distance，而不是欧氏距离：

```text
progress_m = previous_cost_m - current_cost_m
no_progress = progress_m < min_progress_m
```

`min_progress_m` 和允许连续次数由 planner 的 1 米 segment 量纲确定，并作为安全参数固定，不根据
目标类别调整。

## 8. 三类统一与 Image 安全消融

V7.1 主方法对 Object、Description、Image 使用完全相同的动作层和执行状态机。模态差异只存在于
`GoalContext` 的 evidence adapter：

```text
Object       category/detector/semantic -> relevance
Description  text-image embedding       -> relevance
Image        image-image embedding       -> relevance
```

Image 强制回退原 HGR 只作为诊断消融，用于判断收益或退化来自 evidence quality，还是统一的
metric execution；它不是 V7.1 主方法的默认分支。

实验按顺序运行：

| 配置 | 变化 | 目的 |
|---|---|---|
| V7.1-A | 三类统一 MetricAction shadow view | 验证无损统一映射，不改变行为 |
| V7.1-B | A + direct shortcut | 验证统一成本执行 |
| V7.1-C | B + 三类共用 intervention gate | 验证保守复用 |
| V7.1-D | C + 三类共用 target commitment | V7.1 主方法 |
| V7.1-I | D，但强制 Image original HGR | 仅用于定位 Image 证据问题 |

只有前一项通过验收才进入下一项。最终论文至少保留 V6、B、C、D 和 Image 安全消融 I；A
主要用于工程兼容性验证。

## 9. 配置草案

```yaml
active_topology:
  mode: goal_topo_map_v7_1

  metric_execution:
    enabled: true
    pathfinder_direct_first: true
    topology_fallback: true
    reject_unreachable: true
    segment_horizon_steps: 1

  intervention:
    enabled: true
    goal_types: [object, description, image]
    modality_specific_thresholds: false
    min_relevance_margin: null
    max_budget_fraction: null
    max_revisit_to_explore_ratio: null

  target_commitment:
    enabled: true
    switch_relevance_margin: ${active_topology.intervention.min_relevance_margin}
    tie_margin: ${active_topology.intervention.min_relevance_margin}
    min_switch_saving_m: 1.0
    min_switch_saving_ratio: 0.25
    min_progress_m: 0.2
    max_no_progress_segments: 2
    preempt_on_live_object: true
```

配置继承 V7.0/V6，禁止复制并悄悄修改 detector、VLM、planner 或 benchmark 参数。

## 10. Trace 规范

每次候选构建记录：

```text
metric_action_view
  action/topo/source/relevance
  direct_geodesic_m/topology_route_m/effective_cost_m/route_mode
  remaining_budget_m/budget_ratio
```

每次门控记录：

```text
topology_intervention_gate
  allowed
  reason
  best_revisit
  best_alternative
  relevance_margin
  revisit_to_explore_cost_ratio
```

每次 commitment 记录：

```text
target_commitment_created
target_commitment_kept
target_commitment_switched
target_commitment_completed
target_commitment_cancelled
target_commitment_preempted
```

所有事件必须包含 scene、episode、subtask、step、target topo ID、cost before/after 和明确 reason。

## 11. 单元测试

至少覆盖：

1. 直达存在时不生成历史 waypoint；
2. 直达不可达时只使用 VISITED safe route；
3. 两种路径均不可达时回退原 HGR；
4. Object、Description、Image 都生成同构的 MetricAction；
5. 三类 gate 失败时，各自的原 HGR 输入逐项一致；
6. mapping 消失后可到 anchor 重观察，但不能伪造成功；
7. commitment 每段重算 cost；
8. hysteresis 能防止小幅候选抖动；
9. live Object 能无条件抢占；
10. no-progress 达到次数后取消；
11. goal/subtask/episode 切换后 commitment 清空；
12. 不新增 VLM logical calls；
13. 全部指标和 trace JSON 均有限。

## 12. 运行与验收顺序

```text
Step 1  完成 V7.0 split 1，分析已选 REVISIT 的 route detour
Step 2  实现 V7.1-A 全候选 shadow metric view，验证行为兼容
Step 3  V7.1-B direct-first 已实现；先跑 1×1 smoke，再在固定 dev 上比较 V6 vs B
Step 4  据 shadow trace 冻结 gate 小网格，实现并比较 V7.1-C
Step 5  实现 V7.1-D target commitment
Step 6  以 D 为 V7.1 主方法，运行 Image original-HGR 安全消融定位证据问题
Step 7  生成并封存不重叠 train-validation manifest，但暂不查看结果
Step 8  V7.1 在 train-dev 达标后才进入 V7.2 confidence
```

Train-dev 验收条件：

- Distance SR 相对 V6 不下降超过 1 pp；
- Distance SPL 相对 V6提高至少 3 pp；
- 平均总帧数相对 V6下降至少 10%；
- 任一目标类型 Distance SR 下降不超过 2 pp；
- harmful cancellation rate 相对 V6 至少下降 30%；
- VLM logical calls 不增加；
- 所有 Topo 介入和 target 切换均可由 trace 重建。

其中 harmful cancellation 不包含 `current_exact_object_detected` 这类正确抢占，只包含不可达、
无进展、身份解析失败和执行失败。

若只改善整体均值但在不重叠 train-validation 上不成立，V7.1 判定失败，不继续增加 confidence
或 propagation 来掩盖执行层问题。
