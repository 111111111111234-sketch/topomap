# HGR-ActiveTopo：在 HGR 中引入任务条件化持久拓扑

> 文档状态：**复杂 BeliefTopo v1/v2/v3 与 SimpleMemory 已保留；当前方法为 Persistent Goal-Conditioned Topo Map**  
> 目标代码库：`/home/hdd/tangyuxin/projects/Hypothesis_Graph_Refinement`  
> 思想参考：`/home/hdd/tangyuxin/projects/topo/conftopo`  
> 第一版评测入口：GOAT-Bench  
> 暂定方法名：**HGR-ActiveTopo**

## 1. 目标与设计结论

HGR-ActiveTopo 不在 HGR 旁边再运行一套 ConfTopo，而是把两个原则移植到 HGR 原生导航流程中：

1. 用一张持久图保存跨步骤、跨子任务的环境事实；
2. 根据当前 GOAT 子目标，从持久图生成临时的任务条件化候选视图。

第一版保留 HGR 的 TSDF、Habitat Pathfinder、ConceptGraph、HypothesisGraph、SemanticCritic 和导航执行，也保留冻结 Qwen3-VL/DashScope baseline 的 `choose_every_step: true` 调用策略。

第一版只新增以下闭环：

- 跨步骤稳定的 frontier 身份；
- 持久的 `VISITED / OBSERVED / FRONTIER / HYPOTHESIS` 节点；
- 相互独立的任务相关性 R、地图置信度 C、可达性 A；
- `ACTIVE / VERIFY / FOLDED / SUPPRESSED` 生命周期；
- 负面证据的软置信度传播；
- 不可达目标抑制和分级恢复。

第一版只接入 GOAT-Bench，不修改 A-EQA，不训练新模型，保持 train-free。

## 2. 问题定义与现有代码证据

以下判断针对当前仓库实现，不代表对 HGR 论文方法的一般性判断。

### 2.1 普通观测未形成持久语义节点

`src/hypothesis_graph.py` 已提供 `create_observed_node_from_snapshot()`，但 GOAT 主流程当前只调用 `TSDFPlanner.update_observation_history()`。普通 snapshot 因此只进入长度受限的历史列表，没有稳定成为 HypothesisGraph 中的 `OBSERVED` 节点。

第一版补齐以下证据链：

```text
snapshot / detected objects
    -> persistent OBSERVED node
    -> anchor to VISITED node
    -> provide evidence for FRONTIER/HYPOTHESIS nodes
```

### 2.2 frontier 身份只在当前提取结果内有效

`TSDFPlanner._predict_hypothesis_node_semantics()` 根据当前 `frontier_id` 维护假设。frontier 重新聚类、局部移动或暂时消失时，即使仍是同一空间边界，也可能丢失既有语义和导航历史。

第一版在 TSDF frontier 之上增加稳定 `topo_id`。TSDF 仍每步重新提取 frontier，持久拓扑负责把新结果与历史节点匹配。

### 2.3 frontier 排序没有显式利用长期 C/A

当前 `rank_frontiers_by_exploration_score()` 主要组合目标对齐、欧氏距离和语义熵，没有显式纳入历史正负证据、多视角一致性、Pathfinder 路径代价、导航失败、blocked 状态和跨子任务复用价值。

HGR-ActiveTopo 不替换原始感知和 frontier 生成，只在候选层加入 R/C/A 投影和排序。

### 2.4 SemanticCritic 当前直接执行硬删除

`SemanticCritic.verify_hypothesis_node_arrival()` 当前既判断证据，又直接调用 `HypothesisGraph.cascade_delete()`。一次遮挡、误检测或不稳定视角可能删除整条后代链。

第一版拆开职责：SemanticCritic 只返回结构化验证证据；PersistentBeliefTopology 统一更新直接节点及依赖后代的 C；只有持续强反证才物理裁剪。保留 hard correction 开关用于复现原始 HGR 和消融。

### 2.5 与当前 A-EQA 41 题结果的关系

已有 A-EQA 41 题结果用于证明环境、模型和主流程可执行，但不是第一版 ActiveTopo 的主要验证集：

- API 连接失败、Requesty 402 余额不足属于外部服务问题，拓扑修改不能解决；
- 无 navigable point 或无合适 snapshot observation point 受场景几何和候选生成约束，拓扑不能保证将其变为可达；
- 达到最大步数、反复选择低收益或不可达候选，是 ActiveTopo 可能改善的问题。

不能用 A-EQA API 成功率作为 ActiveTopo 有效性的证据。第一版在具有连续多子任务的 GOAT-Bench 中检验记忆复用和目标切换。

## 3. 总体架构与不变量

```text
RGB-D + pose
  +-> ConceptGraph snapshots/objects --------+
  +-> TSDF frontier extraction               |
         +---------------------> PersistentBeliefTopology
                                      | C: map confidence
                                      | A: accessibility
                                      | stable identity/evidence
                                      v
current GOAT subgoal ----------> ActiveTopoProjector
                                      | R: goal relevance
                                      | lifecycle state
                                      v
                             ActiveTopoView candidates
                                      |
                         utility + VLM selection/index mapping
                                      |
                           original TSDF Frontier target
                                      |
                         TSDFPlanner / Pathfinder execution
                                      |
                    arrival / failure / verification evidence
                                      |
                              update C/A and topology
```

必须维持以下不变量：

1. `PersistentBeliefTopology` 保存环境事实、C、A 和证据历史；
2. `ActiveTopoView` 只保存当前目标的 R、状态和排序结果；
3. GOAT 子目标切换只重建 View，不清空持久图；
4. scene/episode 结束后清空持久图，第一版不跨 episode 保存；
5. R 不写回 C，避免任务偏好污染长期事实；
6. 新模块关闭时必须走原始 HGR 路径；
7. 排序或过滤后的候选必须保留到原始 `Frontier` 的显式映射。

## 4. 数据结构与公共接口

建议新增 `src/persistent_belief_topology.py`，集中放置持久图、投影视图和更新策略。第一版不引入外部图数据库。

### 4.1 枚举

```python
class TopoNodeType(str, Enum):
    VISITED = "visited"
    OBSERVED = "observed"
    FRONTIER = "frontier"
    HYPOTHESIS = "hypothesis"

class TopoEdgeType(str, Enum):
    NAVIGABLE = "navigable"
    OBSERVED_AT = "observed_at"
    ANCHORED_TO = "anchored_to"
    DEPENDS_ON = "depends_on"
    SPATIAL_NEAR = "spatial_near"

class ActiveTopoState(str, Enum):
    ACTIVE = "active"
    VERIFY = "verify"
    FOLDED = "folded"
    SUPPRESSED = "suppressed"
```

### 4.2 持久节点和边

```python
@dataclass
class BeliefTopoNode:
    node_id: str
    node_type: TopoNodeType
    position: np.ndarray
    map_log_odds: float = 0.0
    semantic_dist: Optional[SemanticDistribution] = None
    observed_categories: set[str] = field(default_factory=set)
    evidence_refs: list[str] = field(default_factory=list)
    view_count: int = 0
    positive_evidence: float = 0.0
    negative_evidence: float = 0.0
    last_observed_step: int = -1
    visit_count: int = 0
    nav_success_count: int = 0
    nav_failure_count: int = 0
    consecutive_failures: int = 0
    blocked_until_step: int = -1
    consumed: bool = False
    hypothesis_node_id: Optional[str] = None

    @property
    def map_confidence(self) -> float:
        return sigmoid(self.map_log_odds)

    @property
    def accessibility(self) -> float:
        return (self.nav_success_count + 1) / (
            self.nav_success_count + self.nav_failure_count + 2
        )

@dataclass
class TopoEdge:
    source_id: str
    target_id: str
    edge_type: TopoEdgeType
    distance: Optional[float] = None
    path_cost: Optional[float] = None
    traversability: Optional[float] = None
    dependency_confidence: float = 1.0
    visit_count: int = 0
    failure_count: int = 0
    last_updated_step: int = -1
```

`map_log_odds` 限制在 `[-6, 6]`，日志同时记录 log-odds 和概率 C。

### 4.3 当前目标与投影候选

```python
@dataclass(frozen=True)
class GoalContext:
    subtask_id: str
    goal_type: str
    category: Optional[str] = None
    description: Optional[str] = None
    image_embedding: Optional[np.ndarray] = None

@dataclass
class ActiveTopoCandidate:
    topo_node_id: str
    source_frontier_index: int
    relevance: float
    confidence: float
    accessibility: float
    path_cost: float
    revisit_penalty: float
    information_gain: float
    utility: float
    state: ActiveTopoState

@dataclass
class ActiveTopoView:
    goal: GoalContext
    recovery_level: int
    candidates: list[ActiveTopoCandidate]
    created_at_step: int
```

`source_frontier_index` 是强制字段。VLM 返回投影列表中的候选编号，执行前必须先解析为 `ActiveTopoCandidate`，再通过它取回 `tsdf_planner.frontiers` 中的原始对象。

### 4.4 PersistentBeliefTopology 接口

```python
class PersistentBeliefTopology:
    def upsert_visited(self, position, step, evidence_ref=None) -> str: ...
    def upsert_observed(self, snapshot, position, categories, step,
                        anchor_visited_id) -> str: ...
    def match_frontiers(self, frontiers, step) -> dict[int, str]: ...
    def bind_hypothesis(self, topo_frontier_id, hypothesis_node_id,
                        dependency_confidence) -> None: ...
    def record_navigation_result(self, node_id, success, step,
                                 reason=None) -> None: ...
    def apply_verification(self, node_id, evidence, step,
                           correction_mode="soft") -> list[str]: ...
    def build_active_view(self, goal, frontiers, path_costs, step,
                          recovery_level=0) -> ActiveTopoView: ...
    def get_statistics(self) -> dict: ...
```

所有更新操作必须安全处理不存在或已裁剪的 ID，并把跳过原因写入 trace，不能因单个 stale node 中断 episode。

## 5. 核心算法

### 5.1 OBSERVED 节点合并

新 snapshot 与已有 `OBSERVED` 节点同时满足以下条件时合并：

- 三维位置距离 `<= 0.75m`；
- 检测对象类别集合的 Jaccard 相似度 `>= 0.30`。

若 snapshot 没有有效对象类别，则不使用类别条件，只允许在距离 `<= 0.35m` 时合并，否则创建新节点。合并后更新位置滑动均值、类别并集、view count、last observed step 和 evidence references，并刷新到最近 `VISITED` 节点的 `OBSERVED_AT` 边。

### 5.2 稳定 frontier 匹配

每步提取的新 frontier 与未过期历史 `FRONTIER` 节点计算：

```text
S_dist   = exp(-distance / 1.0m)
S_region = region IoU
S_dir    = normalized direction similarity
S_visual = CLIP cosine similarity mapped to [0, 1]

S_match = 0.45*S_dist + 0.25*S_region + 0.15*S_dir + 0.15*S_visual
```

规则：

- 缺少特征时按剩余权重重新归一化；
- 分数必须 `>= 0.60`；
- 按分数降序执行 greedy one-to-one 匹配；
- 匹配成功沿用历史 `topo_id` 并更新特征；
- 未匹配的新 frontier 创建新 ID；
- 当前帧消失的 frontier 只标记 inactive，不立即删除；
- 第一版不做 episode 内 TTL 物理删除，episode 结束统一释放。

### 5.3 地图置信度 C

初始 `map_log_odds = 0`，对应 `C = 0.5`。每个有效步骤先对非本步观测节点衰减：

```text
log_odds <- 0.995 * log_odds
```

直接正、负证据分别更新：

```text
positive: log_odds <- clamp(log_odds + 1.0 * strength, -6, 6)
negative: log_odds <- clamp(log_odds - 1.2 * strength, -6, 6)
```

`strength` 限制在 `[0, 1]`，同一个 `evidence_ref` 不得重复更新。

校准实现中，新建 frontier 的首次出现只建立未知置信度节点，保持 `C=0.5`；只有它在
后续步骤与同一 stable ID 匹配成功时，才以匹配分数作为 positive strength 更新 C。
这样高 R 的首次观测可以进入 `VERIFY`，稳定复现后才转为高 C 的 `ACTIVE`。

v4 为防止连续 stable match 使 C 迅速饱和，frontier match evidence 使用固定缩放
`0.25 * match_score`。这不改变 `positive_delta=1.0`，只是区分一次普通重复观测与
SemanticCritic 等强直接证据。

### 5.4 SemanticCritic 与 soft correction

SemanticCritic 改成无图结构副作用的证据生产者：

```python
@dataclass
class VerificationEvidence:
    node_id: str
    is_positive: bool
    strength: float
    residual: Optional[float]
    reason: str
    evidence_ref: str
```

`correction_mode="hard"` 时复现当前 cascade delete。`correction_mode="soft"` 时先更新直接节点，再对依赖后代计算：

```text
descendant_C <- descendant_C *
    (1 - 0.7 * dependency_confidence * negative_strength)
```

结果重新转换为 log-odds 并限制在 `[-6, 6]`。只有同时满足以下条件才物理裁剪：

- `C < 0.10`；
- 累计至少 2 条不同负面 evidence；
- 最近 5 步没有正面 evidence。

裁剪由统一更新层同时作用于 PersistentBeliefTopology 和 HypothesisGraph，禁止 SemanticCritic 与主循环重复删除。

### 5.5 可达性 A

```text
A = (nav_success_count + 1) /
    (nav_success_count + nav_failure_count + 2)
```

- 成功到达：success `+1`，连续失败清零，立即解除 blocked；
- 同一 stable frontier 的 geodesic distance 相比上次至少下降 `0.1m`：记录
  `0.25` 个 accessibility success evidence，但不标记 consumed；
- 无路径、执行失败或目标点无法落在 navigable area：failure `+1`；
- 单次失败后阻塞 5 步；
- 连续失败 3 次后进入 `SUPPRESSED`；
- 新的明确成功路径证据可以解除 `SUPPRESSED`；
- 仅因当前不相关而 `FOLDED` 不得降低 A。

### 5.6 任务相关性 R

R 只存在于当前 `ActiveTopoView`，范围 `[0, 1]`：

- category goal：节点类别分布与目标类别的概率或语义相似度；
- description goal：目标描述 CLIP text embedding 与节点视觉/语义 embedding 的相似度；
- image goal：目标图像 CLIP embedding 与节点视觉 embedding 的相似度。

无法取得对应 embedding 时使用现有 HGR semantic distribution；两者都不存在时设为 `0.5` 表示未知，而不是直接判为无关。

CLIP 项使用固定区间校准：`R_clip=clip((cosine-0.10)/(0.35-0.10), 0, 1)`；
frontier identity 匹配仍使用 `(cosine+1)/2`。两种量纲必须分离，否则中性 CLIP
相似度会被错误解释为 `R≈0.5`。区间校准也避免直接使用 raw cosine 时把约 75% 的
真实候选折叠。缺失特征的显式 fallback 仍为 `0.5`，不会被当作低相关证据。

### 5.7 生命周期状态

按以下顺序判定：

1. 已 consumed、当前仍 blocked 或连续失败达到 3 次：`SUPPRESSED`；
2. `R >= 0.55` 且 `C < 0.55`：`VERIFY`；
3. `R < 0.20`：`FOLDED`；
4. 其他：`ACTIVE`。

状态标签不随 recovery level 改名。`SUPPRESSED` 不进入正常和 recovery 候选；
`FOLDED` 在 level 0 隐藏，在 level 1/2 按 recovery 规则重新放行，但不改名为
`ACTIVE`，也不从持久图删除。

### 5.8 候选效用

所有项归一化到 `[0, 1]`：

```text
U = 1.0*R + 0.4*C + 0.4*A
  - 0.1*path_cost - 0.2*revisit_penalty
  + 0.05*information_gain
```

- `path_cost` 优先使用现有 `TSDFPlannerBase.get_distance()` 的 Pathfinder 结果，在候选内做 min-max 归一化；全部相同时置 0；
- Pathfinder 明确不可达的节点直接 suppressed；仅查询异常时允许现有欧氏回退并写入 trace；
- `revisit_penalty = min(visit_count / 3, 1)`；
- `information_gain` 使用现有 frontier area/entropy，缺失时置 0；
- VLM 输入按 U 降序排列，同时显示 R/C/A/U/state。

### 5.9 Recovery ladder

```text
Level 0: ACTIVE + VERIFY
Level 1: Level 0 + C >= 0.55 且 A >= 0.50 的 FOLDED
Level 2: 所有非 SUPPRESSED 且 Pathfinder 可达的节点
```

候选为空、连续 12 步无路径改善且无新增相关观测、或首选目标重复导航失败时提升一级。获得新相关观测、成功到达新节点或切换子目标时恢复 Level 0。Level 2 仍为空则走现有 HGR fallback，并记录 `recovery_exhausted`。

## 6. GOAT-Bench 集成流程

### 6.1 生命周期

```text
GOAT episode start
    -> create one PersistentBeliefTopology
    -> for each subtask:
         create GoalContext
         build/rebuild ActiveTopoView
         perception -> projection -> selection -> navigation -> update
    -> aggregate topology metrics
    -> release topology at episode end
```

一个 GOAT episode 内的多个子任务共享同一张持久图，不同 episode 和 scene 不共享。

### 6.2 每步数据流

1. 当前 pose 写入或合并 `VISITED`；
2. 新 snapshot 写入或合并 `OBSERVED`；
3. 当前 TSDF frontiers 与历史 `FRONTIER` 稳定匹配；
4. HGR hypothesis 与对应 topo frontier 建立绑定；
5. 计算 Pathfinder 路径代价；
6. 根据 GoalContext 构建 ActiveTopoView；
7. 把投影候选而不是完整 frontier 列表交给 VLM；
8. 将 VLM 返回编号映射回 `source_frontier_index`；
9. 继续调用原始 TSDF/Pathfinder 导航；
10. 用到达、碰撞、无路径和 semantic verification 更新 C/A；
11. 写入逐步 trace 和 episode 汇总。

第一版不对 snapshot 候选执行 FOLDED。已有 snapshot 仍按原 HGR 方式参与目标选择；ActiveTopo 首先只控制 frontier 候选。

### 6.3 VLM 调用策略

第一版保留：

```yaml
choose_every_step: true
```

只改变 VLM 看到的候选及其拓扑属性，不改变调用频率、Qwen3-VL 模型、DashScope 路由、prompt 基本任务和重试策略，以便把效果变化主要归因于 ActiveTopo。

## 7. 配置、输出与兼容性

### 7.1 独立配置

实现使用独立配置 `cfg/eval_goatbench_activetopo_qwen3vl_dashscope.yaml`，不覆盖原始配置或冻结 Qwen baseline：

```yaml
active_topology:
  enabled: true
  correction_mode: soft
  observed_merge:
    max_distance: 0.75
    min_category_jaccard: 0.30
    empty_category_max_distance: 0.35
  frontier_matching:
    min_score: 0.60
    distance_scale: 1.0
    weights: {distance: 0.45, region_iou: 0.25, direction: 0.15, visual: 0.15}
  confidence:
    decay: 0.995
    positive_delta: 1.0
    negative_delta: 1.2
    prune_threshold: 0.10
    min_negative_evidence: 2
    positive_grace_steps: 5
    dependency_propagation: 0.70
  accessibility:
    block_steps: 5
    suppress_after_failures: 3
  active_view:
    verify_relevance: 0.55
    verify_confidence: 0.55
    fold_relevance: 0.20
    no_progress_steps: 12
  utility:
    relevance: 1.0
    confidence: 0.4
    accessibility: 0.4
    path_cost: 0.1
    revisit: 0.2
    information_gain: 0.05
```

首版配置不得出现 `vlm_scheduler`。原始配置保持不变，作为冻结 baseline。

Qwen/DashScope 的 calibrated dev-10 使用 `vlm_max_images: 40`。该预算不拼图，图片
仍按原始逐图格式发送；超出预算时固定保留目标图、当前 egocentric view、排序后的
frontier，再按原顺序放入完整 snapshot（full image + 至少一个 object crop）。裁剪后
必须同步重建 snapshot/object 映射。公平对照 baseline 使用同样的 40 图预算。

### 7.2 兼容开关

`active_topology.enabled: false` 时不创建拓扑、不改候选、不改 SemanticCritic 行为，输出与原始 HGR GOAT 流程兼容。

### 7.3 输出和 trace

ActiveTopo 默认输出到 `results/exp_eval_goatbench_activetopo_qwen3vl30b_dashscope/`。每个 episode 增加 JSONL trace，每行至少包含：

```text
scene_id, episode_id, subtask_id, step, goal_type, recovery_level
candidate topo_id/source_frontier_index/state/R/C/A/U/path_cost
selected candidate and mapped original frontier
navigation result/failure reason, verification evidence
created/merged/pruned/suppressed node IDs
VLM call count and elapsed time
```

episode 汇总包含 success、SPL、path length、子任务步数、VLM 次数、运行时间、节点创建/合并/复用/裁剪数、stable-ID match rate、blocked target retry count、各级 recovery 次数及 soft/hard correction 次数。

## 8. 实施阶段

源码按以下顺序实施；第一版代码已完成这些阶段，正式完整实验尚未执行：

1. **基础结构与稳定身份**：数据结构、VISITED/OBSERVED 合并、frontier stable-ID 和 trace；不改变目标选择。
2. **证据、C/A 与 soft correction**：拆分 SemanticCritic 副作用，实现 C/A、blocked/suppressed，并验证 hard 模式兼容。
3. **目标投影和候选映射**：实现三类 GOAT goal 的 R、状态、效用、recovery 和原始 Frontier 映射。
4. **GOAT 主流程接入**：接入 episode/step 生命周期、新配置和结果目录，保持 `choose_every_step`。
5. **基线、完整实验与消融**：相同环境下对比并根据 trace 分析收益和失败模式。

## 9. 测试与验收

### 9.1 单元测试

1. Observed 满足空间/类别阈值时合并，越界时新建，空类别走 0.35m 规则；
2. frontier 小幅移动或暂时缺失保持 ID，两个新 frontier 不匹配同一历史节点；
3. 缺少视觉或 region 特征时匹配权重正确重归一化；
4. goal switch 只改变 R/state/view，不改变 C/A 和 node ID；
5. C 的正负更新、衰减、证据去重和裁剪正确；
6. soft propagation 按依赖置信度衰减，未满足条件时不裁剪；
7. hard 模式产生与当前 cascade delete 相同的节点集合；
8. 失败阻塞 5 步、连续 3 次 suppressed、成功后解除；
9. 候选为空、无进展和重复失败能正确升级 recovery；
10. 过滤和重排后 VLM 编号仍映射到正确原始 Frontier；
11. 功能关闭后不创建拓扑且不改变原始流程。

### 9.2 集成测试

使用含至少两个连续子任务的单个 GOAT episode 和固定/缓存 VLM 响应，验证跨子任务复用、episode 间清空、候选映射合法以及 trace 可重放。

### 9.3 正式实验口径

先运行 GOAT `split 1` 的 36 个 episodes，再决定是否扩展全部 split。Baseline 与 ActiveTopo 使用相同 episode 顺序、随机种子、GPU、Habitat 设置、Qwen3-VL/DashScope 配置、`choose_every_step: true`、成功阈值和步数上限。

主指标：Success Rate、SPL、平均路径长度、平均每子任务步数、VLM 调用次数和总运行时间。

诊断指标：stable-ID match rate、跨子任务节点复用率、blocked target 重试率、suppressed 数量、recovery 成功率、soft correction 后恢复节点数和 hard prune 数量。

### 9.4 消融矩阵

| 实验 | Stable topology | Soft revision | Active view | R/C/A utility |
|---|---:|---:|---:|---:|
| HGR baseline | 否 | 否 | 否 | 否 |
| + StableTopo | 是 | 否 | 否 | 否 |
| + SoftRevision | 是 | 是 | 否 | 否 |
| + ActiveView | 是 | 是 | 是 | 否 |
| HGR-ActiveTopo | 是 | 是 | 是 | 是 |

验收要求：

1. 功能关闭时 baseline 行为不变；
2. 无候选越界、重复删除或 stale-ID 崩溃；
3. ActiveTopo 至少降低 blocked target 重试或平均无进展步数；
4. 失败 episode 必须按统一口径计入 SR/SPL；
5. 性能结论同时报告主指标、API 成本和运行成本。

## 10. 风险与控制

- **稳定 ID 错误合并**：使用 one-to-one 匹配、0.60 阈值、逐项 trace 和消融控制。
- **soft correction 保留错误节点过久**：采用 `C < 0.10 + 两条独立反证 + 5 步无正证据` 联合裁剪条件。
- **Pathfinder 比较不一致**：只使用 HGR 当前可调用的信息，不引入额外 ground-truth oracle。
- **图规模增长**：只保留四类节点，Observed 合并，episode 结束统一释放。
- **API 波动掩盖效果**：逻辑测试使用缓存/mock；正式实验记录失败、重试、次数和耗时；外部 API 余额或连接问题单独报告。

## 11. 第一版明确不做

- 不修改 A-EQA 主流程或 41 题结果；
- 不实现自适应 VLM query scheduler；
- 不减少或动态预算 VLM 调用；
- 不增加训练、微调或学习型置信度校准；
- 不加入 `ROOM / PORTAL / LANDMARK / GOAL_REGION`；
- 不复制 ConfTopo 完整 agent/control stack；
- 不替换 TSDF、ConceptGraph、HypothesisGraph 或 Pathfinder；
- 不跨 episode、scene 或进程持久化；
- 不把 snapshot 候选纳入 FOLDED；
- 不用 A-EQA execution yield 代替问答准确率或 GOAT 指标。

## 12. 后续扩展

核心闭环通过 GOAT 对照实验后，再单独考虑自适应 VLM scheduler、snapshot/object 统一投影、Room/Portal 层级拓扑、学习型 R/C/A、A-EQA 接入和跨 episode 持久化。这些扩展必须独立消融，不能与第一版同时引入。

## 13. 最终交付定义

HGR-ActiveTopo 第一版完成时，应能在不改变 HGR 感知模型和 VLM 调用频率的前提下：

- 在一个 GOAT episode 内维护稳定环境事实图；
- 切换子目标时复用历史 C/A，只重算 R 和 ActiveTopoView；
- 避免短期内重复选择已知不可达目标；
- 用软证据修正替代单次反证触发的立即级联删除；
- 在候选耗尽或长时间无进展时可解释地恢复搜索范围；
- 用统一 GOAT baseline、trace、主指标和消融验证有效性。

在上述条件达到前，不宣称已完成 HGR 与 ConfTopo 的有效融合。

## 14. 第一版实现记录

第一版实现已落在以下入口：

- `src/persistent_belief_topology.py`：持久节点/边、稳定 frontier 匹配、R/C/A、ActiveTopoView、recovery 和 trace；
- `run_goatbench_evaluation.py`：episode/subtask/step 生命周期、Pathfinder 代价、导航与验证回写；
- `src/query_vlm_goatbench.py`：投影候选 prompt 和 projected index 到原始 Frontier 的显式映射；
- `src/semantic_critic.py`：证据评估与图修改分离，默认 hard 行为保持兼容；
- `cfg/eval_goatbench_activetopo_qwen3vl_dashscope.yaml`：正式完整实验配置；
- `cfg/eval_goatbench_activetopo_qwen3vl_dashscope_dev10.yaml`：隔离的 dev-10 配置。

本地验收使用 `python -m unittest discover -s tests -v`，不调用外部 API。完整 GOAT 指标仍须以独立 ActiveTopo 结果目录的正式运行结果为准。

## 15. Dev-10 验证记录（2026-08-24）

### 15.1 对照结果

本轮使用与冻结 Qwen3-VL baseline 相同的 10 个 split-1 subtasks：

| 指标 | Qwen baseline | ActiveTopo | 差值 |
|---|---:|---:|---:|
| Snapshot SR | 20.00% | 30.00% | +10.00 pp |
| Distance SR | 70.00% | 60.00% | -10.00 pp |
| Snapshot SPL | 20.00% | 27.31% | +7.31 pp |
| Distance SPL | 51.43% | 57.31% | +5.88 pp |
| 运行时间 | 13:26 | 14:55 | +1:29（约 11.0%） |
| HTTP attempts | 99 | 114 | +15（约 15.2%） |

该样本量只用于开发诊断，不形成性能结论。一个 image 子任务发生 3 次逻辑
VLM 调用失败，对应 15 次 HTTP 400，因此本轮 SR/SPL 也不是干净的算法对照。

### 15.2 已验证的工程性质

- 10/10 subtasks 完成结果落盘，episode trace 可解析；
- weighted stable frontier-ID match rate 为 `80.56%`（87/108）；
- 发生 24 次 OBSERVED 合并；
- 过滤、效用排序和 projected index 到原始 Frontier 的映射可执行；
- 持久图跨同一 episode 的子任务保留，切换目标只重建 View；
- 第一轮离线 `unittest` 共 15 项通过，显式覆盖四种状态、三级 recovery 范围、
  自动恢复 folded 候选、C/A、soft correction、hard compatibility 和候选映射；
- `py_compile` 与 `git diff --check` 通过。

验证过程中修复了一处状态语义错误：旧实现进入 recovery level 1 后会把低 R 的
`FOLDED` 节点重新标成 `ACTIVE`。当前实现保持 `FOLDED` 标签不变，只改变候选放行
范围，并由单元测试锁定该行为。

### 15.3 尚未通过的策略有效性验收

真实 trace 共记录 43 个 step、105 个 frontier candidate：

- R 范围 `0.5574–0.9312`；
- C 范围 `0.7311–0.9975`；
- A 全部为 `0.5`；
- 105 个候选全部是 `ACTIVE`；
- verification、navigation success/failure、blocked retry、suppression 和 recovery
  均未在真实轨迹中触发。

因此本轮只证明 ActiveTopo 的数据通路与候选排序可运行，尚未证明 blocked-target
抑制、A 学习、soft revision 和 recovery ladder 对真实导航有效。冻结阈值
`fold_relevance=0.20`、`verify_relevance=0.55`、`verify_confidence=0.55` 与当前
CLIP/C 数值分布不匹配；不能直接据此运行完整 278-subtask 并宣称方法已验证。

### 15.4 v2 校准实现与下一轮验证门槛

已完成的 v2 修改：

- R 改为 CLIP 原始正 cosine，identity matching 的 cosine 量纲保持不变；
- 新 frontier 保持 `C=0.5`，跨步 stable match 后才增加 C；
- VLM explorer 增加 40 张独立图片的硬预算，不使用 contact sheet；
- 裁剪同步维护 Frontier 前缀索引及 snapshot/object 映射；
- 新增匹配的 capped baseline 配置；
- 离线验收增加到 18 项，全部通过，`py_compile`、配置解析和
  `git diff --check` 通过。

开始完整实验前必须先完成一轮新的 dev-10：

1. 使用本节的无成功标签 R/C 量纲校准，不再根据 dev-10 成功结果改阈值；
2. baseline 与 ActiveTopo 均使用 40 图预算，确认 explorer HTTP 400 为 0；
3. trace 中必须实际出现 `FOLDED`，并至少构造或观测一次 `SUPPRESSED/recovery`；
4. A 必须出现非 0.5 更新，否则单独报告终止反馈未被真实导航触发；
5. 再报告四项 SR/SPL、运行时间、VLM attempts、stable-ID rate、blocked retry 和
   recovery 次数。

只有以上策略路径得到真实或受控集成轨迹验证后，才进入完整 278-subtask 实验。

v2 对照配置：

- `cfg/eval_goatbench_qwen3vl_dashscope_dev10_capped.yaml`；
- `cfg/eval_goatbench_activetopo_qwen3vl_dashscope_dev10.yaml`。

### 15.5 Dev-10 v2 结果与 v3 修复

v2 的两组 10-subtask 实验均完成：

| 指标 | Capped baseline v2 | ActiveTopo v2 | 差值 |
|---|---:|---:|---:|
| Snapshot SR | 30.00% | 30.00% | 0.00 pp |
| Distance SR | 70.00% | 70.00% | 0.00 pp |
| Snapshot SPL | 26.84% | 26.15% | -0.69 pp |
| Distance SPL | 55.53% | 60.75% | +5.22 pp |
| 运行时间 | 17:38 | 16:36 | -1:02 |

R/C 校准在真实 trace 中生效：47 个 step、179 个 candidate 中出现 `ACTIVE=75`、
`FOLDED=90`、`VERIFY=14`；recovery level 1 出现 19 步，两个 episode 合计
stable-ID match rate 为 `162/(162+19)=89.50%`。这验证了 goal projection、folding、
verification state 和 level-1 recovery 的实际数据通路。

本轮仍未出现 frontier navigation terminal feedback，因此 A 全部为 `0.5`，
`SUPPRESSED`、soft correction 和 blocked retry 仍只能由离线测试证明，不能宣称真实
导航收益已经验证。

API 方面，baseline 的 HTTP 400 为 0；ActiveTopo 在一个 microwave 子任务中发生
15 次 HTTP 400 并将该子任务计为失败。失败 prompt 只有 11 张图，证明它不是 40 图
总预算溢出。结合该轨迹包含 detector object crops，v3 增加以下修复：

- 低于 Qwen3-VL 单图默认 `65536` 像素或极端宽高比的 crop，等比放入
  `256×256` 画布；不拼接不同候选；
- provider BadRequest 正文、purpose 和实际 image part 数写入文件日志；
- 同一 BadRequest 不再执行 5×3 次完全相同的重试；
- 离线测试增加到 19 项并全部通过。

v3 使用新的独立结果目录。只有 v3 两组 HTTP 400 都为 0，才把四项指标用于下一阶段
决策。

### 15.6 Dev-10 v3 干净对照

v3 两组均完成 10/10 subtasks，HTTP 400、invalid subtask、429 和其他 API error
全部为 0：

| 指标 | Capped baseline v3 | ActiveTopo v3 | 差值 |
|---|---:|---:|---:|
| Snapshot SR | 30.00% | 20.00% | -10.00 pp |
| Distance SR | 60.00% | 60.00% | 0.00 pp |
| Snapshot SPL | 28.89% | 20.00% | -8.89 pp |
| Distance SPL | 49.85% | 47.10% | -2.75 pp |
| 运行时间 | 13:33 | 18:04 | +4:31（约 33.3%） |
| HTTP attempts | 90 | 124 | +34（约 37.8%） |

两组 Distance success 的 10 项逐题结果完全一致。Snapshot 差异来自
`00832-qyAac8rV8Zk_0_1`：baseline snapshot success，ActiveTopo snapshot failure，
但两者 distance 均成功。10 项样本不足以把该单项差异解释为稳定退化。

ActiveTopo trace 包含 48 个 step、186 个 candidate：

- `ACTIVE=39`（20.97%）；
- `FOLDED=139`（74.73%）；
- `VERIFY=8`（4.30%）；
- recovery level 1 出现 21 步，两个 episode 合计升级 7 次；
- stable-ID match rate 为 `159/(159+27)=85.48%`；
- OBSERVED 合并 21 次。

因此 R/C 量纲校准、folding、VERIFY、候选映射和 level-1 recovery 已通过真实轨迹
验证。A 仍全部为 `0.5`，因为本轮 24 次 frontier selection 均未产生 terminal
frontier arrival 或明确导航失败；相应地 `SUPPRESSED`、blocked retry、soft correction
仍为 0。它们已有确定性单元测试，但尚无真实轨迹证据。

v3 的正确结论是：**工程与 API 验收通过，尚未观察到性能提升**。可以进入更大样本的
split-1 验证来降低 10 项随机波动，但在完整实验结果出来前不得宣称 ActiveTopo 优于
baseline；正式报告还必须包含约 33% 时间和约 38% API attempts 开销。

### 15.7 v4 建图证据优化

v4 只修改 ActiveTopo 的持久建图与证据解释，不改变 YOLO、SAM、OpenCLIP、
ConceptGraph、TSDF、Pathfinder、Qwen3-VL prompt 图片预算或 baseline：

1. CLIP R 从 raw positive cosine 改为固定 `[0.10, 0.35] -> [0,1]` 区间校准；
2. stable frontier match 对 C 的证据强度缩放为 `0.25 * match_score`，降低饱和；
3. 同一 stable frontier 路径下降至少 `0.1m` 时，给 A 增加 `0.25` 个弱成功证据；
4. partial progress 不设置 consumed，不等价于到达，也不掩盖明确导航失败；
5. 新增 A partial-progress 单元测试，本地验收为 20/20 通过。

v4 结果目录固定为：

```text
results/exp_eval_goatbench_activetopo_qwen3vl30b_dashscope_dev10_optimized_v4
```

由于 v4 只改变 `active_topology.enabled=true` 分支，共享的图像合法化、40 图预算和
baseline 路径均未变化，可以直接与干净的 capped baseline v3 比较，无需再次付费
运行 baseline。v4 验收重点是：A 不再全部为 0.5、C 不再大面积饱和、folded 比例
下降、HTTP 400 保持为 0，并同时检查 SR/SPL、step、时间和 API attempts 是否改善。

### 15.8 Dev-10 v4 结果与冻结决定

v4 完成 10/10 subtasks，HTTP 400、invalid subtask 和其他 API error 均为 0。与固定
capped baseline v3 对比如下：

| 指标 | Baseline v3 | ActiveTopo v4 | 差值 |
|---|---:|---:|---:|
| Snapshot SR | 30.00% | 30.00% | 0.00 pp |
| Distance SR | 60.00% | 70.00% | +10.00 pp |
| Snapshot SPL | 28.89% | 23.13% | -5.76 pp |
| Distance SPL | 49.85% | 49.86% | +0.01 pp |
| step | 36 | 72 | +100.0% |
| 运行时间 | 13:33 | 23:56 | +76.6% |
| HTTP attempts | 90 | 173 | +92.2% |

建图证据指标达到 v4 的预期目标：

- A 的范围从恒定 `0.5` 变为 `0.5–0.6923`，25 个候选观测的 A 非 0.5；
- 记录 17 次 geodesic progress evidence，且没有把 partial progress 误记为 terminal arrival；
- C 的中位数从 v3 的 `0.9914` 降至 `0.7751`，不再大面积饱和；
- `FOLDED` 从 v3 的 `139/186=74.73%` 降至 v4 的 `72/245=29.39%`；
- v4 状态为 `ACTIVE=160`、`VERIFY=13`、`FOLDED=72`；
- recovery level 1 只出现 1 步，stable-ID match rate 为
  `208/(208+37)=84.90%`。

Distance SR 的新增成功来自 `00800-TEEsavR23oF_0_3`，但该子任务运行 30 step，
Snapshot SPL 仅 `0.3129`。这解释了“SR 提升但总 step/API/时间显著增加”：v4 愿意沿着
获得正向可达性证据的长路径继续探索，最终找到目标，但效率较低。

同一 dev-10 已参与 R/C/A 的多轮开发。继续依据这 10 项调整 path penalty、A 权重或
停止条件会产生明显测试集过拟合。因此 v4 暂时冻结，不在该 dev-10 上继续调权重。
下一步先扩大 split-1 样本，判断以下现象是否稳定：

1. Distance SR 是否仍高于 baseline；
2. SPL 是否能保持不下降；
3. 长路径成功是否只集中于少数异常 episode；
4. 平均 step、API attempts 和时间开销是否仍接近翻倍。

只有扩大样本后确认效率问题具有系统性，才新增独立的 path-cost/commitment 消融；
不能在当前 10 项上继续按结果调参。

### 15.9 42-subtask 中样本对照与下一阶段决定

在冻结 dev-10 的基础上，继续运行 split-1 的 `0.06–0.17` 区间。新增区间包含
32 个 subtasks；与原 dev-10 合并后，两组各有 **42 个不重复 subtasks、6 个 scenes**。
新增运行没有覆盖原结果。

合并结果如下：

| 指标 | Capped baseline v3 | ActiveTopo v4 | 差值 |
|---|---:|---:|---:|
| Snapshot SR | 28.57% | 30.95% | +2.38 pp |
| Distance SR | 52.38% | 52.38% | 0.00 pp |
| Snapshot SPL | 21.98% | 22.47% | +0.49 pp |
| Distance SPL | 37.80% | 37.38% | -0.42 pp |
| step | 212 | 250（trace 完整记录 249） | +17.9% |
| HTTP attempts | 450 | 547 | +21.6% |
| 累计运行时间 | 1:18:19 | 1:28:21 | +12.8% |

Snapshot success 的逐题比较为 ActiveTopo 3 胜、2 负、37 平；Distance success 为
2 胜、2 负、38 平。ActiveTopo 有一个 subtask 在三次 HTTP 200 后仍未解析出合法选择，
按失败计入指标；两组均未出现多图 HTTP 400。

新增 32 项本身的开销差异明显小于最初 dev-10：baseline/ActiveTopo 分别为
176/178 step、360/374 HTTP attempts、1:04:46/1:04:25。因而合并后的额外开销主要
受最初 dev-10 的单个 30-step 长轨迹影响，不能再概括为稳定的“接近翻倍”。

按 GOAT 目标类型分解后，效果并不一致：

| 目标类型 | 数量 | Snapshot SR（B/A） | Distance SR（B/A） | Snapshot SPL（B/A） | Distance SPL（B/A） |
|---|---:|---:|---:|---:|---:|
| category | 13 | 38.46% / 46.15% | 38.46% / 46.15% | 29.66% / 37.32% | 29.66% / 37.32% |
| description | 15 | 26.67% / 20.00% | 60.00% / 60.00% | 19.44% / 7.82% | 47.43% / 41.54% |
| image | 14 | 21.43% / 28.57% | 57.14% / 50.00% | 17.58% / 24.38% | 35.03% / 32.99% |

这组结果支持“持久拓扑对 category 目标可能有效”，但不支持把同一套 CLIP R 投影
无条件用于所有目标类型。description 目标的 snapshot 指标和两项 SPL 均下降；image
目标则表现为 snapshot 提升、distance 下降。42 项仍不足以形成正式论文结论，但已足够
否定“现在直接按完整模式跑 278 项即可验证整体提升”的假设。

ActiveTopo 的 6 份 trace 合计显示：

- stable frontier ID match rate：`886/(886+107)=89.22%`；
- 249 个完整 step event、987 个 candidate；
- `ACTIVE=798`、`VERIFY=40`、`FOLDED=142`、`SUPPRESSED=7`；
- A 非默认候选观测 111 次，geodesic progress evidence 40 次；
- recovery level 0/1/2 分别出现 223/5/21 步；
- blocked-target retry、terminal navigation failure 和 soft/hard correction 均为 0。

这证明稳定 ID、R/C/A、候选状态和 recovery 已在真实运行中工作，但尚没有真实轨迹
证明 blocked suppression 或 SemanticCritic soft correction 带来收益。

#### SPL 非有限值校准

两组在同一个失败任务 `00813-svBbv1Pavdk_0_2` 上都出现 `gt distance=inf`。旧 logger
对失败样本计算 `0 * inf / inf`，产生 `NaN`，导致聚合 SPL 无法使用。该问题不由
ActiveTopo 引入。现在 `src/logger_goatbench.py` 先判断 success：失败任务的 SPL 固定为
0；成功任务遇到无效距离也保守记为 0。历史结果已使用
`scripts/sanitize_goatbench_spl.py` 校正，原始 pkl 保存在同目录的
`.pre_spl_sanitize.bak` 备份中。新增 3 项 logger 测试后，本地测试为 **23/23 通过**。

#### 冻结结论

42 项的总体结论是：**Snapshot SR 小幅提高，Distance SR 完全持平，两项 SPL 基本持平；
尚不能宣称 ActiveTopo 整体优于 baseline。** 下一阶段不继续在这 42 项上调连续权重，
也不立即付费运行完整 278 项。优先新增并在独立区间比较以下离散消融：

1. persistent stable topology only（不改变候选排序）；
2. stable topology + C/A；
3. category-only ActiveTopo projection，description/image 保持 baseline 选择路径；
4. full ActiveTopo v4。

若 category-only 版本在未参与开发的区间保持 SR/SPL 改善且开销可控，再冻结该版本并
运行完整 278-subtask 正式对照。

### 15.10 三种消融模式实施状态

三种模式已经通过统一的 `active_topology.mode` 枚举实现：

| mode | 持久图 | C/A 参与决策 | R 参与决策 | ActiveTopo 候选视图 |
|---|---:|---:|---:|---:|
| `stable_only` | 是 | 否 | 否 | 否，保持原 HGR 候选和 prompt |
| `confidence_accessibility` | 是 | 是 | 否 | 是，R 固定为中性常数且不展示给 VLM |
| `category_only` | 是 | 仅 category | 仅 category | category 开启，description/image 回退原 HGR |
| `full` | 是 | 是 | 是 | 所有目标开启，即冻结 v4 |

行为门控覆盖候选构建、局部到原始 frontier 索引映射、导航失败处理、recovery 和
SemanticCritic 图修改。`stable_only` 以及 `category_only` 的非 category 子任务只写入
持久图和 trace，不改变原 HGR 的候选顺序、prompt、失败终止和 SemanticCritic 副作用。
`confidence_accessibility` 中所有候选内部 R 固定为 `0.5`，因此不影响相对排序，也不会
生成基于 R 的 `VERIFY/FOLDED`；VLM 只看到 C/A/U。

三个配置继承同一份冻结 v4 配置，只覆盖实验名和 mode：

- `cfg/eval_goatbench_activetopo_stable_only_qwen3vl_dashscope.yaml`；
- `cfg/eval_goatbench_activetopo_ca_qwen3vl_dashscope.yaml`；
- `cfg/eval_goatbench_activetopo_category_only_qwen3vl_dashscope.yaml`。

配置继承由运行入口的 `extends` 加载器完成；Qwen3-VL、DashScope 北京端点、seed 77、
40 图预算、`choose_every_step: true`、导航参数和成功阈值均继承冻结值。三个实验名和
结果目录互不相同，不覆盖 baseline/full v4。模式拼写错误会在任何付费 VLM 请求前
直接报错，避免静默运行错误版本。

新增测试覆盖模式策略、object/category 归一化、C/A 模式无 R 状态、C/A prompt 不展示
R、stable-only 原始 frontier 顺序、配置继承与独立结果名。本地验收为：

- `py_compile` 通过；
- `unittest discover`：**28/28 通过**；
- 不需要真实 API 的测试没有发出 DashScope 请求。

第一轮使用已经固定的 `0.00–0.17` 42-subtask 区间进行诊断，复用已有 baseline v3 和
full v4 结果，只新增三组付费运行。该区间用于选择结构，不作为独立泛化证据。选出的
候选随后必须在未参与开发的区间与 baseline/full 同场比较，才能决定是否运行完整
278 subtasks。

### 15.11 三组消融的 42-subtask 开发结果

三个消融均已在与前述 baseline v3 / full v4 相同的 `0.00–0.17`、42-subtask、6-scene
开发区间完成。每组均有 42 条主指标记录、6 份 episode trace 和独立配置快照；结果目录为：

- `results/exp_eval_goatbench_activetopo_ablation_stable_only_qwen3vl30b_dashscope`；
- `results/exp_eval_goatbench_activetopo_ablation_ca_qwen3vl30b_dashscope`；
- `results/exp_eval_goatbench_activetopo_ablation_category_only_qwen3vl30b_dashscope`。

所有 SPL 已按同一 logger 规则处理：失败任务为 0，成功但 ground-truth distance 无效的任务
也保守记为 0。总体指标如下，帧数为每 subtask 的平均总帧数：

| 方法 | Snapshot SR | Snapshot SPL | Distance SR | Distance SPL | 平均总帧数 |
|---|---:|---:|---:|---:|---:|
| Capped baseline v3 | 28.57% | 21.98% | 52.38% | 37.80% | 72.17 |
| Full ActiveTopo v4 | 30.95% | 22.47% | 52.38% | 37.38% | 74.79 |
| Stable-only | 30.95% | 22.98% | 59.52% | 43.05% | 72.00 |
| Stable topology + C/A | 21.43% | 14.62% | 42.86% | 28.01% | 67.14 |
| Category-only ActiveTopo | 26.19% | 20.94% | 52.38% | 35.34% | 70.83 |

针对 13 个 object/category 子任务，baseline / full / stable-only / C/A / category-only 的
task success 分别为 `38.46% / 46.15% / 53.85% / 30.77% / 46.15%`，对应 SPL 为
`29.66% / 37.32% / 38.49% / 25.75% / 29.93%`。因此 category-only 的 category success
比 baseline 高 7.69 pp，但其总体 Snapshot SR、Snapshot SPL 和 Distance SPL 分别低
2.38、1.04 和 2.46 pp，未满足“其余主要指标下降不超过 2 pp”的预设门槛；C/A 则在
所有总体主指标上明显退化。

#### 解释与冻结决定

`stable_only` 按设计只写持久图和 trace，不改原 HGR 候选顺序或 VLM prompt，却在本次在线
运行中优于 baseline。这表明冻结 seed 不能使 DashScope 的在线生成完全可复现，42 项上
stable-only 的正向差异不能归因于稳定拓扑本身；同样，category-only 的小样本 category
收益也还不足以作为泛化结论。各组均完成且未见 HTTP 400，但没有任何一个行为改变的消融
通过预设的全局进入门槛。

因此不基于这 42 项选择 ActiveTopo 变体，也不直接启动完整 278-subtask 对照。后续若继续
评估 ActiveTopo，应先在未参与开发的区间使用固定 VLM 响应缓存或多次独立重复，以分离
策略差异和在线生成噪声；Persistent Goal-Conditioned Belief Topology 仍按第 16.4 节的
独立冻结协议验证，不能把本节结果作为其有效性证据。

## 16. Persistent Goal-Conditioned Belief Topology v1

### 16.1 实施边界

新增模式 `active_topology.mode: belief_category`，仅接管 GOAT `object/category`
子任务。`description/image` 通过行为门控继续使用原 HGR 的候选顺序、prompt、导航失败
语义和 SemanticCritic 副作用。现有 `stable_only`、`confidence_accessibility`、
`category_only`、`full` 配置与结果目录未修改。

本版本实现以下四部分：

1. `src/goal_belief.py`：来源感知 Evidence Memory、相关视点簇、log-Bayes posterior；
2. 独立节点概率、Top-K 与 Unknown，不把多个合法实例做互斥归一化；
3. `NAVIGATE / VERIFY / EXPLORE / UNKNOWN_EXPLORE` expected utility 与 Top-2
   混合决策；
4. `src/active_verification.py`：TSDF 观察环、可见性/新颖度/可达性筛选，以及条件式
   单图 VLM 验证。

持久拓扑继续只负责 stable ID、结构置信度 C、Beta 可达性 A 和原始执行映射；目标存在
posterior 不再由 frontier stable match 或导航成功/失败直接增强。普通未检测不作为负
证据，主动 VERIFY 的负结果也只更新 posterior，不触发 hard cascade。

### 16.2 冻结接口与行为

- 目标证据来源为 `CLIP_GOAL / VLM_ROOM_PRIOR / DETECTOR / CONTEXT_PRIOR /
  ACTIVE_VERIFY_VLM`，reliability 固定为 `0.5 / 0.7 / 0.8 / 0.3 / 1.0`；
- 同一来源和 observation ID 去重；`0.75m + 45°` 内归为相关簇，簇内取加权
  log-Bayes-factor 均值；同簇 VERIFY 覆盖弱证据的融合贡献，但 trace 保留原始记录；
- posterior 使用 prior `0.20`、结构置信度缩放非直接证据、VERIFY 直接证据及
  `[-6, 6]` log-odds 裁剪；语义证据 episode 内不衰减；
- 目标键规范化为 `category:<name>`，同 category 跨子任务复用，不同 category 隔离；
- 只有本步仍存在原始 Frontier/SnapShot 映射且 Pathfinder 可达的节点可执行；历史
  belief 保留但不能生成幽灵导航目标；
- utility 差值 `>=0.15` 时 planner 直选；否则仅把 Top-2 交给 explorer VLM，API 或
  解析失败确定性回退 Top-1；
- VERIFY 在 `1.0/1.5/2.0m` 环上各取 16 个方向，不使用 Habitat 目标 viewpoint 或
  semantic ground truth；到达后先写 detector/CLIP，posterior 仍在 `[0.30,0.70]`
  才调用一次单图 VLM，不执行同请求重试；
- 每个 node/goal 最多 VERIFY 两次，冷却 5 个 global step；验证观察位通过
  `target_point_override` 执行，并用 `look_at_point` 朝向假设位置。

### 16.3 配置、trace 与汇总

正式配置为：

`cfg/eval_goatbench_belieftopo_category_qwen3vl_dashscope.yaml`

它继承冻结 Qwen3-VL/DashScope 配置，保留北京端点、seed 77、40 图预算、导航参数和
GOAT 成功阈值，使用独立实验名
`exp_eval_goatbench_belieftopo_category_qwen3vl30b_dashscope`。

episode JSONL 额外记录原始证据、视点簇及融合贡献、posterior/entropy/conflict、Top-K、
Unknown、四类动作 utility、planner/VLM 决策来源、VERIFY 观察点与结果、rank switch 和
累计归一化动作成本。汇总命令：

```bash
/home/tangyuxin/miniconda3/envs/hgr/bin/python \
  scripts/summarize_belieftopo_traces.py \
  results/exp_eval_goatbench_belieftopo_category_qwen3vl30b_dashscope
```

### 16.4 当前验收状态

离线验收覆盖 evidence/observation 去重、相关与独立视点、同簇冲突、VERIFY 覆盖、
posterior 裁剪与消除、多实例 Top-K/Unknown、goal 隔离、临时消失不可执行、动作阈值、
utility margin、VERIFY 预算/冷却、观察环和单图一次请求。原 ActiveTopo、SemanticCritic
兼容与配置测试一并通过。

当前状态只代表源码和离线闭环已实现，不代表方法指标已经达标。下一步严格按冻结协议：

1. 在 `0.00–0.17` 的 42-subtask 区间作开发诊断；
2. 冻结结构后，在未参与开发的 `0.17–0.30` 区间同时运行 baseline 与
   belief-category；
3. 至少一项主要 SR/SPL 提升 `>=2pp`、其余下降不超过 `2pp`、HTTP 400 为 0 且 API
   attempts 不高于 full v4，才进入第二个未见区间和完整 278-subtask。

## 17. BeliefTopo v2：缓存、在线校准与确认接近

### 17.1 实施状态与兼容边界

状态：**v2 已完成 6 episodes / 42 subtasks 开发实验，作为固定失败对照保留**。

v2 使用独立模式 `active_topology.mode: belief_category_v2` 和配置
`cfg/eval_goatbench_belieftopo_v2_category_qwen3vl_dashscope.yaml`。v1 的
`belief_category`、已有四种 ActiveTopo 消融、42-subtask 结果和输出目录均保留；默认
参数仍采用 v1 Unknown 和即时 CLIP 标定。description/image、A-EQA、GOAT 成功判定与
“只有当前原始映射可执行”的约束没有改变。

### 17.2 v2 新增机制

1. Evidence 在 posterior-before 计算前以 evidence ID 及
   `(node, goal, source, observation)` 做 O(1) 预检；accepted evidence 通过 revision
   使对应 node/goal 缓存失效，C/A 则自然进入缓存键。
2. frontier 图片提取后先进行 stable-ID matching。成功 VLM room prediction 按稳定
   topo ID 缓存；视觉余弦不低于 `0.85` 且位移不超过 `1.0m` 时重用并重建当前
   HypothesisGraph 绑定。API 失败后的 heuristic distribution 不进入长期缓存。
3. Unknown 改为有证据、可执行、可达且未 eliminated 节点的 C/A 加权几何缺失率；
   trace 同时保留 v1 独立乘积，便于诊断节点数量偏差。
4. 每个规范化 category 先缓冲 16 个唯一原始 CLIP cosine，再以 median/IQR 冻结无标签
   标定参数并回放；episode 内冻结后不漂移，不足 16 个样本时不写入 posterior。
5. 主动 VERIFY 获得直接正证据且 posterior 达到 `0.75` 后进入
   `CONFIRM_APPROACH`：最多 5 步、两次接近，只沿当前仍映射到同 stable ID 的真实
   frontier 行动。随后仅当精确类别 detector `>=0.50` 的真实 snapshot 距 confirmed
   frontier 不超过 `2.5m` 时生成确认型 NAVIGATE；正验证本身仍不宣告成功。

### 17.3 新增观测与汇总

episode trace/summary 新增 evidence fast skip、posterior cache、semantic cache、CLIP
warmup/freeze/replay、v1/v2 Unknown 对照，以及 confirmation 的 created/selected/expired/
reversed 和明确结束原因。汇总脚本支持把 v2 与任意 v1、冻结 baseline、full-v4 目录对比：

```bash
/home/tangyuxin/miniconda3/envs/hgr/bin/python \
  scripts/summarize_belieftopo_traces.py \
  results/exp_eval_goatbench_belieftopo_v2_category_qwen3vl30b_dashscope \
  --compare v1=results/exp_eval_goatbench_belieftopo_category_qwen3vl30b_dashscope \
  --compare baseline=results/<frozen-baseline-dir> \
  --compare full-v4=results/<full-v4-dir>
```

v2 实测 Snapshot SR / Distance SR / Snapshot SPL / Distance SPL 分别为
`21.43 / 54.76 / 14.37 / 33.49`，API attempts 为 708。weighted-geometric Unknown
均值 `0.680`，导致 49% 决策为 UNKNOWN_EXPLORE；3 次 confirmation 没有转换为
snapshot。该结果未达标，不进入未见区间或完整 278 subtasks。

## 18. BeliefTopo v3：数据驱动的保守决策与 Snapshot Gate

### 18.1 失败证据与问题归因

v3 不再从总体权重继续猜测改进方向，而是固定分析
`exp_eval_goatbench_qwen3vl30b_dashscope_dev10_capped_v3` 的同一 42-subtask
baseline。13 个 object/category 任务中 5 个成功、8 个失败；8 个失败全部为
`Snapshot=0 / Distance=0`，且没有 API、Pathfinder 或 agent-step 异常。

| Subtask | 目标 | baseline 最终对象 | 归因 |
|---|---|---|---|
| `00800-TEEsavR23oF_0_3` | microwave | tv | 不兼容类别被当成终止答案 |
| `00800-TEEsavR23oF_0_4` | pillow | pillow（错误实例） | 多实例稳定性/关联错误 |
| `00891-cvZr5TUy5C5_0_1` | mirror | picture | 相似外观误选 |
| `00813-svBbv1Pavdk_0_0` | plant | potted plant | 合法别名但实例未与目标区域对齐 |
| `00813-svBbv1Pavdk_0_3` | rug | towel | 不兼容类别误选 |
| `00813-svBbv1Pavdk_0_5` | rug | power outlet | 不兼容类别误选 |
| `00853-5cdEh9F2hJL_0_0` | carpet | coffee table | 不兼容类别误选 |
| `00853-5cdEh9F2hJL_0_5` | hanging clothes | towel | 类别/实例误选 |

原 HGR 在到达任意 VLM 选择的 snapshot 后立即结束 subtask，因此这些错误没有机会恢复。
v1 对照进一步显示：microwave 从 `0/0` 提升至 `1/1`，carpet 从 `0/0` 提升至
`0/1`，但一个 mirror case 从 `1/1` 退化至 `0/1`。这说明 BeliefTopo 已能改善部分
搜索，却需要一个从“到达附近”转换为“可靠终止 snapshot”的显式门控。

### 18.2 v3 方法

`belief_category_v3` 继承 v2 的 stable topology、evidence posterior、CLIP 校准和
semantic cache，但根据 v2 实测退化只在 v3 category/object 分支增加：

1. 保守的训练自由类别兼容表：允许 `plant/potted plant`、`rug/carpet/mat`、
   `hanging clothes/clothes` 等明确映射；明确拒绝 baseline 中的
   `rug/towel`、`mirror/picture`、`microwave/tv`；
2. `SnapshotAcceptanceDecision` 综合类别兼容、detector confidence、对象累计
   `num_detections`、目标 posterior、候选实例稳定性排名和 active confirmation；
3. 精确类别默认要求 detector `>=0.50`、至少 2 次检测及 posterior `>=0.35`；该较低
   posterior 门槛用于保留已有 refrigerator/mirror 成功 case，稳定性仍由精确类别和多次
   检测共同约束；
4. 别名默认要求至少 3 次检测及 posterior `>=0.85`，除非由 active confirmation
   放行，避免第一帧的 `plant→potted plant` 直接终止；
5. 同类别存在多个对象时，如果另一个实例的稳定性分数高出 `0.05`，拒绝当前实例；
6. weighted-geometric Unknown 只保留为诊断值。仅当 Unknown `>=0.75`、Top-1 posterior
   `<0.30`、无 snapshot/confirmation 且连续预算未满 2 次时允许 UNKNOWN_EXPLORE；新
   stable node、Top-1 变化 `>=0.05`、独立 detector/VERIFY 证据或 goal switch 重置预算；
7. category 不再调用 explorer VLM tie-break，按 confirmation、NAVIGATE、VERIFY、
   belief EXPLORE、UNKNOWN_EXPLORE 的稳定顺序确定性选择；description/image 不变；
8. confirmation 生命周期延长为 8 步。重复正验证更新同一状态而不覆盖；approach 后必须
   等待下一轮观测，snapshot 必须满足类别、置信度、`2.5m` 空间关联和实例稳定性；每个
   created confirmation 必须有明确结束原因；
9. gate 同时覆盖 belief NAVIGATE 和原 HGR safety fallback。拒绝对象抑制 5 步且不进入
   prompt；恢复时优先当前最高 utility 的真实 frontier，不写负后验或 hard cascade；
10. evidence cluster 融合按 revision 独立缓存，C/A 变化不再重复扫描证据；相同 stable
    frontier 的 room/context distribution 不重复写证据；CLIP trace 使用独立的
    `calibration_stage` 字段，修复旧汇总 count 为 0 的显示错误。

独立配置为：

`cfg/eval_goatbench_belieftopo_v3_category_qwen3vl_dashscope.yaml`

独立实验名为：

`exp_eval_goatbench_belieftopo_v3_category_qwen3vl30b_dashscope`

v1/v2 模式、配置和结果目录不变。v3 trace/summary 额外记录 Unknown eligibility、category
tie-break 数、confirmation 完整结束率、snapshot 抑制/释放和 fusion cache 命中率。

### 18.3 验证协议

当前实现已通过 62 项离线 unittest。现有 v2 trace 的 policy-only 回放覆盖全部 13 个
category case：旧 trace 的 217 个 UNKNOWN top action 在 v3 规则下只剩 6 个；该回放只
验证策略变化，不能作为反事实 SR/SPL。

下一步运行完整 `0.00–0.17` 开发区间。验收要求为 API attempts `<=547`、category VLM
tie-break 为 0、Predictor calls `<=187`、UNKNOWN_EXPLORE 不超过 category 决策 20%，
且 Object SR/SPL 不低于 `38.46/27.66`。Overall Snapshot SR、Distance SR、Snapshot SPL、
Distance SPL 分别不得低于 `28.57/54.76/19.98/35.80`。达标后才运行未见的
`0.17–0.30`，不自动运行完整 278 subtasks。

## 19. SimpleMemory：最终收缩方案

### 19.1 收缩原因

同一 smoke episode 上，原始 baseline 的 Snapshot/Distance SR 为 `60/80`，v1 为
`60/80`，v2 为 `60/80` 但 SPL 明显下降，v3 进一步下降到 `40/60`。v3 的 26 次
category 决策中有 23 次 UNKNOWN_EXPLORE；同时真实 refrigerator 已有 `0.889`
detector confidence 和 22 次检测，却被 posterior gate 拒绝。该证据说明继续增加
posterior、Unknown、VERIFY、confirmation 和 snapshot gate 会掩盖 HGR 原有的强检测
证据，并增加路径和 API 开销。

因此停止向 BeliefTopo 状态机增加参数。v1/v2/v3 的源码、配置、trace 和结果继续保留，
但不作为当前候选方法继续扩展。

### 19.2 当前方法

新增 `active_topology.mode: simple_memory`，只改变 frontier 的短期重复导航行为：

1. 对每步重新提取的 frontier 匹配 episode 内稳定 `topo_id`；
2. 记录真实的设置目标失败、agent step 失败和到达结果；
3. 一次失败后隐藏该 frontier 5 个 global steps；
4. 连续失败 3 次后继续抑制；
5. 若全部当前 frontier 都处于冷却，只恢复一个失败最少且仍可达的真实 frontier；
6. 所有目标类型继续使用原 HGR 的 snapshot、VLM、SemanticCritic、成功判定和导航器。

SimpleMemory 不计算目标 posterior，不创建 Unknown/VERIFY/confirmation，不执行 snapshot
gate，不增加 prompt metadata，也不进行额外 CLIP embedding。健康 frontier 的原始顺序、
HGR semantic distribution 和 exploration score 均保持不变。它只从 prompt 中临时移除
近期失败的 frontier，并通过显式 projected-to-source index 映射保证 VLM 选择仍对应正确
的原始 `Frontier`。

配置文件：

`cfg/eval_goatbench_simple_memory_qwen3vl_dashscope.yaml`

实验目录：

`results/exp_eval_goatbench_simple_memory_qwen3vl30b_dashscope`

该版本的研究假设收缩为：**稳定 frontier 身份和最小失败记忆能否减少重复不可达导航，
同时不破坏 HGR 原有的视觉语义决策。**

## 20. Persistent Goal-Conditioned Topo Map

### 20.1 方法定义

当前正式开发方向不再把 Topo 图当成 frontier blacklist，而是明确分为两层：

1. `PersistentBeliefTopology` 是 episode 内持续增长的全局事实图；
2. `ActiveTopoView` 是根据当前 GOAT 子目标从全局图生成的临时连通投影视图。

全局图包含 `VISITED / OBSERVED / FRONTIER / HYPOTHESIS` 节点，以及 `NAVIGABLE /`
`OBSERVED_AT / ANCHORED_TO / DEPENDS_ON` 边。它保存所有已探索事实，在 GOAT 子任务切换
时不清空，只在 episode 结束时释放。这里的“完整”是指对 episode 内已经探索到的环境持续
积累，不使用 HM3D ground truth，也不提前获知未知区域。

### 20.2 动态目标投影

每个步骤的 goal-conditioned View 包含：

- 当前 VISITED 节点；
- 与当前类别直接匹配的历史 OBSERVED 节点；
- description/image 或尚无精确类别观测时，现有 HGR CLIP 最相关的历史观测；
- 从当前位置到上述证据节点的 Topo 最短连接路径；
- 当前步骤仍有原始 `Frontier` 映射的全部可执行探索边界。

动态图不复制或修改全局事实。category 使用现有 detector/类别记录，description/image 使用
HGR 已加载的 CLIP 特征，frontier 使用现有 semantic distribution 和 CLIP；不新增
posterior、Unknown、VERIFY、confirmation 或 snapshot gate。

所有当前 frontier 都保留，避免相关性估计不准时提前删除正确方向；动态 View 首先按目标相关
性排序，同分时使用 frontier 到目标证据节点的最少 Topo hop 数，最后才使用 HGR 原始索引。
整个排序不引入加权系数。历史 frontier/hypothesis 只能提供证据或充当图中的事实节点，只有
当前仍存在原始 source mapping 的 frontier 才能直接执行，禁止把它们的旧坐标作为导航目标。

当 Explorer 选择一个已经存在真实 `SnapShot/object` 映射的历史观测时，Topo Map 不再只做
候选排序，而会执行一次安全的图上返回规划：从当前 `VISITED` 节点出发，只沿机器人实际走过的
`NAVIGABLE` 边，寻找与该 `OBSERVED` 节点通过 `OBSERVED_AT` 相连的历史访问位置。规划采用边长
最短路，并将下一枚历史 `VISITED` 节点作为中间导航目标。到达后立即重新采集 RGB、更新全局图
和动态 View，再决定下一步，而不是一次性盲目执行整条旧路线。

该执行路径有三条硬约束：

- 中间航点类型必须是 `VISITED`，不能是历史 `FRONTIER/HYPOTHESIS/OBSERVED` 坐标；
- 最终成功仍必须通过当前真实 `SnapShot/object` 和 GOAT 原始成功判定，Topo 到达不算成功；
- 图不连通或历史航点被当前 TSDF 判为无效时，回退到 HGR 原本的 snapshot 导航。

### 20.3 与 HGR 的接口

Explorer prompt 新增一段简短的动态拓扑摘要，包括全局节点/边数量、当前连通 View 大小和相关
历史观测类别。frontier 图片、snapshot、prefilter、SemanticCritic 和 GOAT 成功判断保持原
HGR。VLM 选择的局部 frontier index 通过 View 显式映射回原始 `Frontier`；选择历史 snapshot
时，导航器接受由安全 Topo 最短路生成的下一枚 `VISITED` 航点。航点到达、规划失败和回退事件
分别写入 trace，导航结果继续写回全局节点。

独立配置：

`cfg/eval_goatbench_goal_topomap_qwen3vl_dashscope.yaml`

独立实验目录：

`results/exp_eval_goatbench_goal_topomap_route_v3_qwen3vl30b_dashscope`

该模式配置只包含 `enabled: true` 和 `mode: goal_topo_map`，不引入新的实验权重或状态机参数。
原 baseline、stable-only、SimpleMemory 和 BeliefTopo v1/v2/v3 均保持独立，不覆盖已有结果。

### 20.4 首轮诊断与 OBSERVED_AT 修复

首轮 `0.00–0.17` 运行结果保存在
`results/exp_eval_goatbench_goal_topomap_qwen3vl30b_dashscope`，四项指标为 Snapshot SR
`26.19%`、Distance SR `57.14%`、Snapshot SPL `21.28%`、Distance SPL `38.97%`。
该轮完成 304 次动态投影和 41 次 snapshot 返回规划，但产生的中间 Topo 航点为 0，因此只视为
“目标投影/排序诊断”，不视为可执行 Topo 路由实验。

根因是主循环每步遍历全部历史 snapshot 时，已有 `evidence_ref` 被重复添加到当前
`VISITED` 的 `OBSERVED_AT` 边，使历史观测错误地逐步连接到机器人经过的所有位置。修复版在
`goal_topo_map` 中将每个具体 snapshot 的捕获锚点固定在首次写入的位置；重复扫描只更新其存活
状态，不改变空间来源。不同 snapshot evidence 合并到同一 OBSERVED 节点时仍可保留各自真实
锚点。其他 ActiveTopo/BeliefTopo 模式继续使用原接口默认行为，避免历史实验路径漂移。

修复版使用新的 `route_v2` 实验目录，避免覆盖首轮诊断结果。付费运行前先用单 scene 验证
`topology_route_waypoints > 0` 和 `topology_waypoint_arrivals > 0`，再决定是否重跑 42 项。

### 20.5 Route v2 smoke 与路线承诺

Route v2 在 `00832-qyAac8rV8Zk` 的 5-task smoke 中生成并到达 17 个中间航点，证明持久图
已真实控制导航；但 Snapshot/Distance SR 与 baseline 同为 `60/80`，Snapshot/Distance SPL
从 baseline 的 `57.79/59.69` 降至 `48.07/50.03`。总帧数由 76 增至 185。

主要失败模式是每到一个中间航点便重新调用 Explorer，导致第一个 image 子任务在
`observed_8` 与 `observed_36` 之间反复改选，形成 `visited_8 ↔ visited_15` 摇摆。Route v3
因此只增加离散的路线承诺，不增加阈值或置信度参数：首次由 VLM 选择历史 Snapshot 后固定其
真实 Snapshot/object 映射和目标 OBSERVED 节点；中间航点到达后直接对同一目标重算下一条安全
VISITED 路径，不调用 Explorer；到达原始 OBSERVED_AT 锚点后结束承诺并进入 HGR 原始 Snapshot
导航。仅当 Snapshot/object 映射消失、图路不可达、航点在当前 TSDF 无效、导航失败或子任务
结束时取消承诺。Route v3 使用独立实验目录，Route v2 smoke 结果保持不变。

### 20.6 Route v4：object 实时检测优先

Route v3 在 `0.00–0.17` 的 42 个子任务上相对同区间 baseline 将 Snapshot SR、Distance SR
和 Snapshot SPL 分别提高约 `2.38/2.38/2.35pp`，但 13 个 object 子任务的 Snapshot SR、
Distance SR、Snapshot SPL 和 Distance SPL 分别下降 `15.38/7.69/13.90/12.13pp`。Trace
显示 object 退化主要来自返回历史 Snapshot 锚点的额外路径；通用类别目标本可使用途中出现的任意
当前实例。

Route v4 不改变持久图、动态目标投影或 image/description 路线承诺，只为 GOAT `object` 增加
实时优先级：每步只检查本轮 RGB 中实际出现、类别精确匹配且满足原 `min_detection` 要求的检测；
存在时生成真实单对象 Snapshot 并沿 HGR 原始 Snapshot 导航，不创建历史 Topo 路线。若历史路线
正在执行，则以 `current_object_detected` 结束承诺并切换到当前实例。没有实时匹配时仍使用 Route
v3 的持久 Topo 图、VLM 历史 Snapshot 选择和路线承诺。该行为由独立配置
`eval_goatbench_goal_topomap_object_priority_qwen3vl_dashscope.yaml` 开启，Route v3 默认关闭，
已有源码行为和结果保持不变。

### 20.7 Route v5：与 HGR 统一的 TaskTopoView

Route v4 仍属于“原 HGR 决策后按需调用 Topo 路由”的局部融合。Route v5 将当前可见对象、历史
Snapshot 和当前 Frontier 全部投影为同一个 `GoalTopoTaskView`，并统一输出三种可执行动作：

- `DIRECT`：当前 RGB 中精确类别匹配的真实对象节点；
- `REVISIT`：仍有真实 Snapshot/object source mapping 的历史 OBSERVED 节点；
- `EXPLORE`：仍有当前原始 Frontier mapping 的持久 FRONTIER 节点。

每个动作都携带稳定 `topo_node_id` 和原始 HGR source mapping。Object 当前实例不再绕过 Topo，
而是作为当前 TaskTopoView 中路径最短的 `DIRECT` 节点确定性优先；没有 DIRECT 时，Explorer 的
Snapshot/Frontier 返回必须先解析成 `REVISIT/EXPLORE`，解析失败不得直接执行未投影的原始目标。
`REVISIT` 沿持久图路线承诺执行，`DIRECT/EXPLORE` 分别调用 HGR 原始 Snapshot/Frontier 导航；
三种动作的到达与失败统一回写同一持久节点。TSDF、Pathfinder、ConceptGraph 感知和 GOAT 成功
判定保持不变。

Route v5 使用独立配置
`eval_goatbench_goal_topomap_unified_qwen3vl_dashscope.yaml` 和独立实验目录
`exp_eval_goatbench_goal_topomap_route_v5_unified_qwen3vl30b_dashscope`。Route v3/v4 默认不开启
`unified_task_view`，已有实验路径与结果不变。

### 20.8 Route v6：按任务类型启用 Topo 行为

Route v5 已在 `0.00–0.17` 的 6 episodes、42 subtasks 上完整运行。区间专属结果为 Snapshot
SR `28.57%`、Distance SR `57.14%`、Snapshot SPL `24.37%`、Distance SPL `35.57%`。
相对同区间 capped baseline，Distance SR 和 Snapshot SPL 分别提高 `4.76pp` 和 `2.39pp`，
但 Distance SPL 下降 `2.23pp`。其中 object Distance SR 从 `38.46%` 提高到 `61.54%`，
Object SPL 从 `29.66%` 提高到 `30.61%`；image Distance SR 则从 `57.14%` 降至
`42.86%`。Trace 共记录 21 次路线承诺、59 次 Topo 航点到达和 18 次完整承诺结束，证明图已
实际控制导航，而不是只作为 prompt 描述。

因此 Route v6 不增加置信度、阈值、效用权重或新状态机，只增加按任务类型划分的 Topo 行为：
object/category 和 description 继续使用 Route v5 的统一 `GoalTopoTaskView`；image 使用
`projection_only`。image 仍用目标图片 CLIP embedding 从持久大图生成动态目标投影，利用相关
历史 OBSERVED 节点和图连接关系组织当前可执行 Frontier，并把 Topo 上下文交给原 HGR Explorer；
但 Snapshot 候选、VLM 选择和 Snapshot 导航保持原 HGR，不允许历史 Snapshot Topo 返回路线或
路线承诺。这样 image 也实际使用持久大图，但图只回答“接下来在哪里找”，不替代视觉模型判断
“哪张 Snapshot 是目标”。

该门控由新配置中的以下唯一新增字段表达：

```yaml
active_topology:
  projection_only_goal_types: [image]
```

新配置为 `cfg/eval_goatbench_goal_topomap_hybrid_qwen3vl_dashscope.yaml`，独立实验目录为
`results/exp_eval_goatbench_goal_topomap_route_v6_hybrid_qwen3vl30b_dashscope`。未配置
`projection_only_goal_types` 时保持 Route v5 的全任务统一路由行为，因而 v5 代码路径、配置和已有
结果不变。

## 21. V7 后续计划

Route V7 已收敛为“Persistent Topological Memory + Goal-conditioned Topology Overlay +
Cost-aware Conservative Decision”。动态置信调分、typed-edge propagation 和 commitment 的失败版本
只保留为消融，不进入最终配置。完整方法、实验结论与最终运行入口见
`docs/HGR_V7_MODIFICATION_PLAN.md`。
