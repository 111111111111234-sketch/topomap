# HGR 双层动态拓扑地图：后续开发规格

2026-09-12：生命周期、终端新观测验证、RoutePlan 复用和缓存归因已通过回归与在线 R3 验收，自动 source audit 为 `passed`；受控空间记忆对照工具已就绪，导航收益仍待 pilot 与冻结实验，见 [R3 修复说明](HGR_DUAL_TOPO_V2_R3_FIXES.md)。

## 新开发接入：V2-B/C/D（机制验收完成，收益待验证）

统一契约、共享 RoutePlan 执行、GoalState 与假设撤销、路径缓存以及受控冷暖地图工具
已写入独立配置。按用户要求未由开发者运行测试或评测，不代表 A-R2 或 B/C/D 已通过。
实际接口、阶段开关和用户执行命令见 [V2-B/C/D 开发与运行手册](HGR_DUAL_TOPO_V2_BCD_IMPLEMENTATION.md)。
下文保留历史状态；冲突处以该手册的当前实现边界为准，方法目标仍遵循 V2 主规格。

## 下一版主规格：双层动态主地图 V2

最新状态：V2-A 首轮 smoke 未通过后，R2 已修有效 approach / 未完成 frontier hop 的保持，
增加几何查询耗时记录；242 项离线回归通过（16 线程），R2 在线待运行。
详见 [V2-A R2](HGR_DUAL_TOPO_V2_A_IMPLEMENTATION.md)。

V2-A 对象 approach 已接入独立配置，详见
[V2-A 实现及启动](HGR_DUAL_TOPO_V2_A_IMPLEMENTATION.md)。完整离线回归 239 项通过，
16 线程；在线 smoke 尚未运行。其余 V2 阶段仍为设计，不能将 V2-A 视作完整主地图验收。

综合参考方案、当前代码与闭环轨迹，下一版采用
[HGR_DUAL_TOPO_FUSION_V2_SPEC.md](HGR_DUAL_TOPO_FUSION_V2_SPEC.md)。
该文优先于本页和连续执行文中的冲突设计；本页以下实现说明及历史结果继续保留。
V2 是设计修订，尚未实现，不等于现有融合 smoke 已具备完整双层主地图。

最终闭环：HGR 感知/假设更新持久空间层，在同一 Place 身份上维护当前目标层，
原选择器结合两层状态选择真实对象/frontier，拓扑产生到真实终端的路线，HGR 局部执行
消费认证 guidance 并回写反馈。原观察预算和确认/停止协议固定，不恢复独立 Phase C
逐对象 Verify 循环。Topo 作为全局主地图与局部度量地图协作。

实施顺序：**V2-A 对象 approach 与完整成本 → V2-B 连续路线 → V2-C 目标层与事件驱动
→ V2-D 跨任务增量复用**。先解决已经观察到的 capture Place 折返，再改善选择和缓存。
节点采样、额外 fresh verification 与宽 corridor 后置单独消融。
保留所有历史结果，16 线程；长程与场景经验收益分别用受控任务验证，不按任务序号推断加速。

## 当前方向修订：HGR 与 topo 融合（2026-09-10）

本节优先于下文历史 Phase C 的独立 Explore/Verify 选择与验证设计。
目标是让同一张持久 Place 图同时参与 HGR 的下一目标选择和实际导航，保留原 HGR
的候选、视觉推理、frontier 假设、观察预算和最终确认/停止协议。
不再把逐对象、逐方向的独立 VLM Verify 循环作为本轮主开发路径。

当前融合实现：HGR 每步感知继续更新 Place 图和观测/frontier 归属；原选择器收到候选
对应的 approach Place、valid 有向边路线、图路径长度和 frontier 已知终端段长度，
结合原视觉证据与假设选择 Snapshot/Object 或 Frontier。拓扑不重排或额外裁剪候选，
不增加模型请求或图片，原图像预算与预筛选继续生效，提示编号在筛选后对齐。
选择后复用 B-R5 Place 路线执行与同目标续行；中间节点到达不触发最终停止。
执行观测与障碍处理继续回到同一 Place 图，原 HGR 处理真实终端段与最终确认。

边界：Snapshot 路线终点仍为其历史 capture Place，不是认证对象观察点；图距离不包含
当前实际位姿 connector 和 HGR 最后局部段，提示中明确区分。无图路线保留显式诊断与
原局部安全处理，不能宣称全部动作均沿 topo corridor 执行。此轮关闭独立 Phase C
和它绑定的 route guidance；全程 corridor 融合需后续单独验证，不改停止协议。

代码：`src/hgr_topology_context.py`、原 `query_vlm_goatbench.py` /
`eval_utils_gpt_goatbench.py` 选择链路与主入口；独立配置：
`cfg/eval_goatbench_hgr_topology_fusion_smoke.yaml`、
`cfg/eval_goatbench_hgr_topology_fusion_train.yaml`。
历史 Phase C、B-R5 配置和结果保留。已有运行进程不会自动加载此修改。
详细命令和验收见 [融合实现说明](HGR_TOPOLOGY_FUSION_IMPLEMENTATION.md)。

融合与 baseline smoke 已完成，见 [配对复核](HGR_TOPOLOGY_FUSION_SMOKE_REVIEW.md)：
Distance SR 75%→87.5%，SPL 43.08%→44.12%，耗时 19:25→28:12，步数 43→73。
31 次原选择器拓扑输入、14 次跨 Place waypoint 到达/续行确认接入生效；
性能与效率尚不能判定通过，后续子任务起点差异和模型波动限制因果解释。

进一步的[额外路径与主地图接入分析](HGR_FUSION_EXTRA_PATH_ANALYSIS.md)明确：当前融合是
空间图候选上下文与 B-R5 执行，尚未完成双层动态主地图。任务 0 存在探索分岔，任务 3/5
出现 capture Place 与对象终端衔接的折返。最终方向是持久空间导航层与 HGR 目标证据层
共用 Place、意图和执行反馈；跨任务复用不等于重复运行独立 Verify。主地图与局部度量执行
协同，优先修正目的地/完整路线成本，再完善统一目标层和连续执行。详见上述分析。

日期：2026-09-06；整体设计更新：2026-09-10。  
状态：A/B 为已完成开发基线；B-R5 完成 12×2；历史 C-R1/C-R3/C-R4 smoke 均已完成，验收未通过。
目标覆盖修正与保守 corridor 内连续前视执行已接入，在线验收待运行，见
[实现说明](HGR_ROUTE_GUIDANCE_IMPLEMENTATION.md)。  
适用范围：HGR 上后续的空间拓扑、目标动态层和导航接入开发。

## 1. 目标与阅读约定

实现一个 training-free 的双层动态拓扑导航系统：第一层表达探索中逐渐建立的空间连通结构；第二层针对当前目标组织证据并选择探索或验证目标；底层继续由 HGR 的度量地图执行运动。

当前方向为“持久空间拓扑 + 目标动态覆盖层 + 拓扑指导的连续局部执行”。
Topo 决定有效路线/transition，Place 不要求成为必须踩中的精确点；普通节点经过不触发高层推理。
新的详细接口、状态语义、实施顺序和验收约束见
[连续执行设计](HGR_TOPO_GUIDED_CONTINUOUS_DESIGN.md)。本文件与该文共同作为当前主规格；
第 10 节保留 A/B/C 开发历史，历史的 next-hop 行为不表示 corridor 已实现。

已完成实验复核见 [三个 smoke 报告](HGR_PHASE_C_THREE_SMOKES_REVIEW.md)：
C-R4 为 50.00% SR / 23.90% SPL / 49:38，B-R5 smoke 为 87.50% / 47.82% / 17:37。
R4 更快并不代表方法更好，也不能由单场景八任务推断泛化。

本文是后续代码修改的主规格。历史实现与实验结论参考 `V7_REGION_TOPOLOGY_EXPERIMENT_ARCHIVE.md`；历史版本保留用于比较。本文中的“必须”为验收要求，“建议”为可根据接口调整的实现选择。新增模块名、字段名和配置名均为建议接口，不代表已存在。

核心限制：

- Object、Description、Image 共用空间图、目标状态和规划逻辑；差异集中在输入目标编码与目标匹配适配器。
- 不以 scene ID、episode ID、数据集名称、物体类别列表选择行为或阈值。
- 地图构建只使用执行时允许获取的观测、位姿和已探索空间；目标 GT、成功距离和官方最短路径只能用于评测。
- 地图结构必须真正用于路线与下一跳规划，不能仅作为缩小 VLM 候选的名称包装。
- 不把“零回退”“topo 使用率高”作为优化目标；优先考虑任务完成、路线质量和计算成本。
- 每次加入机制都需说明其解决的结构问题；不再按失败场景增加独立补丁规则。

## 2. 当前实现的问题与证据边界

当前 `region_topology.py` 将 VISITED anchor 作为 Region，按最高分成员排序 Region。此时 Region 更接近“观测位置分组”，不能直接解释为房间或稳定空间区域。

`persistent_belief_topology.py` 会给 frontier 添加到不同 visited anchor 的 ANCHORED_TO 边；`_anchor_for_action` 返回第一条匹配边。这混合了“从哪里看到”“属于哪个空间”“从哪里可以到达”三种关系，且归属可能依赖边插入顺序。

现有 dynamic 层已具备证据、uncertainty、freshness 和 typed-edge support，但 Region 排序仍主要来自成员最高相关性；不能把它直接宣称为完整的区域信念规划。

V7.4、V7.5、V7.4.1 说明不同候选约束会明显影响结果。单次小集结果不能证明具体因果，也不能证明某种拓扑表示普遍无效。历史“snapshot 禁止导致过度探索”等解释应视为待轨迹验证的假设。

## 3. 整体架构：两层图，一个执行底座

```text
RGB-D / 位姿 / HGR 当前对象和 frontier
                    |
           HGR 已观测度量地图
                    |
       第一层：持久 Place 连通图
       Place、可通行边、对象/观测归属
                    |
       第二层：当前目标的动态覆盖层
       每个 Place 的目标证据、已搜索状态
                    |
          选择探索或验证目标
                    |
        第一层图搜索生成路线与有序 transition
                    |
       已知自由空间 route guidance / corridor
                    |
          HGR 连续局部路径与动作执行
                    |
        观测 / 成功到达 / 路径失败反馈
```

HGR TSDF 是执行底座，不计为第三层拓扑图。第二层复用第一层的 Place ID 与邻接关系，不再维护第二套空间坐标和独立连通图。

### 职责边界

| 组件 | 职责 | 不承担的职责 |
| --- | --- | --- |
| HGR 度量地图 | 已知自由空间、局部碰撞与局部路径 | 目标信念与区域优先级 |
| Place 图 | 空间身份、已知连通、路线和 transition 证据 | 根据某个目标改写空间几何 |
| Goal 覆盖层 | 目标支持、搜索记录、动作目标选择 | 创造未经验证的可通行边 |
| 目标匹配适配器 | 将不同目标输入转换为可比较的证据记录 | 为不同任务类型另建规划状态机 |
| VLM | 基于当前观测和候选证据选择目标/验证对象 | 凭语言描述宣布通道可通行 |
| 路线 guidance 适配器 | 认证路径、corridor 与连续前视进度 | 用完整场景捷径代替已知拓扑路线 |
| 局部执行器 | 在 guidance 下连续前进到真实终端，报告结果 | 自行切换全局探索目标 |

## 4. 第一层：最小可用 Place 图

### 4.1 Place 的含义

首版采用“局部可通行区域的代表节点”，不做房间语义分割。一个房间可以有多个 Place，一个 Place 不保证代表整个房间。避免为了命名 Region 而引入房间识别模型。

此定义继续用于 B-R5 空间身份。下一轮先增加度二链压缩的 routing view，保留内部节点映射、
观察归属、方向、frontier 接入和路径展开；它属于第一层查询视图，不另建空间图。
doorway/junction 等结构标记作为后续独立几何实验，不立即改为语义模型采样节点。

建议节点字段：`place_id`、代表位姿、创建时间、最后观测时间、关联 observation ID 集合。空间归属依据几何和连通性，房间标签只能作为可选元数据。

增量建图规则保持单一：

1. 当前位姿附近存在已有 Place，且在已知自由空间内存在短可行连接，则关联该 Place。
2. 否则创建新 Place。
3. 邻近的判定采用米制 Place 间距参数；候选关联还必须检查连通，不能仅靠欧氏距离跨墙合并。
4. 若多个候选满足条件，选已知局部路径最短者；平局用稳定 ID。

首版只需稳定关联，不实现复杂在线图合并/拆分。空间地图有较大修正时，先显式使受影响归属失效，再重新关联；不得静默更换所有历史 ID。

### 4.2 观测、对象与 frontier 的归属

明确三个不同字段，禁止互相代替：

| 字段 | 含义 | 更新方式 |
| --- | --- | --- |
| `observed_from_place_ids` | 从哪些 Place 看见该证据 | 随观测追加 |
| `capture_pose` | Snapshot 拍摄位姿 | 对该张观测保持不变 |
| `approach_place_id` | 执行该目标应从哪个已知 Place 接近 | 按当前已知可行路径计算，可更新 |

frontier 是已知空间边界上的探索目标，未知一侧不能直接创建“已知房间”。frontier 的 approach Place 应位于其已知自由空间一侧，并通过已知局部路径连接。

对象的世界位置、Snapshot 的拍摄位置和到对象的安全观察位置也不能混为一谈。导航验证使用安全观察位置；capture anchor 只作为历史证据来源或可选观察点。

多张 Snapshot 应可共享一个稳定对象 ID。目标证据优先绑定稳定对象或 Place，不按每张图累积成多个独立目标。

### 4.3 可通行边

首版只维护 Place 间的可通行边用于图搜索。边由实际走过的轨迹，或已知自由空间中的有效局部路径产生。

建议字段：

```text
source_place_id, target_place_id
path_length_m, local_path_reference, certified_path_geometry
source_anchor, target_anchor, local_geometry_revision
status: valid / invalid
last_verified_step, supporting_observation_ids
```

不把 OBSERVED_AT、语义相似或语言推测关系放入可通行邻接表。图不连通时保留不连通状态，不用直线或语义相似度补可走边。

corridor 必须从实际路径几何或已知网格路径产生；单有端点和字符串 reference 不够。
相关几何更新时重新验证局部路径，禁止让 corridor 膨胀穿过未知空间或跨墙。

局部路径失败先重新检查相关路径：确认失效才使边 invalid；一次控制动作失败不能判定永久不可达。后续新观测可以重新验证并恢复边，不添加场景专用冷却规则。

边可以有方向；只有经过验证的反向通行才可建立反向连接。平地双向局部规划成功时，可一次验证双向边。

## 5. 第二层：轻量目标动态覆盖层

### 5.1 生命周期与状态

第一层跨 episode 内子任务保留，episode 结束清空。第二层每个新目标重新投影；历史可复用的 RGB、对象、几何证据留在第一层，不将上一目标的否定结论迁移到新目标。

第二层建议保留以下最小信息：

- Place 内目标证据引用和支持分数；
- 候选对象是否需要再次观察；
- 已搜索区域/视角记录；
- 当前高层目标与所选路线。

动作状态必须包含 observation region、相关证据和局部几何版本；拒绝/不确定只覆盖实际检查视角。
不能永久用 `(object, Place)` 标记所有视角已检查，也不能因为无关地图更新就解封。
API/解析失败与控制失败不属于语义负证据；Continue/Retry/NoAction/Error 必须在接口上区分。

先不同时增加多套 posterior、utility、freshness 衰减、revisit penalty 和手写 confidence margin。已有状态可复用，但启用项必须在配置里明确列出，不能继承历史实验叠加参数而不说明。

### 5.2 证据语义

每条证据至少含 `observation_id`、`entity_id`、`place_id`、位姿/视角、目标匹配分数、来源和时间。匹配分数是模型输出，不自动等于校准概率。

同一观测重复进入 pipeline 不重复计数；相近视角观测按已有去重机制处理。对同一稳定对象优先维护代表观测与新验证结果，避免每张 snapshot 让区域信心不断上升。

首版可使用对象级去重后的最强观测作为候选支持，并显式称为“支持分数”。不要把任意 VLM 分数直接作为独立 Bernoulli 证据进行 Bayesian 更新。

负证据必须有实际观察覆盖支持：走到拍摄点或没检测到目标本身不够。若当前代码不能证明目标可能出现区域已被观察，首版只记录“已检查的视角”，不降低整个 Place 的目标支持。

初版不启用跨 Place 的目标概率传播。相邻并不意味着目标存在性可复制；后续要传播，只作为探索先验的独立消融，不能产生“已看到目标”的证据。

### 5.3 两类高层目标

仅保留两类决策意图：

| 意图 | 目标 | 到达后的行为 |
| --- | --- | --- |
| Explore | 有真实 frontier 的 Place/边界 | 观察未知空间，更新图和候选 |
| Verify | 有目标支持的对象安全观察点 | 重新观察/匹配，按现有成功判断协议处理 |

DIRECT 是当前局部可验证目标的执行特例，无需独立复杂状态机。历史 Snapshot 是 Verify 的证据，不应自动作为终止信号，也不应自动转换成 Explore。

所有输出先通过当前真实源映射与几何可达性检查。保留目前有效的目标确认协议，避免同时改地图结构和停止条件。

## 6. 选择、路线与执行

### 6.1 先拆开接口，再改变策略

第一轮接入沿用已有 HGR 的目标选择结果，只把它映射到 Place 图并生成路线。这一步用于隔离“图是否可正确导航”，不检验新候选排序。

确认路线正确后，第二轮才让目标动态层生成 Explore/Verify 候选，VLM 根据支持证据和路线信息选高层目标。沿用当前可用的匹配模型，不增加第二个规划模型。

首版尽量保留全部有效候选；受图像预算限制时先做稳定对象去重和代表图选择，记录被裁剪候选，不硬编码“只有前两个 Region 才存在”。

输入给决策模块的每个候选应带：意图、Place ID、目标支持/代表图、真实已知路线长度、验证或探索目的。这样可以比较导航代价，避免依赖名称不透明的 utility 分数。

### 6.2 路线与连续执行

在第一层 valid 边上使用 Dijkstra 或 A*，首版只用已知路径长度作权重。暂不把置信度、风险、freshness 混成带多个系数的路径代价。

图搜索终点为 approach Place；终端局部段负责到 frontier 或对象观察点。路径长度和经过节点应完整记录。

现有 B-R5 使用 edge target anchor 下一跳，并已持续保存目标与执行进度。
下一版先沿认证路线选最远局部可达前视点，随后将有序 transition 与已知自由空间 corridor
交给受限局部规划器。经过节点不要求到中心、不触发停止或高层 VLM。
同 Place 或局部已知可达终端可直接执行，但必须满足碰撞和已选 transition 约束。
局部执行仍为连续闭环，不能取消每步感知与碰撞检查。

当前 HGR 局部执行仍使用 Habitat Pathfinder；它不能仅凭 terminal 或 preferred_direction
执行 corridor 约束。新适配器需使用已知空间路径或传入认证轨迹并验证输出轨迹。
详细接口见 [连续执行设计第 6 节](HGR_TOPO_GUIDED_CONTINUOUS_DESIGN.md#6-连续-route-guidance-的具体契约)。

### 6.3 重规划和失败

采用事件驱动重规划，只保留三个触发类别：

1. 当前目标已到达、完成验证或失效；
2. 路线被新几何证据判定不可行且同目标重路由也无法恢复；
3. 当前观测产生需要按既有协议立即验证的目标。

局部执行层保留已有无进展检测。停止重复动作后回到上述统一重规划接口，不再增加 Region 专用计时器、类别 TTL 或独立失败黑名单。

局部路径重算、同目标拓扑重路由、高层语义重选是三种不同事件，分别记录。
普通 Place 通过、新 RGB、路径浮点代价变化不能自动触发高层推理。
新目标证据默认增量登记；抢占只沿用可检验的立即验证协议，不引入含糊的“更强分数”阈值。

异常分清：源映射失效、图不连通、局部路径失败、模型请求失败。沿用现有有限重试；没有合法行动时显式返回失败原因。实验开发期可以保留原 HGR 高层安全回退，但必须单独记录它重新选择目标的次数。

不能为了方法叙述删除安全处理；也不能把调用 TSDF 的每一步称为回退。

## 7. 边描述：后置、结构化、可消融

当前动态边已有长度、可靠性等模板文字。首版新 Place 边先完成结构化连接信息；文本由字段生成，不为每条边新增 VLM 调用。

可选例子：`Place A -> Place B：已验证通行，已知路径 3.2m。`

后续若已有视觉证据支持，可加“经门口”“走廊连接”等短标签，并保存来源 observation ID。未知填 unknown；语义标签不能创造可通行边。

只有决策模块实际消费边描述时，它才构成行为机制。日志里的 description 不等于语言参与规划。是否提供描述必须以“同一结构图、有/无文本描述”的消融检验。

## 8. 建议代码接口与改动范围

新增独立实验模块，避免继续把所有分支塞入评测主循环：

```python
# 建议接口，非现有 API
scene_graph.update(observations, pose, known_map) -> GraphDelta
scene_graph.resolve_approach(target, known_map) -> ApproachResult
goal_view.update(goal, observations, graph_delta) -> GoalView
selector.select(goal_view, candidates) -> NavigationIntent
router.plan(current_pose, intent, scene_graph) -> TopologicalRoute
guidance.build(route, known_map) -> RouteGuidance
executor.step(guidance, observation) -> ExecutionFeedback
```

| 文件/模块 | 后续职责 |
| --- | --- |
| `persistent_belief_topology.py` | 现有证据适配与兼容；迁出独立 Place 建图接口 |
| 建议 `place_topology.py` | Place 身份、归属、有效可通行边和增量更新 |
| `goal_topology_overlay.py` / `place_goal_navigation.py` | 当前目标覆盖层；修正证据版本、视角覆盖与失败生命周期 |
| `topology_navigation.py` | 已有 anchor 执行基线；后续接入 route guidance 与同目标重路由 |
| `tsdf_planner.py` 的路径适配器 | 认证局部路径/连续前视/corridor 内规划，不能只传目标点 |
| `region_topology.py` | 历史 V7.4 分组策略保留作对照，不再扩充补丁 |
| `query_vlm_goatbench.py` | 候选与稳定 ID 映射，维持模型请求适配 |
| `run_goatbench_evaluation.py` | 调用接口、记录结果；尽量不包含算法细节 |

算法模块不得 import GOAT manifest、读取 episode ID 决定策略或访问评测 GT。GOAT 的字段转换放在入口适配器。

## 9. 最小参数预算

首版最多新增一个空间尺度参数：Place 代表节点间距，单位为米。需要的碰撞容差、局部到达距离、无进展预算优先复用现有执行器设置。

去重和模型图像预算沿用现有公共配置。新参数若确有必要，先说明不能复用现有定义的原因，并设计跨范围敏感性测试；不按失败 scene 搜索参数。

实验开关只体现开发阶段：`shadow`、`route_only`、`goal_navigation`。边文字描述为后期单一消融开关。不同阶段共享同一组几何参数，不复制一套近似值。

## 10. 分阶段任务与验收

### 阶段 A：空间图 shadow

实现状态（2026-09-07）：已完成代码接入、164 项结构/回归测试、首次 smoke
结构审计、已知 TSDF 自由空间连接和冻结决策离线重放工具。实现位于
`src/place_topology.py`，通过
`active_topology.place_topology.stage: shadow` 接入。该阶段不会读取 Place
图来改变候选、目标、路线、停止或动作输出。

任务：实现 Place、观察关系与 approach 关系分离；修复依赖第一条锚定边的归属；生成有效连通边；不改变导航输出。

验收：

- 相同输入重放时，调换边插入顺序不改变归属；
- 隔墙的近点不因欧氏距离近而合并；
- frontier 重提取保持可解释的 approach 映射；
- 每条可通行边可追溯到已知路径或实际轨迹；
- 小轨迹重放中，shadow 开关前后导航候选和执行输出一致。

交付：模块、独立测试、地图可视化、shadow 配置和审计结果。无需先跑完整导航集。

当前交付对应关系：

| 交付 | 路径或输出 |
| --- | --- |
| Place 图模块 | `src/place_topology.py` |
| 独立结构测试 | `tests/test_place_topology.py` |
| shadow 配置 | `cfg/eval_goatbench_goal_topomap_v7_place_shadow_qwen3vl_dashscope_train.yaml` |
| smoke 配置 | `cfg/eval_goatbench_goal_topomap_v7_place_shadow_qwen3vl_dashscope_train_smoke.yaml` |
| 地图可视化 | `scripts/render_place_topology.py` |
| 在线审计 | episode summary 的 `place_topology.audit` |

阶段 A 有意保持的限制：不执行复杂 Place 合并/拆分或边语义描述。Place 间边来自
实际执行轨迹或当前 `tsdf_planner.island` 内的已知自由空间路径；frontier approach
也只接受该已探索网格内的有效路径。Place 模块不接收 Habitat Pathfinder，因而不会
把完整场景 geodesic 当作拓扑捷径。

首次 smoke（`00062-ACZZiU6BXLz/ep_0`，8 subtasks）得到 24 个 Place、
23 条 valid 轨迹边、76 个 observation 和 13 个 frontier；13 个 frontier
均存在 approach 映射，`place_topology.audit.valid=true` 且违规数为 0。
两次独立在线 VLM 运行不用于证明逐动作严格相等，因为 API 输出和轨迹可能不同；
shadow 不干预性由调用边界与单元测试保证，后续 route-only 前仍应补同一冻结决策
输入的离线重放对照。

R2 已为每个 step 记录 pose、已验证连接、observation/frontier 增量、冻结导航输出
和 shadow 输出。实际 trace 已离线重放 42 steps / 42 decisions，得到
`navigation_mismatches=0`、`graph_audit.valid=true`；重建图为 9 个 Place、20 条
valid 边，且没有 invalid 边。阶段 A 的结构与不干预性验收因此关闭。

### 阶段 B：route-only

实现状态（2026-09-07）：已新增 `src/topology_navigation.py` 并接入
`place_topology.stage: route_only`。独立配置使用 `stable_only`，确保旧 ActiveTopo、
V6/V7 排序、direct-first 和旧拓扑路线不会改变原 HGR 的目标选择。稀疏连接修复后
代码已通过 173 项结构/回归测试，等待 R2 train smoke 验证在线路线与诊断。

任务：输入 HGR 选出的同一目标，通过 Place 图得到路线，局部执行下一跳；目标选择与停止协议保持固定。

验收：

- 路线全部使用 valid 边；不可达返回显式结果；
- 同 Place 目标、跨 Place 目标、边失效重规划都正确；
- 不以完整 Habitat 场景 geodesic 为在线规划捷径；
- 记录图路径、实际轨迹、下一跳和失效原因，检查是否有结构性绕路。

交付：路由模块、执行接入、同目标路线对照、少量 train smoke。若失败先修图或路由，不改目标分数。

当前实现行为：

- 同 Place：不覆盖目标，交给 HGR 执行原终端局部段；
- 跨 Place：只在 valid 有向边上运行 Dijkstra，并覆盖为下一 Place waypoint；
- 跨 waypoint：保持同一个 HGR 已选 source，不重新选择语义目标；
- 多个 frontier approach：按图路线长度加终端段长度选择总成本最短的可达项；
- 图不连通或 source 无映射：显式记录原因，并保留同一 HGR 目标作为开发期安全路径；
- waypoint 被当前 TSDF 明确判为障碍：使对应边 invalid；后续已知自由空间重新验证
  可以恢复该边并统一重规划；
- Place 路由不调用 Habitat Pathfinder；Habitat Pathfinder 仍只存在于原 HGR 局部
  执行和原有评测/诊断路径。

交付路径：

| 交付 | 路径 |
| --- | --- |
| 路由模块 | `src/topology_navigation.py` |
| 路由测试 | `tests/test_topology_navigation.py` |
| 完整开发配置 | `cfg/eval_goatbench_place_route_only_qwen3vl_dashscope_train.yaml` |
| smoke 配置 | `cfg/eval_goatbench_place_route_only_qwen3vl_dashscope_train_smoke.yaml` |
| 稀疏连接 R2 smoke | `cfg/eval_goatbench_place_route_only_qwen3vl_dashscope_train_smoke_r2.yaml` |

首次 route-only smoke（`00062-ACZZiU6BXLz/ep_0`，8 subtasks）完成 36 次
路线规划，其中 6 次使用跨 Place 下一跳；目标映射失败、图断连、waypoint 回退和
非法边均为 0，Place 图审计通过。该次运行同时暴露出连接候选继承全部历史邻居会把
11 个 Place 扩张为 110 条有向边的完全图。实现已将候选收敛为当前位姿附近 Place
加刚离开的 active Place；这保留反向连接验证，但不再把旧邻接复制给新 Place。
R2 使用独立输出重跑后得到 9 个 Place、20 条 valid 有向边（14 条实际轨迹边、
6 条已知自由空间边），不再是完全图；28 次规划中 4 次使用跨 Place 下一跳，
其中包含一条经过 5 个 Place 的路线。目标映射失败、图断连、waypoint 回退、非法边
和 API 失败均为 0，Place 图审计通过。当时将其判定为 Phase B smoke 通过并扩大到
12×2；后续发现该验收遗漏了 waypoint 到达后的续行，故只能认定为图结构通过。

### B-R3：修复 waypoint 与语义目标到达混淆（2026-09-08）

12×2 两组均完成 173 subtasks。HGR baseline Distance SR/SPL 为
55.49%/39.31%，旧 Route-only 为 50.29%/33.76%。这些是单次闭环运行的观察结果。
按 Route-only 是否使用 next-hop 分组属于运行后的描述性统计，不能独立证明
退化由 Place 中心或某种任务类型引起。

已定位实现缺口：`agent_step` 对任何当前局部目标到达都返回 `target_arrived`；
新 Place 路由未像旧路由一样拦截中间到达，Snapshot 因此进入原停止分支。
此外 `place_route_plan` 每步清空，不能用它追踪跨多个 local step 的执行。
旧 trace 中 76 次 Snapshot 到达时最近路线仍是 next-hop，其中 48 次 Distance
失败。这支持提前停止缺口存在，不表示修复后这 48 次必然成功。

R3 实现：

- 用 `PlaceRouteExecution` 保存当前已安装的 waypoint，生命周期跨 local steps；
- waypoint 到达由执行层消费，不能触发 Snapshot 停止或 Frontier 最终验证；
- 保留同一 source 继续规划，进入目标 Place 后安装原 HGR 终端观察段；
- 重绑定 Snapshot 时保留原选对象 ID 集合，不把整张图的 cluster 恢复为目标；
- 对象/图像失效显式释放；预算结束前未完成的路线记录 release；
- 汇总新增 Place 路由事件，避免旧统计器输出 plans=0 误导验收。

该修复不引入阈值、目标类别分支或 GT 信息。176 项测试通过，包括多步、多跳、
最后一个 Place 到达仍须执行终端段，以及对象集合保留。边入口/出口锚点仍是
可能的后续几何改进，尚无证据将其视作本次退化的确定原因。

配置：`cfg/eval_goatbench_place_route_only_qwen3vl_dashscope_train_smoke_r3.yaml`
和 `cfg/eval_goatbench_place_route_only_qwen3vl_dashscope_train_r3.yaml`，使用独立结果目录。
R3 smoke 必须检查 `place_route_waypoint_arrived`、同 source resume、
终端段启动与最终完成顺序；不能再仅凭图审计/next-hop 次数宣布执行通过。
通过后重跑固定 12×2 与已有同模型 baseline 配对，确认路线质量后进入 C。

R3 smoke 已完成：51 次规划（40 same-place、11 next-hop）产生 10 次 waypoint
到达、10 次同 source resume 和 5 次进入终端段；trace 中不存在 waypoint 仍 active
时的 semantic `target_arrived`。0 断连、0 waypoint fallback、0 执行失败，图审计
通过；115 次 VLM 请求均成功。8-subtask 的 SR/SPL 受单次闭环 VLM 输出影响，仅用于
检查执行链路，不能作为性能结论。下一步是同一固定 12×2 上的 R3 配对运行。

R3 的完整 12×2 运行完成 173 subtasks。Distance SR 为 60.69%，高于同模型
baseline 的 55.49%，但 Distance SPL 为 32.39%，低于 baseline 的 39.31%。
跨 Place 的 89 个 subtask 平均步数从 baseline 的 5.27 增至 13.58；同 Place 的
84 个 subtask 则略有改善。这说明路线已被完整执行，但当前节点中心 waypoint 带来
额外路径代价。

R4 只修正边的几何执行端点：每条 valid edge 保存实际验证得到的
`source_anchor` 和 `target_anchor`；执行下一跳使用该边的 target anchor，视向下一条
边的 target anchor，避免每条边强制回到 Place 代表点。轨迹边锚点来自实际离开/进入
位姿，已知自由空间边锚点来自当前观测位姿和已验证目标端；它们进入 audit。Dijkstra
权重、Place 尺度、HGR target identity 与停止协议均不变。R4 配置为
`cfg/eval_goatbench_place_route_only_qwen3vl_dashscope_train_smoke_r4.yaml` 和
`cfg/eval_goatbench_place_route_only_qwen3vl_dashscope_train_r4.yaml`；177 项测试通过。

### B-R5：边执行进度与观测归属分离

R4 smoke 虽然 Distance SR/SPL 为 75%/48.47%，但不通过执行验收：
同一 frontier 的 `place_5 -> place_7` 锚点在 step 21–57 重复到达 37 次，
任务最终耗尽预算。锚点位于两个 Place 的重叠区域；局部执行已到达，
观测关联仍选 source Place，导致下次重规划重复同一条边。先前关于节点中心
必然造成 SPL 下降的表述属于未经控制实验验证的解释，不能作为确定因果。

R5 在 `PlaceRouteExecution` 保存已到达的边目标 Place ID。只有局部执行确认
waypoint 到达才推进此状态；同目标续行从该节点搜索 valid 边，并保持已选
approach Place。空间图的 `current_place_id` 继续按观测几何关联，不被执行进度
强制覆盖。终端段、新目标与回退清空执行进度，避免跨目标污染。
trace 分别记录 `observed_place_id` 与 `execution_reached_place_id`。

179 项测试通过，包含观测归属不变时多跳续行、最后一跳后进入终端段、
目标切换清理及边失效后的不可达处理。使用独立 R5 smoke/train 配置与目录；
在线验收需确认重复同一已完成边的循环消失，再扩大运行。该实现仍复用 HGR
局部执行器，单元测试不能证明闭环性能提高或局部路径始终沿记录的边几何行走。

### 阶段 C：goal-navigation

实现说明与运行命令见 [HGR_PHASE_C_IMPLEMENTATION.md](HGR_PHASE_C_IMPLEMENTATION.md)。
使用独立 `place_goal_phase_c` 配置，R5 对照结果以 SHA256 固定。

2026-09-10：三个 smoke 已完成，报告见 [复核](HGR_PHASE_C_THREE_SMOKES_REVIEW.md)。
C-R4 仍有 checked 合并调用、证据重新激活、解析失败恢复和 matched/Distance 不一致问题。
先完成 C-correctness，不能以 210 项单测或审计零 violation 判定本阶段已通过。

2026-09-10 后续修复：启用 `evidence_coverage` 时，地图几何更新不再直接解封
语义已检查视角，也不因无关 Place/边代价变化增加语义决策 revision。
`rejected` 保存该方向的矛盾记录，其他未检查方向仍可验证；`uncertain` 保存未解决覆盖，
同方向仅在新增独立来源视角时重开。局部几何版本继续用于执行失败恢复和可达性检查；
新可达的未检查方向可成为新动作。没有有效 Verify 机会时只保留可执行 Explore，
两者均无时沿用显式 no_action。当前覆盖仍是八方向近似，不证明精确可见性。
本轮完整离线回归 226 项通过，运行线程为 16；观察/图像预算、停止标准及历史结果未修改，
尚未运行本轮在线 smoke，不能据此宣称 Phase C 验收通过。

任务：建立精简的目标动态覆盖层，统一 Explore/Verify 候选；让支持证据与真实路线信息进入选择；执行反馈更新目标状态。

验收：

- 三种目标类型共用决策流程；
- 当前可靠目标可以进入验证；历史匹配不会直接触发无验证成功；
- 视角重复不无限增加支持，目标切换不遗留错误负证据；
- 记录“目标改变”的原因；有效长程任务可显示实际经过的拓扑路线；
- 跑固定开发集进行配对统计，不仅看总均值或回退率。

交付：完整候选到执行闭环和实验报告；结果明确区分机制生效、性能变化和未确认原因。

### 阶段 D：冻结与泛化验证

任务：冻结配置后评估独立场景；再做边描述的独立消融。没有可靠收益时可不采用边语义描述。

验收：提供代码版本、配置、数据清单、模型版本、调用失败统计和完整分母。论文措辞只覆盖已经验证的条件。

## 11. 泛化与实验规范

training-free 仍然可能通过 prompt、阈值与规则开发过拟合。现有 3x2 和 12x2 已被反复用于开发，应明确视作 development 数据，不能再宣称未见验证集。

1. Debug 使用 train 小集；覆盖通道、隔墙、循环、对象多视角、断连等结构现象，不按成功率选择容易 scene。
2. 参数/方案选择使用预先固定的 train 开发集，保留所有失败任务。
3. 场景不重叠的 internal validation 清单在评估前固定。若持续据此修改，就重新标记为开发集，并另留最终 holdout。
4. 官方 val_seen、val_seen_synonyms、val_unseen 按各自协议单独报告；方法冻结后执行。已用于开发的 split/episode 必须如实披露。
5. 额外泛化先测观测采样、节点尺度和对象检测缺失等扰动；其他数据集的迁移在适配完成后单独验证，不能凭无 scene ID 规则就声称跨数据集泛化。

对照至少包含：同模型适配的 HGR baseline、V6、V7.4、route-only、新完整双层方法。必须核实 baseline 源码与共同基础设施修复，不能把 V6 或另一 V7 分支称为原版 HGR。

报告 SR、SPL、路径长度、动作步数、VLM 调用及耗时；统一任务清单和指标修复。新旧代码若运行条件不同，不能直接把历史结果当作严格因果比较。

有随机采样/API 模型波动时进行重复运行；不挑最高的一次。置信区间优先以 scene/episode 为单位重采样，避免把同一 episode 内相关 subtask 当独立样本。

## 12. 必须记录的诊断与测试

每个决策点记录：goal ID、候选稳定 ID、证据引用、Explore/Verify 意图、目标 Place、路线 Place 序列、下一跳、局部执行结果和重新规划原因。

回退分开统计：局部执行、图重规划、原 HGR 高层重新选目标、模型失败。只统计“原 HGR 高层重新选目标”才回答策略是否交回的问题。

必要测试包括：归属顺序不变性、跨墙隔离、重复证据去重、目标切换、无有效路线、路径失效后恢复、Snapshot/对象/source 索引一致性。禁止用 GT 目标信息修补运行时行为。

测试应检验上述行为边界；测试项数量不作为方法正确或有效的证据。基于同一轨迹的重放只能检验实现一致性，闭环策略效果仍需真实导航运行。

## 13. 开发完成标准

满足以下条件才可称为“完整接入双层动态 topo map”：

- 第一层的 Place 和可通行边具有可验证的空间含义；
- 第二层目标状态随着新观测与验证反馈变化；
- 高层输出能在第一层生成认证路线/transition，并由 guidance 接口连续执行；
- 局部失败反馈能更新路线并触发统一重规划；
- Explore 和 Verify 都有完整路径，不需要强制把一种意图转换成另一种；
- 代码中不存在场景/类别专用行为分支，关键参数有公共定义；
- 在固定对照上报告性能与成本，并在独立场景上验证。

近期顺序：C-correctness → 固定目标/停止协议的 route-guidance → corridor 约束执行 →
证据增量处理 → 稀疏 routing view → 完整组合评估 → 独立场景冻结验收。
路线、目标选择、停止协议和节点采样分别消融；本轮不新增 VLM 边描述。
