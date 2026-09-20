# HGR 动态双层 Topo V2-A→D Smoke 结果分析（2026-09-12）

## 结论

本轮证明了统一 RoutePlan 的选择—执行协议可以在线运行，GoalState 能区分实测对象证据与
frontier 假设依赖，Place 图也保持结构有效。它尚未证明级联纠错或跨任务缓存带来导航收益。
最关键的缺口是：所有运行都没有发生一次 hypothesis arrival verification，因此 V2-C 的核心
验收“假设证伪后撤销候选/意图”没有被触发。

## 汇总指标

| 阶段 | Distance SR | Distance SPL | Snapshot SR | Snapshot SPL | 距离 (m) | VLM 调用 | 总时间 (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| A | 50.0% | 23.4% | 37.5% | 22.1% | 44.69 | 73 | 1362 |
| B | 75.0% | 53.4% | 37.5% | 27.5% | 41.07 | 107 | 1770 |
| C-shadow | 50.0% | 40.1% | 25.0% | 19.1% | 30.74 | 71 | 1348 |
| C | 37.5% | 17.1% | 37.5% | 17.1% | 32.78 | 89 | 1244 |
| C-persistent | 50.0% | 32.7% | 25.0% | 11.5% | 38.02 | 64 | 1301 |
| D | 62.5% | 42.1% | 50.0% | 29.6% | 41.05 | 71 | 1425 |

每个阶段只有同一场景、同一 episode 的 8 个连续子任务。模型选择和前序轨迹改变了后续任务
的实际起点；例如 subtask 2 的记录最短距离在 A 为 0.288 m、D 为 7.528 m。因此以上横向差异
只能用于发现问题，不能作为方法优劣或泛化收益结论。

## 已证明有用的部分

1. **统一路线执行契约成立。** B/C-shadow/C/C-persistent/D 共记录 295 次运动提议，295 次均被
   实际位置确认；每阶段 `v2_trace_audit.json` 都是零违规。所有已选路线的
   `route_changed_after_selection` 都为 false。B→D 共覆盖 13 条多 Place 路线和 22 个 transition。
2. **空间图保持有效。** 六个阶段的 Place audit 全部通过，没有 invalid edge；最终 Place 数为
   6–10，HGR stable ID match rate 为 92.3%–96.1%。这说明目标状态没有破坏持久空间事实。
3. **证据类型确实分开。** C-shadow/C/C-persistent/D 中安装的 32 个 TARGET_APPROACH intent 都带
   实测 snapshot evidence、没有假设依赖；32 个 EXPLORE intent 都带 hypothesis ref、没有伪装成
   实测证据。C 的 goal-state 字段也实际进入了 HGR 选择上下文。
4. **持续意图有降低重选的信号。** C 为 55 步/20 次路线安装，C-persistent 为 56 步/13 次；
   对应 VLM 调用由 89 降到 64。单次非确定运行不能确认因果，但行为方向符合设计。
5. **路径缓存有明显工程信号。** D 记录 9797 hits、2552 misses，命中率 79.3%；candidate preparation
   为 70.8 s，低于 C-persistent 的 119.7 s。该比较尚未控制轨迹，暂不算独立任务收益。

## 出现的问题

1. **V2-C 核心闭环没有覆盖。** 所有阶段的 hypothesis verification、falsification、retraction 都是
   0，`revoked_hypotheses` 始终为空。当前验证只在带 hypothesis 的 frontier 真正到达时执行。
2. **动态 frontier 身份使持续意图提前失效。** C 系列和 D 每轮有 4–6 次
   `source_mapping_invalid`，全部来自 EXPLORE intent。选中 frontier 在地图更新后从实时列表消失，
   runtime 按精确 `topo_id` 查找失败并释放 intent，所以假设 frontier 往往到不了验证终点。
3. **没有覆盖动态路线修复。** 295 次 `ensure_route` 全是 `retained`；没有 local repair、same-target
   reroute、route unavailable，也没有几何导致的边失效。这轮只验证了正常路径。
4. **D 的命中来源不可归因。** 当前只记总 hit/miss，无法区分同一次候选枚举内的重复查询、同目标
   跨步复用和跨目标复用。选择器生成 approach 后，主循环还会对同一个选择重新规划；大量命中可能
   主要是在消除重复工作，而不是保留下来的空间经验。
5. **效果样本不足且非配对。** 8 个连续子任务、单场景、单次 VLM 运行无法支持 SR/SPL 排序。
   image 任务也仍最弱：C-shadow/C/C-persistent 的 image Distance SR 都是 0/4，D 为 1/4。
6. **主要耗时仍在感知和记忆更新。** D 中 perception 585 s、memory update 385 s；缓存降低路线准备
   后，这两项成为下一阶段工程优化重点，但不应先于正确性闭环。

## 下一步修改顺序

1. **先修 frontier intent 的生命周期。** RoutePlan 安装后，即使动态 frontier 暂时从候选列表消失，
   只要认证终点与路径仍有效，就继续执行；同时保留原 hypothesis ref。可对附近新 frontier 做几何
   rebind，但不能仅因精确 topo ID 消失就释放。对象源仍按实体 merge/removal 语义处理。
2. **让 arrival verification 成为可验收事务。** 到达探索终点后先对 intent 保存的 hypothesis ref
   生成结构化 verification feedback，再完成/释放 intent。trace 必须记录 affected hypotheses、候选
   剩余独立证据、intent 是否取消，以及 Place/edge 是否保留。
3. **增加确定性的级联验收场景。** 构造一个必然证伪的 hypothesis、一个仅依赖该 hypothesis 的候选、
   一个后来获得独立实测支持的候选。在线或记录回放必须同时证明前者撤销、后者保留、空间边不删除。
4. **消除选择后的重复规划。** 选择器展示给 HGR 的 RoutePlan 应直接交给 executor；只在地图版本或
   终点发生变化时重新认证。这样既满足“同一条路线”，也能降低 D 中每条路线数百次 cache query。
5. **细分缓存统计。** 为 cache entry 记录创建 goal/step/decision，分别输出 same-decision、cross-step、
   cross-goal hits，并记录节省的规划时间。跨目标命中才属于空间经验复用证据。
6. **完成受控冷暖对照。** 从同一 checkpoint、同一起点、同一目标和固定模型响应分别运行 cache off/on；
   先要求路线与决策等价，再比较查询次数和耗时。之后用多场景、多 seed、独立 holdout 验证 SR/SPL。

在上述 1–4 完成前，不建议直接扩大到完整 train。下一轮最优先的是让至少一次真实 frontier 到达并
触发假设证伪事务，然后验证错误认知收缩而 Place 空间事实保留。
