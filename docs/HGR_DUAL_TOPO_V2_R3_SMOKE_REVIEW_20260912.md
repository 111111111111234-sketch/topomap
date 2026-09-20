# HGR V2 R3 首轮 Smoke 复核（2026-09-12）

## 结论

第二次 pin 修复已在线通过。最新 C-persistent 的 7 次 hypothesis arrival 全部完成真实验证，`skipped=false`，不再出现 `Hypothesis node not found`；7 次均触发 falsification 和跨层 retraction。58 次运动确认全部通过且审计零违规。

自动 source audit 也通过：7 次撤销均取消当前 intent、清除所有受影响 hypothesis refs、保留候选集合与独立观测引用，并且 `spatial_facts_preserved=true`。新审计产物为 `v2_trace_audit_source.json`，其中 `cascade_acceptance=passed`。因此 active-intent hypothesis 生命周期和在线跨层撤销缺口可以关闭。

本场景的 7 次验证恰好全部为负向验证，且每次 `cascade_deleted_count=0`，因此终端正向独立证据和多后代 DAG cascade 仍没有在线覆盖；二者由确定性回归测试覆盖。动态恢复分支仍为 `not_covered`。

## 第二次 C-persistent pin 复核（通过）

结果目录：`results/exp_dev_goatbench_hgr_dual_topo_v2_c_persistent_smoke_20260912_180035`

| Distance SR | Distance SPL | Snapshot SR | 步数 | VLM 调用 | verification | falsification | retraction |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 37.5% | 29.9% | 25.0% | 58 | 82 | 7/7 | 7/7 | 7/7 |

- `trace_contract_passed=true`，violations 为空。
- 7 次 retraction 全部 `intent_cancelled=true`、`spatial_facts_preserved=true`，自动 source audit 零违规。
- 撤销前后候选键集合不变，所有 `observed_evidence_refs` 不变，仅受影响的 hypothesis refs 被删除。
- 发生 2 次 `source_detached`，仍无 `source_mapping_invalid`。
- 单场景随机 VLM 运行的 SR/SPL 只作诊断，不用于宣称方法增益。

## 根因记录

第一次 pin 补丁只把 active hypothesis refs 接到了 belief-v2 的 deferred semantic 分支。R3 实际继承 `active_topology.mode: stable_only`，`belief_v2_enabled=false`、`belief_planner_enabled=false`，因此走的是 `update_frontier_map()` 内的 eager stale-node 清理，保留集合没有到达真实清理点。第二次修复把 pin 参数接入 eager 路径，并由主循环在所有 dual-v2 阶段传入 active intent refs；验证或意图释放后集合自然为空。280 项完整回归测试通过。

## 第一次 `verify-r3` 结果（仍未通过 pin 验收）

| 阶段 | Distance SR | Distance SPL | Snapshot SR | 步数 | VLM 调用 | verification |
|---|---:|---:|---:|---:|---:|---:|
| C-persistent | 62.5% | 47.5% | 12.5% | 80 | 77 | 0/7 有效，7 次 node not found |
| D | 37.5% | 29.6% | 37.5% | 72 | 70 | 0/6 有效，6 次 node not found |

两段 trace contract 均通过且零违规；各出现 2 次 `source_detached`，没有 `source_mapping_invalid`。D 的 14 条固定记录缓存回放全部路线等价；在线累计 15759 hits / 3503 misses，其中 13454 同决策、461 跨步、1844 跨目标。这些结果不改变此前已通过的路线与缓存结论，但也不能抵消 verification 的失败。

## 首轮 R3 结果

| 阶段 | Distance SR | Distance SPL | Snapshot SR | 步数 | VLM 调用 | Candidate 时间 |
|---|---:|---:|---:|---:|---:|---:|
| B | 62.5% | 43.1% | 37.5% | 79 | 117 | 322.5 s |
| C-shadow | 50.0% | 36.0% | 0.0% | 107 | 145 | 599.3 s |
| C | 37.5% | 29.2% | 37.5% | 38 | 78 | 76.7 s |
| C-persistent | 25.0% | 21.4% | 12.5% | 41 | 62 | 48.4 s |
| D | 25.0% | 25.0% | 12.5% | 50 | 71 | 80.8 s |

单场景、单次模型运行仍不能用于方法排序。本轮指标的主要用途是发现行为和覆盖问题。

## 已通过的部分

- B/C/D 共 315 次 motion acknowledgement，全部满足路径/游标/到达协议，所有 audit 零违规。
- persistent frontier 修复生效：C-persistent 与 D 不再出现 `source_mapping_invalid` feedback；分别产生 1 和 2 个 `source_detached`，但仍继续到达验证终点。
- C-persistent 产生 3 次、D 产生 4 次 `v2_hypothesis_verification`，证明探索意图不再因实时 frontier 消失而必然中断。
- D 固定记录缓存对照的 12 条 RoutePlan 全部满足 cache off/cold/warm 结果等价。
- D 在线累计 13162 hits / 2990 misses：10894 同决策、1 跨步、2267 跨目标。跨目标复用已经存在，但导航收益仍需 checkpoint 冷暖配对实验。

## 尚未通过的部分

- 最新运行没有正向 verification；终端正向独立空间语义证据尚未在线覆盖。
- 7 次 falsification 的 `cascade_deleted_count` 均为 0；在线多后代 DAG cascade 尚未覆盖，但跨层候选引用撤销已通过。
- 本场景没有触发 local repair、same-target reroute 或 route unavailable，动态恢复在线覆盖仍为 `not_covered`。
- image 目标仍弱，本轮 D 的四个 image 目标全部 Distance failure；按既定范围先保留原协议并使用 `v2_goal_diagnostic` 定位。
- B/C/C-shadow 未开启持续意图，`source_mapping_invalid` 属于原逐步选择对照行为；不应把它与 persistent 模式混合统计。

R3 生命周期补丁无需再重跑。后续在线实验应针对当前未覆盖的正向验证、多后代 cascade 或动态恢复设计专门场景，不应继续重复同一 smoke 来等待随机触发。
