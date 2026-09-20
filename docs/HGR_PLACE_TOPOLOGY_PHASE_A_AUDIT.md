# HGR Place Topology Phase A Smoke 审计

> 说明：本报告记录首次 smoke。2026-09-07 后续 R2 已将 frontier approach 和
> 回环连接改为仅使用 HGR 已探索 TSDF 网格，并新增冻结输入离线重放；首次 smoke
> 不能替代 R2 验证，R2 结果完成后应追加到本报告。

日期：2026-09-07  
场景：`00062-ACZZiU6BXLz`，episode 0  
任务数：8  
配置：`eval_goatbench_goal_topomap_v7_place_shadow_qwen3vl_dashscope_train_smoke.yaml`

## 结论

Phase A 的首次在线 smoke 完整结束，没有 Traceback 或运行时异常。Place 图仅以
shadow 方式记录事实，没有进入候选排序、目标选择、路线选择、停止条件或动作执行。

结构审计通过：

| 项目 | 结果 |
| --- | ---: |
| Place | 24 |
| valid 有向边 | 23 |
| invalid 边 | 0 |
| Snapshot/observation | 76 |
| frontier | 13 |
| 具有 verified approach 的 frontier | 13/13 |
| 回环关联 | 0 |
| 审计违规 | 0 |

全部可通行边均来自实际执行轨迹，边状态均为 `valid`，证据类型均为
`traversed_trajectory`。本次轨迹没有产生经过已知自由空间验证的回环，因此图表现为
保守的有向轨迹骨架；这符合 Phase A 的安全边界，但进入 route-only 前需要继续验证
跨 Place 路线和回环连接。

## 指标说明

本次 8-task smoke 的 Distance SR 为 75%，Distance SPL 为 37.23%。历史冻结 V7
smoke 的 Distance SR 也是 75%，Distance SPL 为 48.20%。这两次运行调用了在线 VLM，
且历史结果来自较早代码运行，轨迹并不相同，因此 SPL 差异不能归因于 Place shadow，
也不能用来说明 Place 图改进或降低了导航效果。

严格的 shadow 行为一致性应使用同一组冻结模型响应/决策输入做离线逐步重放；独立
API 运行只用于检查集成稳定性和地图结构。

## 产物

- Episode summary：
  `results/exp_dev_goatbench_v7_place_shadow_qwen3vl30b_dashscope_train_smoke/active_topology_traces/00062-ACZZiU6BXLz_ep_0.summary.json`
- Place 图：
  `results/exp_dev_goatbench_v7_place_shadow_qwen3vl30b_dashscope_train_smoke/place_topology_00062_ep0.png`
- 逐步 trace：
  `results/exp_dev_goatbench_v7_place_shadow_qwen3vl30b_dashscope_train_smoke/active_topology_traces/00062-ACZZiU6BXLz_ep_0.jsonl`

## 下一步门槛

在 Phase B 开始改变导航前，先实现冻结输入的离线 shadow 重放检查，并补充：

1. 同 Place、跨 Place 和断边三种路由测试；
2. 已知自由空间路径生成的连接边，而非使用完整场景 geodesic 作为拓扑捷径；
3. 路线只能穿过 valid 边的运行时断言；
4. route-only 仍复用 HGR 的目标选择与停止协议。
