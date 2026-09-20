# HGR topo 融合与 baseline smoke 对照

复核对象：baseline `exp_dev_goatbench_hgr_baseline_smoke_20260910_202124`；
融合 `exp_dev_goatbench_hgr_topology_fusion_smoke_20260910_204529`。
两者最终指标均包含相同的 `00062-ACZZiU6BXLz/ep_0` 八个子任务，无无效指标。
早期融合目录 `exp_dev_goatbench_hgr_topology_fusion_smoke_20260910_200849` 无最终
`subtask_metrics.json`，不混入本次完整结果，也不删除。

| 指标 | baseline | 融合 |
| --- | ---: | ---: |
| Distance SR | 75.00% (6/8) | 87.50% (7/8) |
| Distance SPL | 43.08% | 44.12% |
| Snapshot SR | 25.00% (2/8) | 50.00% (4/8) |
| Snapshot SPL | 6.41% | 27.24% |
| 日志总耗时 | 19:25 | 28:12 |
| 累计动作步数 | 43 | 73 |
| 累计子任务观测帧数 | 145 | 232 |
| 累计运动距离 | 39.37m | 63.73m |
| VLM logical calls | 85 | 96 |
| VLM telemetry 累计耗时 | 189.33s | 145.45s |

融合多成功一个 image 任务；总耗时增加 8:47（45.2%），动作增加 69.8%，
距离增加约 61.9%。API 累计耗时反而减少，不能将变慢归因于 API。
额外运动和感知步数明确存在，但现有计时不足以把剩余耗时精确分配给感知、建图和局部执行。

## 同 ID 配对

编号为 subtask ID 最后一段，SPL 为 Distance SPL。

| 编号 | 类型 | 成功 baseline→融合 | SPL baseline→融合 | 步数 baseline→融合 | 距离 m baseline→融合 | 融合 next-hop 数 |
| --- | --- | --- | --- | --- | --- | ---: |
| 0 | image | 是→是 | .9033→.4543 | 6→11 | 5.92→11.77 | 0 |
| 1 | image | 是→是 | .1423→.6340 | 1→1 | .63→.22 | 0 |
| 2 | object | 否→否 | 0→0 | 1→1 | .51→.10 | 0 |
| 3 | image | 是→是 | .3466→.2624 | 16→23 | 15.85→21.18 | 3 |
| 4 | image | 否→是 | 0→.7773 | 9→11 | 7.80→8.42 | 4 |
| 5 | description | 是→是 | .3704→.5464 | 2→10 | 1.57→8.40 | 3 |
| 6 | object | 是→是 | .6841→.4818 | 7→12 | 7.09→10.06 | 4 |
| 7 | description | 是→是 | 1→.3733 | 1→4 | 0→3.58 | 0 |

第 0 个任务没有跨 Place next-hop，却已明显增加距离，不能把所有额外路径归为 waypoint
绕路。此时原选择器的拓扑提示可能改变探索决策，模型随机性也是未控制因素。
第 5 个任务起始最短距离为 baseline .58m、融合 4.59m；此前任务结束位置不同会传递到
后续任务。因此同 ID 配对不是同起点、同观测下的纯路线实验。

## 机制检查

- 融合日志有 31 次 `HGR topology selection context`，原选择器确实收到拓扑信息；
  这能证明输入接入，不能单独证明某次决策由路线文本导致。
- Place 图最终 11 个节点、20 条 valid 有向边，审计 valid，0 violation。
- 45 次路线计划：31 same-place、14 next-hop；14 waypoint 到达、14 commitment resume，
  4 次 commitment 创建和 4 次完成。step trace 中没有 target_arrived 与 active
  place_route_commitment 同时出现。
- 路线汇总中断连、源映射失败、waypoint fallback 均为 0。
- 融合 VLM purpose：hypothesis_predictor 33、goat_prefilter 31、goat_explorer 31、
  semantic_critic 1；96 次全部成功，无独立 place_goal_select / place_goal_verify。

结论：拓扑输入与跨 Place 执行链路确实生效，当前 smoke 完成；这次观察到成功数增加，
但路径和耗时成本显著增加，不能称为效率改善或泛化通过。
下一步应对第 0/3 个任务检查探索决策，对第 4/5/6 个任务检查路线及终端段，
再以相同模型、清单和执行器补 route-only 对照，区分提示选择和路线执行的作用。
不根据这一个 episode 新增类别阈值或改变预算/停止标准。历史结果未修改。
