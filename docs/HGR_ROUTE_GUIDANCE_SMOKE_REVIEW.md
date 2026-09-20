# 首轮 correctness / route guidance smoke 复盘

2026-09-10。来源为 `results/exp_dev_goatbench_phase_c_correctness_smoke` 与
`results/exp_dev_goatbench_route_guidance_smoke` 的 subtask_metrics.json、主日志和 active_topology_traces。
两组均有 8 个子任务摘要，主日志均以 All scenes finish 结束。完成运行不等于通过验收。

| 指标 | correctness | route guidance |
|---|---:|---:|
| Distance SR | 12.50% (1/8) | 25.00% (2/8) |
| Distance SPL | 8.68% | 22.68% |
| 总 subtask_steps | 16 | 109 |
| 总路径 | 1.16 m | 45.94 m |
| VLM logical calls（全部用途） | 42 | 136 |
| 主日志总耗时 | 8:44 | 42:52 |
| 非法选择导致终止 | 6 | 4 |
| Verify matched 次数 | 2 | 3 |
| HTTP 失败（telemetry） | 0 | 0 |

## 已定位的问题

10 次选择失败的 30 个响应全部是可解码 JSON，但 candidate_id 不在当前 presented_ids；
这些 ID 全部出现在同子任务的历史 feedback 中。示例：旧 `verify:40@place_0/view:3`
已经被覆盖，新候选是 `/view:2`，模型仍连续三次选择旧 ID。
`query_place_goal.py` 将 recent_feedback 和当前候选一起提供，validated_request 在失败后重复
同一请求，不提供具体的无效 ID 纠正信息，最终主循环结束子任务。
这是新方向 ID、历史反馈与选择协议之间的接口问题，不是 API key 或网络故障。
不能放宽校验接受旧 ID，因为那会重新执行已经检查的动作。

correctness 很快主要因为提前结束：全部路径仅 1.16 m，不能解释成性能优化。
guidance 仍有 39 次 rejected/uncertain，只有一次 frontier_observed；目标层仍偏向反复 Verify。
同一候选多次出现本身不足以证明去重失效，需要核查每次独立来源/局部几何版本是否有实质变化。

guidance 有 44 次安装、105 个执行事件，现有 corridor 重放审计为零违规，说明执行器确实接入，
但不证明真实碰撞、目标身份、路径效率已经通过验收。3 次 matched 仅 2 个 Distance 成功，
matched 与实际距离不一致仍存在。

guidance 全部用途 API telemetry 共 365.21 s，仅占 2572 s 总时间约 14.2%；
候选 build 累计 84.70 s（其中 search 79.72 s，不能重复相加），选择视觉准备 185.43 s，
Verify 视觉准备 16.98 s。剩余时间不能仅凭这些计时全部归因于 topo；需要感知、建图、渲染等阶段计时。

## 结论与下一步

暂不进入固定 12×2。先隔离历史动作 ID 与当前可执行 ID，加入有针对性的非法选择纠正，
用本轮真实失败响应建立回归测试；再排查反复 Verify 的版本触发条件及 matched/Distance 不一致。
修正后使用新输出目录重跑同一 smoke。两组是同一 episode 的连续子任务，后续起始位置和地图不同，
当前存在大量协议提前终止，不能把指标差异直接归因于 corridor 的收益。

本次仅修正审计脚本直接运行时的项目 import 路径；未更改导航行为或历史结果。

## 后续代码修正：选择协议 R2

以上为首轮结果，不会被后续修改覆盖。R2 在 coverage 模式下将当前候选映射为请求内
`choice_N`，表格、图片标签同步映射，历史反馈去除可执行 candidate_id，保留实体及验证结果。
返回值必须属于当前短编号集合，验证后映射回真实候选；不接受过期 ID，也不将其静默转换为新视角。
非法响应的后续请求附带合法编号及字段约束；保留最多三次和 transport 不倍增的限制。
解析器也拒绝列表等非字符串 ID，避免模型畸形输出导致 TypeError。

进一步回放发现 guidance 的 25 次已检查方向重选全部版本不同，25 次有局部几何变化，
其中 8 次还包含来源变化。现有日志不足以区分视角真正解遮挡、物体中心漂移和网格微小更新。
R2 增加 sources_changed / local_bounds_changed / local_mask_changed 诊断，暂不引入任意几何阈值
或关闭几何重新激活；此部分仍需验证。matched/Distance 不一致也未宣称解决。

回归测试覆盖真实旧 ID 返回、历史反馈隔离、当前合法短编号纠正及映射。
新 smoke 配置为 `eval_goatbench_phase_c_correctness_smoke_r2.yaml` 和
`eval_goatbench_route_guidance_smoke_r2.yaml`，输出独立目录；线程设置沿用 16。
