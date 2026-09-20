# Phase C 三次完整 smoke 复核

复核日期：2026-09-10。数据来自当前磁盘输出；本次未运行新的在线实验。

## 数据与比较口径

三个 smoke 分别为 C-R1（目录无 revision 后缀）、C-R3、C-R4，均已记录
`All scenes finish`、8 条 subtask summary 和 8 条 `subtask_metrics.json`。
参照为 B-R5 route-only smoke。四次运行均为
`00062-ACZZiU6BXLz/episode_0`，相同 8 个 subtask ID。

结果根目录为 `../results/`，目录名分别为：

- `exp_dev_goatbench_place_goal_phase_c_qwen3vl30b_dashscope_train_smoke`
- `exp_dev_goatbench_place_goal_phase_c_r3_qwen3vl30b_dashscope_train_smoke`
- `exp_dev_goatbench_place_goal_phase_c_r4_qwen3vl30b_dashscope_train_smoke`
- `exp_dev_goatbench_place_route_only_qwen3vl30b_dashscope_train_smoke_r5`

SR/SPL 使用 Distance 口径；步数、路径来自 metrics；调用数为 telemetry 的 logical_calls，
此处也与日志 HTTP 请求计数一致。总耗时取日志最终 elapsed，含初始化/输出等，非纯推理时间。
各轮 GPU/负载/API 环境未受控，且跨 subtask 复用地图使同 ID 的初始位置和记忆依赖之前轨迹。
因此这些是配对开发观察，不是独立重复、严格因果比较或泛化证据。
R5 manifest 冻结的是历史结果哈希，并非工作区源代码。

## 完整结果

| 指标 | B-R5 smoke | C-R1 | C-R3 | C-R4 |
|---|---:|---:|---:|---:|
| 完成评测 subtask | 8 | 8 | 8 | 8 |
| Distance 成功数 | 7 | 4 | 5 | 4 |
| SR | 87.50% | 50.00% | 62.50% | 50.00% |
| SPL | 47.82% | 38.20% | 36.64% | 23.90% |
| 总步数（metrics） | 76 | 179 | 201 | 98 |
| 平均路径 m | 8.430 | 13.131 | 15.178 | 8.204 |
| 全部 VLM logical calls | 106 | 243 | 229 | 119 |
| Phase C 选择尝试 | — | 104 | 95 | 39 |
| fresh Verify 结果数 | — | 98 | 87 | 34 |
| 路线 plan 数 | 50 | 118 | 129 | 52 |
| next-hop plan 数 | 15 | 15 | 36 | 15 |
| elapsed | 00:17:37 | 02:08:20 | 01:24:40 | 00:49:38 |

R4 相对 R3 elapsed 减少约 41.4%、调用减少 48.0%，但少成功一个任务，SPL 降 12.74 个百分点。
R4 总路径较短不能单独解释为效率改善：失败任务可能提前退出，SPL 也包含失败惩罚。
三个 C 版本均未超过 B-R5 smoke；不应据此进入“方法有效、冻结推广”的阶段。

| subtask 后缀 | B-R5 成功/步数 | C-R1 成功/步数 | C-R3 成功/步数 | C-R4 成功/步数 |
|---|---|---|---|---|
| 0 | 是 / 17 | 否 / 50 | 否 / 50 | 是 / 41 |
| 1 | 是 / 1 | 是 / 21 | 否 / 50 | 否 / 3 |
| 2 | 否 / 1 | 否 / 1 | 是 / 10 | 否 / 2 |
| 3 | 是 / 28 | 否 / 48 | 是 / 27 | 否 / 15 |
| 4 | 是 / 11 | 是 / 6 | 否 / 50 | 是 / 17 |
| 5 | 是 / 6 | 是 / 8 | 是 / 3 | 是 / 10 |
| 6 | 是 / 11 | 是 / 8 | 是 / 9 | 是 / 8 |
| 7 | 是 / 1 | 否 / 37 | 是 / 2 | 否 / 2 |

## 闭环与延迟证据

| 事件 | C-R1 | C-R3 | C-R4 |
|---|---:|---:|---:|
| Verify matched | 6 | 5 | 6 |
| Verify rejected | 39 | 57 | 15 |
| Verify uncertain | 53 | 25 | 13 |
| frontier_observed | 2 | 3 | 2 |
| source_mapping_invalid 反馈 | 3 | 3 | 1 |
| selection 失败 | 1 | 0 | 2 |
| 平均 / 最大候选数 | 56.19 / 80 | 41.09 / 68 | 46.95 / 81 |

C-R1 同一任务对象 `verify:1887` 被选择 21 次；C-R3 的 `verify:3` 在 subtask 1 被选择 15 次。
C-R4 按 Place 关闭动作后，对象仍可能从多个 Place 验证，但不能据此把整个 Place 判空。
大量选择跟随验证完成事件；代码已保持 active intent，普通 waypoint 由 executor 消费。
没有证据支持“每个移动 step 都无条件触发高层 VLM”这一说法。

| 已记录阶段总时间 s | C-R3 | C-R4 |
|---|---:|---:|
| 候选构建（包含路径查询） | 315.65 | 110.08 |
| 选择输入准备 | 342.09 | 347.06 |
| 选择请求处理与等待 | 427.55 | 204.92 |
| 验证输入准备 | 37.22 | 14.57 |
| 验证请求处理与等待 | 178.49 | 93.35 |
| 上述互斥阶段合计 | 1301.00 | 769.98 |

R1 无这些细分计时。planning.search_seconds 已包含在 build_seconds 内，不重复相加。
R4 选择次数更少，但输入准备总耗时未下降；每次约 8.90 秒，R3 约 3.60 秒。
不能直接归因于缓存或单一硬件瓶颈，需记录图片字节量、缓存命中和进程资源负载。
剩余 elapsed 包含观测、感知、建图、其他模型、运动、可视化、输出等，不能全部归给 Hypothesis Graph。

## 验收缺口

三次 C 运行的现有审计均为零 violation，且 R5 reference 哈希校验通过。
这只证明脚本覆盖的约束未触发，不证明导航或验证正确：

1. C-R1/C-R4 各有 6 次 matched，但 Distance 只成功 4 次；两者均在 subtask 2、3
   出现 matched 与 Distance 不一致。需联合当前 RGB、选中对象、坐标/朝向和到达误差分析，
   暂不能判定是目标误认还是执行位置问题。GT 仅参与离线诊断。
2. C-R4 subtask 1、7 的 `failure_stage=parse`，直接结束任务；它们仍有合法候选。
   日志未保存足够原始响应诊断，不能确认是 ID 格式、JSON 还是 support 校验失败。
3. trace step 数为 178/201/96，metrics 为 179/201/98；失败发生在 step trace 写入前可能造成差异。
   汇总必须并列记录，不把缺少 step trace 当作从未执行。
4. `PlaceGoalNavigation._rebind_evidence` 用 `key[0]` 枚举 checked 的实体，
   实际 key 为 `(intent, entity, place)`，故未把 checked 正确重绑定到合并对象。
   helper 单测通过不代表主流程覆盖到了这条路径。
5. `finish` 对执行失败也调用 mark_checked；`is_checked` 没有证据/几何版本失效逻辑。
   因此 C-R4 实际是 goal 内永久关闭该对象-Place 动作；并未实现承诺的新增证据重新激活。
6. revision 把精确 cost/terminal 纳入变化，却未完整覆盖 support/视觉证据内容；
   `no_goal_topology_change` 返回 None，主入口统一 break。应区分“继续执行”“重试”“无合法行动”。

以上是代码审阅发现的机制缺口，不能声称它们已分别解释了全部 smoke 失败。
本次只更新分析和设计，不修改运行逻辑；修复应使用新的代码快照与输出目录。

## 决策

保留双层拓扑方向，撤销“C-R4 已完成整体方案”的表述。
先修正 C 层证据生命周期、解析恢复和验证诊断；另用固定目标/停止协议隔离评估 route guidance。
稀疏化与 corridor 是待检验的改进方向，不预先承诺 SR/SPL 提升。
下一版完整设计见 [主规格](HGR_DUAL_TOPO_GENERAL_DEVELOPMENT_SPEC.md) 与
[连续执行设计](HGR_TOPO_GUIDED_CONTINUOUS_DESIGN.md)。
