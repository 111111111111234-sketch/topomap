# HGR 统一分层导航器：实现与运行说明

状态：代码与离线回归已接入，完整 unittest 309 项通过。首轮 shadow 已完成，但暴露出
语义协议接受空/缺项 `assessments` 的问题；该轮不构成新策略性能证据。协议与短流程配置
已修复，等待 shadow R2 验证，当前不能声称 Distance SR/SPL 已提升。

## 1. 已实现控制关系

新入口为 `hierarchical_navigation`。它与旧 ActiveTopo 行为、Phase C 和主动 V2-D
配置互斥；底层内部复用 V2-D 的 Place 图、对象 approach、认证跨 Place 路线、source
重绑定、hypothesis pin 和路径缓存。

语义请求只接收原 HGR 的目标图像、Snapshot/Object crop、Frontier 图像及语义预测，
不接收路线长度、Place、terminal 或 waypoint。模型为每个来源返回
`high/medium/low/no_match` 和语义等价排序组。只有 high 对象可成为目标；medium 对象
只能触发新视角验证；没有可信对象时才探索 Frontier。路线只在同一语义组内决胜。

一个已安装 intent 在普通感知和 waypoint 期间持续执行。新稳定对象证据、source/edge
失效、恢复耗尽或终点验证结果才触发高层重选。object 目标使用类别一致的新稳定对象事件；
description/image 不用类别启发式替代语义判断，任一新稳定视觉对象都会触发一次重新评估。

同 Place 由精确 Place ID 判断，使用原 TSDF 局部控制器直达真实 terminal，不安装
`RouteGuidance`。跨 Place 继续执行认证 transitions 和最终 TSDF terminal。局部失败的
几何恢复仍由 V2 runtime 完成，失败不会自动删除空间边。

终点到达只代表几何到达。新鲜 RGB 会触发目标类型对应的结构化 VLM 验证：

- confirmed：登记独立观察；只有 TARGET_APPROACH 可以结束任务；
- rejected：撤销目标关联和没有独立支持的 hypothesis 后重新选择，Place/边不删除；
- uncertain：导航到一个新的可达观察点，最多再验证一次；
- error：不产生正负证据，对未变化的请求摘要临时抑制对象选择，转向其他来源。

## 2. 配置

| 配置 | 用途 |
| --- | --- |
| `eval_goatbench_hierarchical_navigation_shadow.yaml` | 在 V2-D 执行旁记录新决策，不改变执行策略 |
| `eval_goatbench_hierarchical_navigation_smoke.yaml` | 固定 8-task smoke，启用完整统一闭环 |
| `eval_goatbench_hierarchical_navigation_train.yaml` | 固定 12×2 development 配置 |

配置默认进行两次语义 JSON 解析尝试、最多两次终点验证，并启用路径缓存。object、description、
image 三类目标共享同一状态机。

## 3. 运行顺序

```bash
cd /home/hdd/tangyuxin/projects/Hypothesis_Graph_Refinement
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 NUMEXPR_NUM_THREADS=16

# 离线回归
/home/tangyuxin/miniconda3/envs/hgr/bin/python -m unittest discover -s tests -v

# 先运行 shadow
CUDA_VISIBLE_DEVICES=7 /home/tangyuxin/miniconda3/envs/hgr/bin/python \
  run_goatbench_evaluation.py \
  -cf cfg/eval_goatbench_hierarchical_navigation_shadow.yaml --split 1

# 再运行完整闭环 smoke
CUDA_VISIBLE_DEVICES=7 /home/tangyuxin/miniconda3/envs/hgr/bin/python \
  run_goatbench_evaluation.py \
  -cf cfg/eval_goatbench_hierarchical_navigation_smoke.yaml --split 1
```

每次正式运行前应复制配置并修改 `exp_name`，避免覆盖既有结果。在线重点检查：
`hierarchical_semantic_decision`、`hierarchical_intent_installed`、
`hierarchical_waypoint_arrived`、`hierarchical_terminal_verification`、
`hierarchical_goal_end`。普通 waypoint 事件的 `semantic_request_triggered` 必须为 false。

## 4. 验收边界

离线测试覆盖语义硬门控、同组路线决胜、medium revisit、同 Place 直达、Frontier 信息增益、
事件生命周期、两次验证、错误非负证据、hypothesis pin、序列化和新配置隔离。

在线仍须依次完成 shadow、R4 towel 单任务、8-task smoke、12×2 三重复和冻结 holdout。
最终仍使用 Distance SR/SPL、Snapshot SR/SPL、路径、耗时及 95% CI gate。只有 gate 通过后
才能表述为导航性能提升；代码接入和单元测试通过本身不构成性能证据。

## 5. Shadow R1 复核（2026-09-13）

R1 因 shadow 配置错误继承 development 清单，实际完成 12 个场景、85 个子任务，而非短程
smoke。V2-D 冻结执行得到 Distance SR/SPL `0.3294/0.2119`、Snapshot SR/SPL
`0.1412/0.0973`；这些数字只描述冻结执行路径，不是统一导航器的收益。

结构审计共覆盖 584 次运动确认，路线违规为 0；62 次 hypothesis 撤销的来源不变量全部
通过。新语义层记录 277 次决策，其中 273 次最终选择均携带
`safe_default_after_missing_or_invalid_assessment`，与冻结 V2-D 路线仅 26/277 一致。
根因是解析器将空或缺项的 JSON 数组当成成功，随后把缺失对象填成 `no_match`、Frontier
填成低置信 fallback。

R2 修复包括：候选清单显式写入 prompt、图像标签与 JSON ID 的逗号格式统一、空/缺项数组
严格判失败并按上限重试、失败响应摘要和置信分布写入 trace、导航统计按子任务复位，以及
shadow 配置改用固定短程清单。R1 结果保留作故障证据，不与后续性能结果混合。
