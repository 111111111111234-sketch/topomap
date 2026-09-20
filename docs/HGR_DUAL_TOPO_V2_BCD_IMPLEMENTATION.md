# HGR V2-A→D：开发接入与用户运行手册

2026-09-12 R3 修复已完成在线验收，具体变化与结果见 [R3 修复说明](HGR_DUAL_TOPO_V2_R3_FIXES.md)。以下首轮状态属于历史记录；路线合同、active-intent hypothesis 生命周期、在线撤销和缓存等价性均已通过，空间记忆导航收益仍待受控实验。

状态：统一路线、GoalState、级联撤销、路径缓存、研究检查点及评测工具均已写入；R3 自动 source audit 已通过，单场景冷暖地图 pilot 已配置。
用户返回的首次运行结果：V2 专项 21 项通过；完整回归 263 项中 262 项通过、1 项错误。
错误来自旧主循环片段测试未注入新增的 dual_v2_enabled 变量，尚未执行到行为断言。
已补齐测试环境并增加 B 跨 Place 续行、持续意图生命周期的主循环测试；修复后待用户重跑。
按用户要求，开发者没有执行测试、导航评测或模型请求。在线验收仍未通过，原 A-R2 仍先验收。

## 1. 实际接口与边界

`src/hgr_dual_topo.py` 定义 Candidate、NavigationIntent、RoutePlan、Feedback、GoalState。
空间图保留唯一 Place 邻接；目标层引用原对象/观测和 hypothesis DAG。
当前目标的候选、撤销集合、观察位姿记录与 active intent 由 GoalState 持有；
RoutePlan 是几何产物，CertifiedExecution 单独持有提议/确认游标。

新模式的 `resolve_terminal_route` 委托统一规划器，完整路径经过当前 connector、
有序边 source/target anchors 和真实终端。选择上下文、执行和成本来自该路径。
选择后地图变化可导致重认证路径改变，trace 显式记录 selector_route_id 与变化标志。
无法认证时返回失败并重新选择，不静默使用原 Pathfinder 路径；旧 A 回退协议保留。
Frontier 终端复用 HGR 沿探索方向向已知侧退回的几何规则，查询不修改导航状态。
有效对象终端保持；终端被新障碍占据时，新模式重新生成同对象观察点，不凭图像拍摄点回退。
仍是窄认证路径上的前视执行，不是宽 corridor 优化器或多楼层导航。

运行时检查真实 source：对象只沿已记录 alias 合并重绑定，frontier 消失释放意图；
对象中心或 frontier 终端变化时重新认证同目标。普通 transition 不触发停止。
运动提议不推进已到达状态，实际位置返回后 acknowledge 才推进。
B 保留 R2 的跨 Place 续行，在终端局部段恢复原 frontier 选择节奏；C-persistent
才将持续意图扩展到整个 frontier 终端，二者不混作无损缓存。
局部失败允许一次同目标认证重建后的执行恢复；再次失败才释放，没有永久边黑名单。

V2-C 的 semantic critic 沿原验证触发工作，不增加 Verify 请求。
级联首先保留带独立 observation reference 的事实分支；失去前提的推断被撤销。
目标层同步移除假设支持，清除真实 frontier 上失效的语义预测；真实 frontier 和空间边保留。
有独立实测来源的候选不随先验一起删除。对象来源图像不等于已确认目标匹配。
原 frontier-arrival 验证调用的 RGB 不保证是运动后终端的新视角，因此仅获得 positive/observed
标记不自动成为独立证明。独立来源默认来自真实 Snapshot 构造；其他验证适配器必须显式提供
observation_id 与 independent_support，不能用旧协议的成功标志推断终端可见性。
observed_pose_only 记录仅证明采集视角，不推断整个 Place 已搜索完，也不生成负证据。

目标切换重置 GoalState 与执行，保留同 episode 空间图和客观证据。
路径缓存按完整、局部受限搜索域以及端点哈希，路径外出现新捷径也会使相关查询失效。
每次仍搜索当前 valid 图边。未实现模型答复缓存或跨目标匹配分数复用。

## 2. 分阶段配置

所有新配置均独立输出；不替换旧 baseline、A、Phase C 文件。

| 阶段 | 配置后缀 | 新增行为 |
| --- | --- | --- |
| A-R2 | `v2_a_smoke_r2` | 原 R2 对象 approach / hop 保持 |
| B | `v2_b_train/smoke` | 统一认证路线与实际执行 |
| C-shadow | `v2_c_shadow_train/smoke` | 只登记目标状态，不影响提示和纠错 |
| C | `v2_c_train/smoke` | 目标上下文、独立证据保留、假设撤销同步 |
| C-persistent | `v2_c_persistent_train/smoke` | 持续意图消融，改变原探索选择频率 |
| D | `v2_d_train/smoke` | 在 C-persistent 上启用几何路径缓存 |

完整文件名前缀为 `cfg/eval_goatbench_hgr_dual_topo_`。
`v2_baseline_train/smoke` 使用同模型与当前共同基础设施、关闭 ActiveTopo 行为；
报告应称为当前 checkout 的 HGR 对照，不冒称未修改上游源码。

计时输出在 subtask_metrics 的 timing 中，阶段互斥，model_request 嵌套时间从外层扣除。
阶段名为 perception、mapping、memory_update、candidate_preparation、route_execution、feedback、other。
这些是管线区间耗时，perception 仍包含部分深度融合；不能把它称为纯模型/GPU 时间。
几何子计时仅作诊断，不能与外层阶段重复求和。

## 3. 先执行离线检查，再运行阶段 smoke

推荐使用统一入口。第二个参数是 GPU 编号；脚本固定 16 个 CPU 线程，
自动创建带时间戳的配置和独立结果目录，并在 A/B/C/D 后运行相应的路线重放；
B/C/D 还会自动执行轨迹契约审计。可分别运行，也可用 `all` 顺序运行全部阶段；
`all` 遇到任何非零退出立即停止。自动检查不能替代各阶段结果的人工性能和失败轨迹复核。

```bash
cd /home/hdd/tangyuxin/projects/Hypothesis_Graph_Refinement
bash scripts/run_hgr_v2_stage.sh tests 7
bash scripts/run_hgr_v2_stage.sh all 7

# 或者分别运行：
bash scripts/run_hgr_v2_stage.sh a 7
# A-R2 结果验收后：
bash scripts/run_hgr_v2_stage.sh b 7
bash scripts/run_hgr_v2_stage.sh c-shadow 7
bash scripts/run_hgr_v2_stage.sh c 7
bash scripts/run_hgr_v2_stage.sh c-persistent 7
bash scripts/run_hgr_v2_stage.sh d 7
```

若 `all` 已完成 A-R2 后在审计处停止，修复/复核 A 后可从 B 一键继续，避免重复产生模型调用：

```bash
bash scripts/run_hgr_v2_stage.sh after-a 7
```

可选 `baseline` 用同一 smoke 清单运行当前 checkout 的 HGR 对照：

```bash
bash scripts/run_hgr_v2_stage.sh baseline 7
```

若已激活 hgr conda 环境，脚本使用该环境的 Python；否则使用项目文档中的 hgr Python。
也可用 `HGR_PYTHON=/实际路径/python` 覆盖。正常完成后终端最后会打印结果目录，
把该路径发回即可继续复核。

以下命令由用户执行。GPU 0 仅为示例，换成空闲卡。

```bash
cd /home/hdd/tangyuxin/projects/Hypothesis_Graph_Refinement
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 NUMEXPR_NUM_THREADS=16
/home/tangyuxin/miniconda3/envs/hgr/bin/python -m unittest tests.test_hgr_dual_topo_v2 -v
/home/tangyuxin/miniconda3/envs/hgr/bin/python -m unittest discover -s tests -v
```

随后生成**单阶段**运行脚本；生成工具不启动导航、不请求模型。首次先跑 A-R2：

```bash
run_dir="cfg/generated/v2_a_r2_$(date +%Y%m%d_%H%M%S)"
/home/tangyuxin/miniconda3/envs/hgr/bin/python scripts/hgr_v2_experiments.py commands \
  --suite smoke --methods a --repeats 1 --gpu 0 \
  --manifest cfg/manifests/goat_train_v7_1_d_smoke_aczz_ep0_seed77.json \
  --output "$run_dir"
bash "$run_dir/run.sh"
```

A-R2 通过后，用新的 run_dir，依次将 `--methods a` 改为 b、c_shadow、c、c_persistent、d。
不要因为脚本成功退出就跳过 trace 审计，也不要一次把所有阶段无条件串行执行。
结构错误先修复，再扩大；输出目录、指纹和决策快照拒绝覆盖。

新模式在线检查：v2_route_selected、v2_execution、v2_motion_acknowledged、
v2_intent_installed、v2_source_events、v2_hypothesis_retraction、v2_goal_end。
检查实体/图片/终端一致，游标只在 acknowledge 后推进，terminal_arrived 不出现在中间 transition，
未认证路线无实际运动，撤销不删实测空间。记录所有失败和恢复。

重放某个新模式结果中的已保存路径：

```bash
/home/tangyuxin/miniconda3/envs/hgr/bin/python scripts/hgr_v2_experiments.py replay \
  results/实际运行名称/decision_records
```

重放只回答当时已知空间中的路径可行性与成本对账，不替代闭环策略评测。
快照保留当时 known-free mask、Place 图、当前位姿、候选、认证 RoutePlan；不借用未来地图。
smoke 默认保存决策快照，train/holdout 默认关闭网格快照以避免压缩 I/O 干扰性能比较；
需要同输入诊断时，在独立运行中启用 `hgr_dual_topo.record_decisions`。

检查新模式运动是否遵守记录的认证路径及终端顺序：

```bash
/home/tangyuxin/miniconda3/envs/hgr/bin/python scripts/hgr_v2_experiments.py audit \
  results/实际运行名称/active_topology_traces --output results/实际运行名称/v2_trace_audit.json
```

该审计拒绝没有运动事件的空结果，也不将路线记录本身当作实时几何正确性的证明。
状态来源和几何还需要上述重放、结构测试及在线诊断联合检查。

新增测试还覆盖：已经激活的意图接收新独立证据、对象终端被占据后的同目标更新、
frontier 已知侧终端、RoutePlan 成本/终端契约拒绝、检查点坐标不匹配、
两组同时缺少任务时的完整分母检查。测试是否通过以用户运行输出为准。

## 4. 开发集、独立场景与冷暖地图

阶段通过后生成开发集对照，先每项一次：

```bash
/home/tangyuxin/miniconda3/envs/hgr/bin/python scripts/hgr_v2_experiments.py commands \
  --suite development --repeats 1 --gpu 0 \
  --manifest cfg/manifests/goat_train_dev_12x2_seed77.json \
  --output cfg/generated/v2_development_唯一名称
```

冻结方法后生成新的 12×2 holdout。`--history` 必须列出实际存放历史结果的所有目录，
包括其他 baseline checkout；默认还排除本项目 cfg/manifests 下所有已有场景。
额外历史清单使用重复的 `--exclude-manifest`。旧无日志实验的场景需额外提供排除清单，
工具不能证明未记录的实验从未发生。

```bash
/home/tangyuxin/miniconda3/envs/hgr/bin/python scripts/hgr_v2_experiments.py holdout \
  --history results \
  --output cfg/manifests/goat_train_v2_holdout_12x2_seed77.json
/home/tangyuxin/miniconda3/envs/hgr/bin/python scripts/hgr_v2_experiments.py commands \
  --suite holdout --repeats 3 --gpu 0 \
  --manifest cfg/manifests/goat_train_v2_holdout_12x2_seed77.json \
  --output cfg/generated/v2_holdout_唯一名称
```

冷暖地图先由 `cfg/eval_goatbench_hgr_dual_topo_v2_memory_export.yaml` 运行原选择器，
旁路构建 Place 图。在每 episode 第一子任务结束后保存共同前缀检查点，停止该 episode。
先使用继承此配置的 smoke 清单验收序列化，再在冻结清单运行 split 1 和 2。
前缀不足两任务的 episode 不适用于默认一任务前缀，不能静默缺少检查点后比较。

```bash
/home/tangyuxin/miniconda3/envs/hgr/bin/python run_goatbench_evaluation.py \
  -cf cfg/eval_goatbench_hgr_dual_topo_v2_memory_export.yaml --split 1
/home/tangyuxin/miniconda3/envs/hgr/bin/python run_goatbench_evaluation.py \
  -cf cfg/eval_goatbench_hgr_dual_topo_v2_memory_export.yaml --split 2
/home/tangyuxin/miniconda3/envs/hgr/bin/python scripts/hgr_v2_experiments.py commands \
  --suite memory --repeats 3 --gpu 0 \
  --manifest cfg/manifests/goat_train_dev_12x2_seed77.json \
  --checkpoints results/hgr_v2_memory_checkpoints \
  --output cfg/generated/v2_memory_唯一名称
```

四组为 baseline/V2 × cold/warm：都从检查点实际位姿、同一首个评测子任务和后续序列开始。
warm 恢复相同 HGR 对象、图像、TSDF、假设与随机状态；V2 额外使用同轨迹的 Place 图。
cold 在同位置初始化空记忆。后续任务起点允许随闭环自然分化，不能逐任务假定仍配对同起点。
memory_*.json 保存共同前缀成本。完整冻结验收时须改用同一 holdout 清单导出新前缀。
检查点是本 checkout 产生的 pickle，仅加载自己生成且可信的文件；不是跨数据集地图格式。
四组交互报告要求检查点 SHA-256、首个评测子任务和起点完全一致；只有路径相同而内容不同的
检查点也会被拒绝。默认三次重复共用冻结的合法探索前缀，API 波动单独由重复运行体现。

## 5. 配对报告与完成标准

`compare` 要求两组的全部子任务与指定 manifest 相符。每个参数表示一次重复，
同一次重复的两个 split 目录用逗号连接；不同重复使用空格分隔。

```bash
/home/tangyuxin/miniconda3/envs/hgr/bin/python scripts/hgr_v2_experiments.py compare \
  --manifest cfg/manifests/goat_train_v2_holdout_12x2_seed77.json \
  --baseline 'results/基线r1s1,results/基线r1s2' 'results/基线r2s1,results/基线r2s2' 'results/基线r3s1,results/基线r3s2' \
  --candidate 'results/V2r1s1,results/V2r1s2' 'results/V2r2s1,results/V2r2s2' 'results/V2r3s1,results/V2r3s2' \
  --output results/v2_holdout_comparison.json
```

冷暖交互分析中，上述 baseline/candidate 传 warm 两组，额外传 `--baseline-cold` 与
`--candidate-cold`，并设置 `--start-subtask 1`。报告
`(V2 warm − baseline warm) − (V2 cold − baseline cold)` 的 scene bootstrap 区间。
不能仅凭 warm 比 cold 快就声称 topo 有独立收益。

最终主配置门槛：结构测试与在线审计通过；SR 差值 95% CI 下界 ≥ −0.02；平均 SPL 改善；
路径或总耗时差值至少一项 95% CI 上界 < 0。按 scene 重采样 10,000 次，失败不剔除，
三次重复不挑最佳。数值门槛脚本计算，结构通过必须另行审计。
未达标如实交付结果并保留实验状态，不通过修改 holdout 或停止协议追逐分数。
