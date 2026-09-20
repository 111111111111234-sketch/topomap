# HGR 双层拓扑性能迭代：开发实施记录

日期：2026-09-14。

状态：首轮代码开发和离线回归完成；没有在本次开发中启动 simulator，也没有发起模型请求。在线性能尚未验证，不能据此声明 SR/SPL 已提高。

对应计划：[HGR_DUAL_TOPO_PERFORMANCE_ITERATION_PLAN.md](HGR_DUAL_TOPO_PERFORMANCE_ITERATION_PLAN.md)。

## 1. 本轮结论

旧批次 `20260913T123622024858Z` 中，Memory 和 Intent 的第一个 goal 不是因为路径规划提前到达而结束。两者都在 step 0 选择 `0-view_0.png` 的实体 `7`，生成从 `[18, 17]` 到 `[21, 13]` 的有效已知空间路径，路径长约 `0.524 m`；执行约 `0.5 m` 后到达该终端并按原版 HGR 停止协议结束。评测真值显示 Snapshot 与 Distance 均失败，最终目标距离约 `5.486 m`。

因此该样本的直接问题是语义选择了错误对象。统一链路在模型请求前把候选裁剪为“当前可执行子集”，并在提示中加入 APPROACH/REVISIT 用途，这改变了原版 HGR 看到的输入。它不是中间 topo waypoint 被误当作最终目标，也不是已知空间路径证书错误。

本轮增加 `original_hgr` 选择模式：模型看到与原版 HGR 相同的 Snapshot、对象、Frontier 和提示；HGR 返回具体来源后，统一 navigator 才绑定 Snapshot、实体、证据和终端，并检查已知空间路径。选中来源不可执行时返回结构化 `NoAction`，不改选其他对象，也不使用未评分几何候选替代模型结果。

Memory/Intent 继续保留原版停止协议。只有 Verified 可以在同一实体的新鲜图像验证为 `confirmed` 后停止。

## 2. 已完成的代码

### M0：结束状态和错误诊断

- 新增 `SubtaskExecution`，使用首次结束原因锁定。
- 每个新子任务记录 `execution_status`、`termination_reason`、`termination_detail` 和 `error_counts`。
- 区分无可执行候选、语义请求/解析失败、导航设置失败、执行失败、步数耗尽、原版成功停止和验证确认。
- `execution_quality.json` 汇总完成、耗尽、错误、旧数据缺失和各类错误计数。存在未知结束原因或 error 状态时，`acceptance_ready` 为 false。
- `navigation_diagnostics` 只写有界摘要，包含实际 backend、决策来源、机制开关、候选淘汰原因、L1/L2 图规模和当前 intent，避免把完整导航状态塞进每条指标。

### M1：选择与到达契约

- 新增 `semantic_selection_mode`：
  - `executable_subset` 保留首版 Memory/Intent/Verified 的候选预过滤行为。
  - `original_hgr` 保留原版请求内容，再对选定来源执行已知空间授权。
- `original_hgr` 模式关闭任务用途提示和未评分 Frontier fallback，固定输入测试确认发送给模型的请求 payload 与原版 selector 一致。
- 不可执行的已选对象与 Frontier 分别记录 `selected_source_not_executable` 和 `selected_frontier_not_executable`。
- SemanticCritic 输入张量迁移到模型实际 device/dtype；特征计算异常计入 `feature_residual_errors`。CUDA 测试在有 GPU 时检查真实设备一致性，无 GPU 时明确跳过。

### M2：跨 goal 的全局事实

- SceneMap 继续在 episode 内跨 goal 保留 Place、边、对象、alias 和证据。
- 每条对象证据新增首次出现 `(goal, step)`、最近出现 `(goal, step)` 和最近观察位置。
- alias 合并时迁移这些证据字段。
- GoalGraph、checked views、suppression、增量去重和 intent 在 `begin_goal` 时重建，旧 goal 的拒绝结论不会污染新 goal。

这些能力仍限定在同一 episode。跨 episode 或进程重启的长期记忆不在本轮范围内。

### M3：独立机制开关

以下开关写入有效配置和运行指纹：

| 开关 | 作用 |
| --- | --- |
| `semantic_selection_mode` | 原版候选呈现或首版可执行子集呈现 |
| `enable_candidate_annotations` | 是否给 HGR 增加 APPROACH/REVISIT 用途提示 |
| `enable_revisit_deduplication` | 当前 goal 内排除已检查物理视角 |
| `enable_revisit_cost_gate` | 在选择后应用剩余预算与探索机会成本门控 |
| `enable_persistent_intent` | 路线仍有效时持续执行同一 intent |
| `enable_incremental_preemption` | 新证据经同实体确认后允许抢占 |
| `enable_terminal_verification` | 到达对象终端后执行 Verified 协议 |
| `enable_unscored_frontier_fallback` | 模型选择失败时是否采用无语义评分 Frontier fallback |

持续 intent 新增停滞检测：只有发生位置移动且同一路线的已知路径距离连续 3 次没有减少一个体素时才触发恢复；无位移观察不计数。恢复预算仍是一次局部修复加一次同终端重规划，同一 intent 不逐帧重置。

请求或验证错误不会写入语义负证据，也不会永久 suppression 实体。任务数、各任务授权路程、增量评估、成功抢占、不可执行来源和停滞次数进入 navigator 统计。

### M4：Verified 质量口径

- 验证 trace 增加 scene、episode 和 subtask 身份，可与评测真值关联。
- 汇总输出验证次数、verdict、后续 action、确认 subtask、确认后的 Snapshot/Distance 成功率，以及按 Snapshot 口径的错误确认数。
- 验证请求错误单独计入 `verification_request`，不生成 rejected 证据。
- 重复反馈和旧 intent 反馈继续幂等忽略；第二视角由当前 goal 的 checked-view 距离约束选择。

## 3. 消融配置

旧的七阶段配置不改含义。新增配置按下列顺序每次只开启一个行为：

| 阶段名 | 配置 | 相对上一步的变化 |
| --- | --- | --- |
| `memory-semantic-first` | `eval_goatbench_hgr_dual_topo_memory_semantic_first.yaml` | 原版 HGR 请求，选择后来源绑定与已知空间授权 |
| `memory-dedup` | `eval_goatbench_hgr_dual_topo_memory_dedup.yaml` | 开启当前 goal 视角去重 |
| `memory-cost-gate` | `eval_goatbench_hgr_dual_topo_memory_cost_gate.yaml` | 开启重访成本与剩余预算门控 |
| `intent-persistent` | `eval_goatbench_hgr_dual_topo_intent_persistent.yaml` | 开启持续 intent |
| `intent-preempt` | `eval_goatbench_hgr_dual_topo_intent_preempt.yaml` | 开启增量抢占 |
| `verified-ablation` | `eval_goatbench_hgr_dual_topo_verified_ablation.yaml` | 从当前保留的 persistent 候选开启终点验证，不启用已淘汰的增量抢占 |

## 4. 测试结果

执行命令：

```bash
cd /home/hdd/tangyuxin/projects/Hypothesis_Graph_Refinement
MPLCONFIGDIR=/tmp/hgr-matplotlib \
YOLO_CONFIG_DIR=/tmp/hgr-ultralytics \
/home/tangyuxin/miniconda3/envs/hgr/bin/python -m unittest discover -s tests -q
```

最新结果：`Ran 376 tests`，`OK (skipped=1)`。跳过项是当前环境不可用的 CUDA device/dtype 测试；CPU dtype、NumPy/历史 RGB 特征兼容、异常计数、选择 payload 等价、门控后有界重选、选择后授权、跨 goal 证据消费、alias、机制配置、停滞阈值、Verified 幂等、真值汇总、短诊断限制及实验工具均已通过。

## 5. 推荐在线执行顺序

现有三 goal 定向 smoke 作为 A 的结构验收，不再补跑 8 goal。完整 B 被取消：
旧 Baseline 只作开发参考，Shadow 使用已有 smoke 和离线等价性测试，Route-only
并入 C-lite。默认命令只运行 C-lite 的 5 个配置乘 6 个 episode：

```bash
cd /home/hdd/tangyuxin/projects/Hypothesis_Graph_Refinement
export DASHSCOPE_API_KEY='你的 API Key'
python scripts/run_hgr_experiment_plan.py --gpu 4 --threads 16
```

顺序为 Route-only、Memory-dedup、Memory-cost-gate、Intent-persistent 和
Verified-ablation。已确认低效的 semantic-first 与 intent-preempt 不再重复运行。
每项结束后自动检查执行质量和运动证书并生成 JSON/Markdown 对比报告。

D 不会自动衔接。审阅 C-lite 报告、冻结候选且不再调参后，才单独执行：

```bash
python scripts/run_hgr_experiment_plan.py \
  --phase final \
  --final-candidate verified-ablation \
  --gpu 4 \
  --threads 16
```

下面的单阶段命令保留为历史诊断入口，不再是当前推荐顺序。

先只跑修复后的语义优先 Memory smoke：

```bash
cd /home/hdd/tangyuxin/projects/Hypothesis_Graph_Refinement
python scripts/run_hgr_stages.py --stage memory-semantic-first --gpu 4 --threads 16 --skip-offline
```

检查新结果中的 `execution_quality.json`，要求 8 个子任务都存在明确结束原因。随后审计已知空间运动：

```bash
python scripts/hgr_rebuild.py audit results/<memory-semantic-first运行目录>
```

如果该 smoke 无未分类结束和实现异常，再串行运行全部六项消融：

```bash
python scripts/run_hgr_stages.py --stage ablations --gpu 4 --threads 16 --skip-offline
```

仅打印命令、不执行：

```bash
python scripts/hgr_rebuild.py commands --stage ablations --gpu 4
python scripts/run_hgr_stages.py --stage ablations --gpu 4 --threads 16 --dry-run --skip-offline
```

结果对比示例：

```bash
python scripts/summarize_goatbench_results.py \
  results/<route-only目录> --label route-only \
  --compare memory-semantic-first=results/<memory-semantic-first目录> \
  --compare verified-ablation=results/<verified-ablation目录> \
  --output artifacts/hgr_rebuild/ablation_summary.json \
  --markdown-output artifacts/hgr_rebuild/ablation_summary.md
```

smoke 通过后，下面的命令按 stage、repeat、split 串行展开两个分片。相同 stage/repeat 的两个 split 写入同一结果目录，便于聚合：

```bash
python scripts/run_hgr_stages.py \
  --stage intent-persistent \
  --manifest cfg/manifests/goat_train_dev_12x2_seed77.json \
  --split all \
  --repeats 3 \
  --matrix-id dev12x2_v1 \
  --gpu 4 \
  --threads 16 \
  --skip-offline
```

### 2026-09-16 REVISIT 修复后完整 C-lite 结论

四项受影响配置已在 3×2 开发清单上完整结束。Memory-dedup、
Memory-cost-gate、Intent-persistent、Verified-ablation 的 GOAT SR 分别为
67.35%、57.14%、61.22%、71.43%，Route-only 为 65.31%。Verified 虽有最高
SR 点估计，但 GOAT SPL 为 34.38%，相对 Route-only 的 scene-block 95% 区间为
[-39.95, -3.99] pp，且产生 8 个 exhausted、1 个 selection error、99 次终点验证
和 697 次总模型调用。Memory-dedup 与 Route-only 的 SR/SPL 差异区间均跨零，
同时成本更高。

据预设门槛，本轮没有新机制可靠优于 Route-only。停止链路修复保留；
Memory-dedup 仅保留为研究对照，成本门控、持续 Intent 和当前 Verified 消融淘汰。
完整分析见 `docs/HGR_REVISIT_STOP_FIX_C_LITE_RESULTS.md`。

`verified_quick3_v2` 在线完成后，严格 GOAT SR 为 66.67%、GOAT SPL 为 18.22；3 个 goal 均以 `verified_confirmation` 结束，26 条运动无违规，无内部错误。历史 sink 的停止顺序已变为第一视角 `matched/new_view`、第二视角 `matched/stop`，证明两视角授权生效；但该 sink 仍距评测目标视点 6.63 米，说明多视角只能确认类别与物理实体一致，无法从纯类别指令区分数据集未提供的特定实例身份。

Snapshot 审计将 3 个 goal 分为 1 个 `matched`、1 个 `selected_id_mismatch`和 1 个 `mapping_unavailable`。汇总不再把 `mapping_unavailable` 当作 Verified 错误确认；本次报告 1 次可评估的 Snapshot 错误确认和 1 次确认后映射不可用。

同指纹时期的 `persistent_quick3_v1` 对照中，3 个 goal 均以 `no_executable_candidate` 耗尽，GOAT SR/SPL 为 0；它虽有两个 goal 最终位置在阈值内，但没有取得停止授权。Verified 在同三 goal 上 GOAT SR 提高 66.67 个百分点，step 从 29 降至 26，VLM 调用从 56 降至 50，因此保留 Verified，淘汰无终点确认的 persistent 作为最终候选。该结论仍是单场景开发诊断，下一步进入 8-goal、25-step 上限的候选 smoke。

### 2026-09-15 intent-preempt 短诊断与决策

`exp_hgr_rebuild_intent_preempt_r01_intent_preempt_quick_v1` 完成相同首 goal、25 step 上限的短诊断。结构验收通过：无内部错误，21 条已知空间运动无违规，完成 1 次 REVISIT，重复证据选择为 0。

与 `intent_fix_quick_v2` 的 `intent-persistent` 做同 subtask 对照，Snapshot SR 都为 0%，Distance SR 都为 100%；Distance SPL 从 61.17 降至 40.77，step 从 18 增至 22，路程从 8.74 米增至 13.11 米，VLM 逻辑调用从 35 增至 74，累计响应时间从 45.44 秒增至 98.76 秒。该运行执行了 11 次增量评估，但没有任何一次通过确认并形成有效抢占。

因此当前淘汰 `intent-preempt`，保留 `intent-persistent` 作为 Intent 候选。`verified-ablation` 改为直接继承 `intent-persistent`，下一步只验证终点确认机制的增量影响。这两次都是单 goal 开发诊断，不构成总体性能结论。

### 2026-09-15 verified-ablation 单 goal 短诊断

`exp_hgr_rebuild_verified_ablation_r01_verified_quick_v1` 完成。它在第 16 step 对实体 247 执行 1 次新鲜观测验证，模型返回 `matched`，系统以 `verified_confirmation` 正常结束。运行无内部错误，16 条已知空间运动无违规，证明 Verified 的选择、APPROACH、新鲜观测、验证与停止链路已经实际执行。

与同一首 goal 的 persistent 对照相比，Snapshot SR 均为 0%，Distance SR 均为 100%；Verified 的 step 从 18 降至 16，但路程从 8.74 米增至 9.36 米，VLM 调用从 35 增至 38，Distance SPL 从 61.17 降至 57.09。该确认在 Distance 真值口径上成功，最终距离仅 0.097 米；在 Snapshot 对象 ID 口径上失败，因此报告为 1 次 `false_confirmation_count_by_snapshot`。这个冲突既可能是相邻同类实体的误确认，也可能是 Scene 对象合并/ID 映射与距离真值不一致；单个样本不足以修改停止协议。

当前将 Verified 保留为待验证候选，不直接进入耗时约三小时的 8-goal smoke。下一步运行同场景前 3 个 goal、每 goal 最多 25 step，同时查看确认准确性、拒绝/不确定分支和跨 goal 历史证据复用。

### 2026-09-15 verified-ablation 三 goal 短诊断

`exp_hgr_rebuild_verified_ablation_r01_verified_quick3_v1` 完成同场景前 3 个 goal。运行产生 27 条已知空间运动且无违规，无内部错误；2 个 goal 以 `verified_confirmation` 结束，1 个耗尽 25-step 诊断预算。总体 Snapshot SR 33.33%、Distance SR 66.67%，Snapshot SPL 9.70、Distance SPL 22.76。

后续口径复核发现，上述 Distance SR 只表示最终位置在距离阈值内，没有要求有效 STOP；因此它把第一个预算耗尽的 goal 误计为任务成功。结果分析器已新增严格 `GOAT SR/GOAT SPL`：只有 `execution_status=completed` 且 Distance 成功才计入。本次严格 GOAT SR 为 33.33%，GOAT SPL 为 9.70；Snapshot 和 Distance 两组指标降为诊断口径。

全局 SceneMap 在第二个 goal 开始时保留 7 places、72 entities、153 条历史证据，第三个 goal 开始时保留 7 places、76 entities、167 条历史证据。第三个 goal 实际选择了第二个 goal 首次观测的实体 357，因此“同场景跨 goal 保留地图并在执行中复用历史证据”已有在线 trace 证据。

三个 goal 的 Verified 行为分别为：第一个 image goal 将错误的 toilet 正确拒绝，随后预算耗尽；第二个 image goal 确认 red towel，Snapshot 和 Distance 均成功；第三个 object goal 复用历史 sink 并确认，但 Snapshot 和 Distance 均失败，最终距离 6.24 米。因此 2 次 confirmed 中只有 1 次通过两种真值口径，当前不接受 Verified 为性能候选。

下一步先改进当前 goal 对历史实体的停止授权与目标 ID 对齐诊断，而不是继续运行完整 8-goal smoke。策略不能读取评测真值；真值只用于离线归因和报告。

### 2026-09-15 Snapshot 审计与历史停止授权修复

每个新 subtask 现在记录 GT object ID、逐帧 mask-IoU 映射投票、原始/解析 alias 后的目标 ID、最终选择 ID 和 alias 表。Snapshot 判定先解析 Scene 对象合并 alias，诊断明确分为 `matched`、`mapping_unavailable` 和 `selected_id_mismatch`；策略仍不读取这些真值字段。

Verified 对跨 goal 历史来源新增两视角停止授权。第一次 `matched` 只用于转入另一个已知空间可达视角，不允许 STOP；第二个物理独立视角再次 `matched` 后才确认。若没有第二视角，释放当前 intent 并继续选择，不使用旧 crop 直接宣告新 goal 成功。

严格 GOAT SR/SPL、Snapshot 映射审计和历史二次确认共通过 374 项离线测试，1 项 CUDA 测试因当前测试进程无可用 CUDA 而跳过。下一步用新指纹重跑同一 3-goal 诊断：

```bash
python scripts/run_hgr_stages.py \
  --stage verified-ablation \
  --gpu 4 \
  --threads 16 \
  --matrix-id verified_quick3_v2 \
  --max-subtasks 3 \
  --max-steps-per-subtask 25 \
  --resume \
  --skip-offline
```

短诊断 `exp_hgr_rebuild_intent_persistent_r01_intent_fix_quick_v2` 已完成。1 个 goal 实际运行 18 step 后以 `selected_source_not_executable` 耗尽；结构验收通过，critic 错误为 0，17 条已知空间运动无违规，1 次 REVISIT 完成并消费证据，重复证据选择为 0。该 goal 的 Distance SR 为 100%、Snapshot SR 为 0%，因此只证明两项实现修复生效，不证明语义选择正确。

与修复前完整运行中的同一个首 goal 对比：step 从 25 降至 18，路程从 12.15 米降至 8.74 米，VLM 调用从 47 降至 35，VLM 累计响应时间从 83.10 秒降至 45.44 秒，同时消除了 2 次 critic 错误。`intent-preempt` 已完成同限制筛查并被淘汰；接下来只运行 `verified-ablation` 短诊断，再决定是否进入完整 8-goal smoke。

中断后可原命令加 `--resume`。脚本只有在 `completed.json` 存在，且源码、配置继承链、manifest、split、repeat 与非空结果全部匹配时才跳过；部分结果会继续运行，指纹不一致会拒绝覆盖。`--dry-run` 只展开命令。开发集结果用于选择候选；最终性能结论仍需未参与调参的独立清单和三次重复。

## 6. 在线验收判断

第一优先级是结构正确：无 unknown 结束原因、无未审计运动、无 critic device 异常、无验证错误导致的负证据。随后比较相同 subtask 的 Snapshot SR、Distance SR/SPL、步数、路程和模型调用。

`memory-semantic-first` 用于检验首步错误选择是否来自候选裁剪和提示变化。单次 smoke 变好只能支持该故障假设，不能作为总体性能结论。若逐项开启某机制后成功率下降，保留上一项配置；若成功数相同，则只有路程或模型调用下降且另一项不增加时才视为明确效率改善。

### 2026-09-14 首次 semantic-first 在线诊断（未完成）

运行 `20260914T073451849235Z` 在 156 个导航 step 内已收到 329 个 DashScope 200 响应。前三个子任务分别约耗时 19、39、37 分钟；第二、第三个子任务各执行 50 次选择，其中同一 REVISIT 来源分别重复 48 次。第四个子任务前 6 次选择也已出现 3 次相同来源。

这证明无去重、无持续 intent 的 `memory-semantic-first` 对照会产生预期的重复选择开销。它是故障定位对照，不适合作为高效最终配置；取得这些证据后可以停止该运行，下一步单独运行 `memory-dedup`，随后运行 `intent-persistent` 检查重复访问与模型调用是否下降。脚本现在会在启动该诊断配置时打印慢运行提示。

### 2026-09-14 首次 persistent-intent 在线结果与修复

运行 `20260914T100107482110Z` 完整结束，耗时约 47 分钟。99 个导航 step 产生 126 个模型请求，HGR 主选择仅执行 26 次，说明持续 intent 已明显减少逐步重选；但 Snapshot SR、Distance SR/SPL 均为 0，不能接受为性能候选。

结构诊断显示 8 个子任务全部为 `exhausted`：1 个耗尽步数，另外 7 个在 HGR 选中 REVISIT 后被成本门控以 `revisit_opportunity_cost` 拒绝，并被 runner 直接当作 `no_executable_candidate` 结束。该行为是确定的授权恢复缺陷。

修复后，成本门控拒绝首次选择时不会立即结束 goal。navigator 会保留原 HGR 响应作为审计证据，再让 HGR 在剩余已授权对象和 Frontier 中执行一次有界重选；无剩余候选时才返回 NoAction，请求失败则记录 RequestError。新增回归覆盖“昂贵 REVISIT 被拒绝、随后选择可执行 Frontier”。

本次运行还暴露 9 次 SemanticCritic 特征错误：旧 Snapshot 特征是 NumPy 数组，而计算路径只接受 PyTorch Tensor。现在两种输入都转换到 CLIP 模型实际 device/dtype；执行质量验收也会在存在任何内部错误计数时返回 false。

### 2026-09-14 结果审计与实验矩阵补齐

结果分析器新增总步数、帧数、路程、VLM 逻辑调用、HTTP 尝试、耗时、请求用途和 goal 类型统计。统一链路另行统计来源种类、同一 subtask 重复来源、有界授权重选、停滞恢复、已知空间运动、SceneMap 在每个 goal 起止的规模，以及 Verified 与评测真值的对应关系。

新 trace 对每个被执行的 Snapshot 来源记录 `source_first_seen_goal` 和 `historical_source`。SceneMap 同时记录证据总数、当前 goal 开始前的证据数和涉及实体数。因此后续运行可以分别回答“地图是否保留旧证据”和“规划执行是否真正选用了旧证据”。旧 trace 没有来源年龄，报告会显示未知，并保留旧的 REVISIT 数作为兼容信息，不再把 REVISIT 自动解释为跨 goal 复用。

分析器支持 Markdown 输出和相同 subtask 配对差值。配对统计按 scene 做 block bootstrap 95% 区间，并报告逐场景胜、平、负，避免把同场景多 goal 当成相互独立样本。

运行器支持 `--split all`、`--repeats`、稳定 `--matrix-id` 和严格 `--resume`。每个 split 只在完整聚合结束后原子写入完成标记；重复启动同一部分结果时，相同指纹可以续跑，源码或有效运行身份变化会立即报错。

### 2026-09-14 修复后 persistent-intent smoke 审计

运行 `exp_hgr_rebuild_intent_persistent_r01_intent_fix_smoke_v1` 完整生成 8 个子任务和完成标记。已知空间运动审计通过：341 条运动事件、0 条违规。指标为 Snapshot SR 0%、Distance SR 50%、Distance SPL 9.11；总计 346 step、178.51 米、340 次 VLM 逻辑调用。8 个任务全部耗尽，其中 5 个 `no_executable_candidate`、3 个步数耗尽，因此该版本不能进入 `intent-preempt`。

本次 trace 确认两项新缺陷。第一，原版 HGR 的 `Frontier.feature` 保存 RGB 图像，而 critic 将其展平后直接与 512 维 CLIP embedding 比较，产生 7 次 `388800 vs 512` 维度错误。critic 现在区分已编码向量和图像证据，对历史 RGB 先执行同一 CLIP 编码。第二，REVISIT 到达只记录观察位置，没有消费证据摘要，导致相同历史 crop 可从不同终端反复调度；本次出现 100 次 REVISIT 和大量重复来源。现在 `complete_revisit` 会按当前 goal 同时记录观察位置并消费该实体的证据摘要，只有摘要发生变化才允许再次进入。

结果汇总新增“重复证据选择、已完成 REVISIT、已消费的不同证据数”，将同一 Snapshot 名称下的不同实体和证据与真正的重复调度区分开。上述修复完成后全量离线回归为 372 项通过；由于源码指纹变化，需要用新 `matrix-id` 重跑 persistent-intent smoke。

为避免每次局部修复都重跑约三小时，runner 新增 `--max-subtasks` 和 `--max-steps-per-subtask`。它们只用于真实链路短诊断，并写入有效配置、运行指纹、完成标记和运行摘要，不会与完整 smoke 混淆。当前两项修复先运行第一个 goal、最多 25 step；该运行通常只需要完整 smoke 的一小部分时间。短诊断要求 critic 错误为 0、相同证据完成一次后不再被选择，满足后才重新运行 8/8。

```bash
python scripts/run_hgr_stages.py \
  --stage intent-persistent \
  --gpu 1 \
  --threads 16 \
  --matrix-id intent_fix_quick_v2 \
  --max-subtasks 1 \
  --max-steps-per-subtask 25 \
  --resume \
  --skip-offline
```

### 2026-09-15 C-lite Memory 停止链路修复

C-lite 的配对结果证明此前的“消费历史证据后重新选择”处理破坏了停止契约：
Memory-dedup 和 Memory-cost-gate 大量到达 REVISIT 终端，但 runner 随即释放
intent 并重新选择，最终多以 `no_executable_candidate` 耗尽。Agent 已经接近目标
却没有合法停止，表现为 Distance SR 明显高于 GOAT SR。

非 Verified 链路现在在 REVISIT 到达时保留同一 HGR 选择、实体、证据摘要和
intent，把任务原子地从 `REVISIT` 提升为 `APPROACH`，随后落入原版 HGR 的
Snapshot 到达完成分支。到达前仍没有停止权限，未知空间约束和来源绑定均不变，
且没有读取真值距离。Verified 继续走独立的新鲜 RGB 验证分支，只有确认后停止。
旧的重复抑制只作用于尚未到达的候选调度，不再在已经完成导航后抑制整个实体并
丢弃停止机会。

该修改影响 Memory、Intent 的历史 REVISIT 到达结果及其后续选择轨迹，也影响
Verified 的代码指纹但不改变其验证分支。Baseline、Shadow、Known-space 和
Route-only 不经过这一分支，既有结果仍可作为同清单参考。修复后的 C-lite 使用
新 matrix id；先跑一个跨 goal 短诊断，再重跑 Memory-dedup、Memory-cost-gate、
Intent-persistent 和 Verified-ablation。Route-only 无需因本修复重跑。

一键重跑受影响的 C-lite 配置：

```bash
export DASHSCOPE_API_KEY='你的 API Key'
python scripts/run_hgr_stages.py \
  --stage revisit-fix \
  --gpu 4 \
  --threads 16 \
  --manifest cfg/manifests/goat_train_fast_3x2_seed77.json \
  --split all \
  --matrix-id revisit_stop_fix_v1 \
  --resume \
  --skip-offline
```
