# HGR 自适应融合版实现与评测说明

日期：2026-09-16。

## 一、为什么增加融合版

C-lite 表明各机制各有有效部分，但把它们全程打开会明显增加成本：

- Route-only 的 GOAT SR/SPL 为 65.31%/50.78%，是当前正式基线。
- Memory-dedup 的 GOAT SR/SPL 为 67.35%/52.85%，说明跨 goal 记忆可能有益，
  但提升区间跨零且运行成本上升。
- Intent-persistent 减少了语义重选，却因长期锁定路线增加到 444 step。
- Verified-ablation 的 GOAT SR 点估计最高，但需要 99 次终点验证，SPL 降至
  34.38%，并存在 5 次距离口径错误确认。
- 当前 Memory-cost-gate 同时降低成功率并增加模型调用，应停用。

因此不能直接把最后一个累积配置称为最优版本。本次实现一个独立的
`adaptive-hybrid`，只保留每项机制中有证据支持的部分，并用同一开发清单与两个
对照配置重新筛选。

## 二、最终策略

融合版以 Route-only 的已知空间拓扑执行为基础，增加以下行为：

1. 保留全局 scene map、goal-scoped 状态、跨 goal 证据和 REVISIT 去重。
2. 关闭已失败的 revisit cost gate。
3. Snapshot/Object intent 继续执行到路线完成；Frontier intent 最多连续执行 3 个
   实际运动步，随后释放并让原版 HGR 用新观测重新选择。
4. 只对“跨 goal 历史来源 + image/description goal + REVISIT”执行终点验证。
5. 当前 goal 的新鲜 Snapshot、object goal 和普通 APPROACH 沿用原版 HGR 到达停止。
6. 历史目标第一次 `confirmed` 即允许停止；`uncertain` 最多尝试一个额外视角；
   `rejected`、`error` 或无法得到新视角时释放任务并重选。

选择性验证由任务字段 `requires_verification` 绑定到具体 source-bound intent，runner
不能依据全局开关误把所有 Snapshot 都送去验证。这个策略只读在线观测和记忆来源
年龄，不读取 GOAT 真值距离或成功标签。

## 三、代码与配置

- `cfg/eval_goatbench_hgr_dual_topo_adaptive_hybrid.yaml`：融合版冻结配置。
- `src/dual_dynamic_navigation/unified_navigator.py`：选择性验证、受限 Frontier intent
  和任务级验证策略。
- `src/dual_dynamic_navigation/task_planner.py`：任务验证标记和 intent 实际运动步计数。
- `run_goatbench_evaluation.py`：按具体任务决定是否进入验证分支。
- `src/goatbench_results.py`：汇总选择性验证任务数和受限 intent 到期次数。
- `scripts/run_hgr_stages.py`：增加 `adaptive-screen` 一键对照评测。

评测输出新增：

- `hypothesis_aware_navigation.selective_verification_tasks`
- `hypothesis_aware_navigation.bounded_intent_expirations`

它们分别证明验证是否只在目标子集触发，以及 Frontier intent 是否真的受到三步限制。

## 四、运行方式

先只运行融合候选，快速确认在线链路：

```bash
cd /home/hdd/tangyuxin/projects/Hypothesis_Graph_Refinement
export DASHSCOPE_API_KEY='你的 API Key'
python scripts/run_hgr_stages.py \
  --stage adaptive-hybrid \
  --gpu 1 \
  --threads 16 \
  --skip-offline
```

候选无运行错误后，运行同源对照。该命令顺序执行 Route-only、Memory-dedup 和
Adaptive-hybrid，每项使用同一 smoke manifest 的全部 episode：

```bash
python scripts/run_hgr_stages.py \
  --stage adaptive-screen \
  --gpu 1 \
  --threads 16 \
  --split all \
  --matrix-id adaptive_hybrid_v1 \
  --resume \
  --skip-offline
```

加入 `--dry-run` 只打印命令。固定 `--matrix-id` 后重复运行会检查源码、配置、
manifest 和完成标记，只恢复真正匹配的完整结果。

## 五、候选门槛

融合版当前是待评测候选，不能因代码完成直接称为最优版本。筛选时使用 GOAT SR
作为任务成功主指标、GOAT SPL 作为效率主指标；Snapshot SR 仅用于实体映射诊断。

进入独立最终评测至少应满足：

- 没有未分类错误，已知空间运动审计通过。
- GOAT SR 不低于 Route-only，并且 GOAT SPL 不出现有支持的下降。
- VLM 调用、step 和路程不能重现全量 Verified/Intent 的大幅增长。
- 选择性验证只出现在跨 goal 的 image/description REVISIT。
- 若点估计改善但 scene-block 区间跨零，结论仍为尚不确定。

通过筛选后，最终批次可使用：

```bash
python scripts/run_hgr_experiment_plan.py \
  --phase final \
  --final-candidate adaptive-hybrid \
  --gpu 1 \
  --threads 16
```

未通过时继续保留 Route-only 为正式版本。

## 六、首次在线 smoke 状态

`adaptive_hybrid_v1` 的单 scene、8 goal 对照已经完成。Adaptive-hybrid 获得
87.50% GOAT SR 和 46.23% GOAT SPL；Route-only 分别为 75.00% 和 44.83%。
Adaptive 同时将 VLM 调用从 114 降至 79。三个配置均 8/8 completed，无在线错误。

该结果通过功能性 smoke，但样本不足以冻结最终候选。机制和逐项结果见
`docs/HGR_ADAPTIVE_HYBRID_SMOKE_RESULTS.md`；下一步为 Adaptive-only 的 3×2 开发
清单验证。

## 七、3×2 开发集状态

3 scene、6 episode、49 goal 已完成。Adaptive-hybrid 的 GOAT SR/SPL 为
77.55%/61.05%，Route-only 为 65.31%/50.78%；Adaptive 的总 VLM 调用为 387，
Route-only 为 428。Adaptive 有 45 completed 和 4 exhausted，无运行错误。

当前实现已经冻结为独立最终集候选，不继续基于这 3 个 scene 调参。配对区间、验证
精度和异常结束审计见 `docs/HGR_ADAPTIVE_HYBRID_3X2_RESULTS.md`。
