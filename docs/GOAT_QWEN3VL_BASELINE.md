# GOAT-Bench Qwen3-VL Baseline（后续开发基准）

状态：**已完成并冻结**  
更新日期：2026-08-24

## 目的

本文固定 HGR 后续 ActiveTopo 开发所使用的 GOAT-Bench 基准、快速回归子集和正式评测协议。后续拓扑改动必须先在开发子集上验证，再在完整基准上报告结果；不得覆盖这里记录的 baseline 输出。

## 正式 baseline

| 项目 | 固定值 |
| --- | --- |
| 任务集 | GOAT-Bench `val_unseen`，split 1 |
| 覆盖范围 | 36 个 episode、278 个 sequential subtasks |
| 方法 | 未改动导航、TSDF、ConceptGraph、HypothesisGraph 与 SemanticCritic 逻辑的 HGR |
| VLM | `qwen3-vl-30b-a3b-instruct`（百炼官方 API） |
| 端点 | `https://dashscope.aliyuncs.com/compatible-mode/v1`（北京区） |
| Key 环境变量 | `DASHSCOPE_API_KEY` |
| 随机种子 | 77 |
| 配置 | `cfg/eval_goatbench_qwen3vl_dashscope.yaml` |
| 结果目录 | `results/exp_eval_goatbench_qwen3vl30b_dashscope/` |

该模型为 Qwen3-VL Instruct 版本，固定非思考模式。主探索、Hypothesis Predictor 和 SemanticCritic 统一使用同一模型与端点。

### 与原始 HGR 配置的关系

除 VLM 供应商/模型、认证端点和独立结果目录外，实验行为保持 `cfg/eval_goatbench.yaml` 的设定：

- `choose_every_step: true`；
- `prefiltering: true`，`top_k_categories: 10`；
- egocentric view、图片尺寸 `360 x 360`、导航参数和成功阈值不变；
- 保持原始逐图 prompt：**不使用图拼接**、不设图片数量上限；
- 保持严格解析：不读取 `reasoning_content`，不从解释文本中正则提取候选选择。

因此，模型空回复或不符合 `Snapshot i, Object j` / `Frontier i` 格式的回复，会按原始 HGR 路径判为无效，不做 Qwen/DeepInfra 专用补救。

## 已冻结结果

完整运行于 2026-08-24 完成，实际耗时约 **8 小时 26 分**。

| 指标 | 数值 |
| --- | ---: |
| Snapshot SR | 27.34% |
| Distance SR | 48.92% |
| Snapshot SPL | 21.89% |
| Distance SPL | 33.43% |
| Image SR | 47.73%（88 题） |
| Description SR | 38.46%（91 题） |
| Object SR | 59.60%（99 题） |
| 平均筛选 snapshot 数 | 4.88 |
| 平均总 snapshot 数 | 15.39 |
| 平均总帧数 | 68.95 |

指标原始记录在结果目录下的 `success_by_*.pkl`、`spl_by_*.pkl`、`n_total_*.json`。SPL 以 pickle 内有效数值计算；运行日志中存在历史汇总的 `nan` 显示，不应直接用该显示值覆盖本表。

## 开发回归集：dev-10

完整 278 题用于正式结论，但每次拓扑实现改动都完整重跑成本过高。开发阶段固定使用下列 10 个 subtask 作为 **dev-10**：

```text
00832-qyAac8rV8Zk_0_0 ... 00832-qyAac8rV8Zk_0_4
00800-TEEsavR23oF_0_0 ... 00800-TEEsavR23oF_0_4
```

它对应 seed 77 下 split 1 的前两个场景；执行时使用 `--start_ratio 0 --end_ratio 0.06 --split 1`。该子集上的冻结 Qwen3-VL 基线为：

| 指标 | dev-10 baseline |
| --- | ---: |
| Snapshot SR | 20.00% |
| Distance SR | 70.00% |
| Snapshot SPL | 20.00% |
| Distance SPL | 51.43% |

dev-10 仅用于定位回归和快速消融，**不可**作为论文或正式结果。

## 后续 ActiveTopo 评测规则

1. 新实现先建立独立配置和结果目录，例如 `cfg/eval_goatbench_activetopo_qwen3vl_dashscope.yaml`、`results/exp_eval_goatbench_activetopo_qwen3vl_dashscope/`。
2. 除 ActiveTopo 开关及其参数外，保持本文件的 VLM、端点、seed、数据、split、GPU 条件和成功阈值不变。
3. 先运行 dev-10；报告四项 SR/SPL、平均 VLM 调用数、运行时间、frontier ID 稳定率与 blocked-target 重试率。
4. dev-10 无明显退化后，运行完整 36 episode / 278 subtask。
5. 正式报告必须将完整 ActiveTopo 输出与本文件的完整 baseline 对比；不得把 dev-10、A-EQA 41 题或 DeepInfra Qwen3.5 的拼图适配运行混入比较。

## 运行命令

完整 baseline 的复跑命令：

```bash
export DASHSCOPE_API_KEY='...'
CUDA_VISIBLE_DEVICES=5 /home/tangyuxin/miniconda3/envs/hgr/bin/python \
  run_goatbench_evaluation.py \
  -cf cfg/eval_goatbench_qwen3vl_dashscope.yaml
```

复跑 dev-10 时必须使用**不同的 `exp_name`/结果目录**，避免程序因为已有完整结果而跳过任务：

```bash
CUDA_VISIBLE_DEVICES=5 /home/tangyuxin/miniconda3/envs/hgr/bin/python \
  run_goatbench_evaluation.py \
  -cf <独立_dev10_配置>.yaml \
  --start_ratio 0 --end_ratio 0.06 --split 1
```

不要将真实 API Key 写入配置、脚本、日志或文档。
