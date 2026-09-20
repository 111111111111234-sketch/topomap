# A-EQA 41 题本地复现结果

更新日期：2026-08-23

## 这 41 道题是什么

本次运行的输入是仓库自带的 `data/aeqa_questions-41.json`。它包含 **41 道 A-EQA 问题、5 个 HM3D val 场景**，每道题记录了：问题文本、人工参考答案、初始位姿、场景 ID，以及部分题目的可接受替代答案。

它是开源仓库默认配置 `cfg/eval_aeqa.yaml` 指定的小规模评测集，不是论文的完整 A-EQA 评测集。论文的表 5 使用的是 184 题子集；完整 A-EQA 数据集则有 557 题。

## 当前运行配置

- 方法：HGR（Hypothesis Graph Refinement）默认设置
- VLM：`openai/gpt-4o`，通过 Requesty 的 OpenAI 兼容接口调用
- 随机种子：77
- 最大探索步数：50（未改动）
- 场景数据：本地 HM3D val
- 主配置：`cfg/eval_aeqa.yaml`

## 结果在哪里

| 文件 | 内容 | 是否作为正式结果使用 |
| --- | --- | --- |
| `results/exp_eval_aeqa/gpt_answer_merged_unique.json` | 38 条去重后的有效预测答案 | 是 |
| `results/exp_eval_aeqa/aggregation_summary.md` | 41 题运行完整性汇总 | 是 |
| `results/exp_eval_aeqa/log_0.00_1.00.log` | 首次 41 题运行日志 | 追溯用 |
| `results/exp_eval_aeqa/log_0.83_1.00.log` | 首次运行后半段重跑日志 | 追溯用 |
| `results/exp_eval_aeqa_api_retry/gpt_answer.json` | 7 个 API/VLM 异常题的重跑答案 | 已并入正式结果 |
| `results/exp_eval_aeqa/gpt_answer.json` | 脚本自动汇总文件，含早期 1 题冒烟测试的重复记录 | 否，勿用于统计 |

## 当前数值

| 指标 | 数值 | 含义 |
| --- | ---: | --- |
| 已执行问题 | 41 / 41 | 所有问题均已按固定配置启动并结束 |
| 有效预测答案 | 38 / 41（92.68%） | 成功得到模型最终答案的比例，不是问答准确率 |
| 源码设定下的失败 | 3 / 41（7.32%） | 无最终答案，问答评分时应计 0 分 |
| 平均筛选快照数 | 3.37 | 以最终重跑结果替换 API 失败题后统计 |
| 平均总快照数 | 10.51 | 同上 |
| 平均总帧数 | 31.27 | 同上 |

仍失败的 3 题及原因：

1. `6d132959-fd48-4fef-a736-4e5853849547`：达到源码 `num_step: 50` 的探索上限，未得到答案。
2. `b05e7b30-6a4d-4381-9d05-a42ed0c90e30`：Habitat pathfinder 找不到下一可导航点。
3. `f17869a2-2a4d-4ce4-b262-cb69618e3394`：找不到满足源码观察距离约束的快照观察点。

这些失败保留为结果的一部分；没有修改算法参数来使其通过。

## 问答准确率（待生成）

论文对 A-EQA 使用 GPT-4 评分的 `LLM-Match`（0--100），而不是字符串精确匹配。仓库没有发布该评分器，因此新增了：

```text
scripts/score_aeqa_llm_match.py
```

它将对 38 条答案进行 GPT-4o 语义评分，并将上述 3 条无答案样本自动记为 0 分，生成：

```text
results/exp_eval_aeqa/aeqa_llm_match.json
```

当前该评分尚未完成，原因是 Requesty 返回 `402 Payment Required`（账户余额不足），所以现在不能报告真实的问答准确率或 LLM-Match。充值后运行：

```bash
python scripts/score_aeqa_llm_match.py --batch-size 1
```

完成后该 JSON 会同时包含：

- `llm_match_all_questions`：41 题平均 LLM-Match，3 个失败题为 0；
- `binary_accuracy_all_questions`：完全语义正确答案的 41 题比例；
- `items`：逐题预测、参考答案、分数和评审理由。

## 与论文的关系

本实验说明环境、数据路径、Habitat EGL 渲染、检测模型、HGR 流程和 GPT-4o 调用都已跑通，并给出了 41 题上的本地 baseline。

它不能直接与论文表 5 的 `55.9` 对比：论文使用 184 题子集并按其 GPT-4 评审协议汇总；这里是 41 题且评分尚未完成。若要接近论文设置，应改用 `data/aeqa_questions-184.json`（本地 HM3D val 已覆盖其 57 个场景），并使用同一评分脚本和固定实验设置。
