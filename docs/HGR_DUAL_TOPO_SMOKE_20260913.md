# HGR 双层拓扑七阶段 smoke 结果

运行批次：`20260913T123622024858Z`。固定输入为 ACZZiU6BXLz episode 0，共 8 个子任务（image 4、object 2、description 2）。七个在线进程和离线回归均以退出码 0 完成，所有结果的 `metric_validity.json` 均含 8 个有限样本。

## 聚合指标

| 阶段 | Snapshot SR | Snapshot SPL | Distance SR | Distance SPL | 步数 | 路程 m | VLM 调用 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | 25.00 | 13.41 | 75.00 | 46.15 | 44 | 40.20 | 87 |
| Shadow | 0.00 | 0.00 | 75.00 | 46.95 | 46 | 41.19 | 83 |
| Known-space | 25.00 | 20.32 | 50.00 | 41.67 | 43 | 30.20 | 71 |
| Route-only | 37.50 | 22.72 | 87.50 | 55.98 | 85 | 54.11 | 127 |
| Memory | 0.00 | 0.00 | 37.50 | 16.78 | 282 | 153.22 | 347 |
| Intent | 12.50 | 6.36 | 62.50 | 21.43 | 279 | 142.64 | 518 |
| Verified | 50.00 | 21.70 | 75.00 | 36.16 | 147 | 55.81 | 355 |

这是单场景、单 episode 的功能 smoke，样本间还共享场景历史，不能据此声明性能提升。Route-only 在本次样本的两套指标中最高；Memory 明显退化且执行长度大幅增加。Verified 的 snapshot SR 最高，但其在线运行受到下述实现缺陷影响，当前数值不能作为最终验收值。

## 结构与运行审计

| 阶段 | 已记录 known-space 运动 | blocked | 未认证运动 | 审计 |
| --- | ---: | ---: | ---: | --- |
| Known-space | 43 | 1 | 0 | 通过 |
| Route-only | 85 | 0 | 0 | 通过 |
| Memory | 281 | 0 | 0 | 通过 |
| Intent | 278 | 0 | 0 | 通过 |
| Verified | 147 | 0 | 0 | 通过 |

Shadow 按设计不切换 known-space executor，因此没有 `known_space_motion`。Baseline 不生成新链路 trace。所有新阶段均记录了 8 次 goal start/end；没有 Python traceback。运动审计只证明已记录运动带有效几何证书，不能证明不存在漏记运动。

## 发现的问题与处理

Intent 日志出现 9 次、Verified 出现 55 次 Semantic Critic 特征计算设备错误：预处理图像停留在 CPU，而 CLIP 权重位于 CUDA。异常被降级为固定 feature residual 0.5，运行没有崩溃，但可能改变语义验证和后续动作。已在 `src/semantic_critic.py` 中将输入迁移到模型的 device/dtype，并将历史特征迁移到实际特征的 device/dtype；新增回归测试后全量 348 项离线测试通过。

Memory 和 Intent 各有一个 `query_vlm_for_response failed`。Memory 的 `00062-ACZZiU6BXLz_0_4` 被日志明确标为 invalid，但最终保存为有限的失败值，所以 `metric_validity.json` 仍报告 invalid=0。这意味着当前 validity 文件只检查数值有限性，不能识别上游请求失败。Memory 和 Intent 的本轮聚合指标应标记为受请求失败影响。

Intent 另有 20 次、Verified 有 6 次 `explore_step failed and returned None`；分别伴随 52 次和 15 次 `no snapshot is available`。它们没有导致进程失败，但显示候选耗尽/响应解析路径频繁触发，扩大评测前需要按 trace 对照判断是合理阻断还是调度缺陷。

## 建议重跑

修复后至少重新运行 Intent 和 Verified。为了消除请求失败对阶段对比的影响，也应重跑 Memory；使用新的独立运行名，保留本批次作为失败证据。重跑通过后再扩大到 12×2 清单。当前批次不满足最终性能验收。
