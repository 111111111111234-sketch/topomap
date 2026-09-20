# Adaptive-hybrid 3×2 开发集结果

日期：2026-09-16。

本轮 `adaptive_hybrid_3x2_v1` 已完成两个 split，共 3 个 scene、6 个 episode、
49 个 goal。对照使用同一清单上已经完成的 Route-only 和修复后 Memory-dedup。

## 一、主结果

| 配置 | GOAT SR | GOAT SPL | Snapshot SR | 状态 | Step | 路程（m） | VLM 调用 |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| Route-only | 65.31% | 50.78% | 22.45% | 49 completed | 254 | 155.10 | 428 |
| Memory-dedup | 67.35% | 52.85% | 20.41% | 49 completed | 288 | 171.36 | 461 |
| **Adaptive-hybrid** | **77.55%** | **61.05%** | **34.69%** | 45 completed、4 exhausted | 349 | 190.37 | **387** |

Adaptive 相对 Route-only：

- GOAT SR 增加 12.24 个百分点，即净增加 6/49 个成功。
- GOAT SPL 增加 10.27 个百分点。
- VLM 调用减少 41 次（9.6%）。
- Step 增加 95（37.4%），路程增加 35.26 m（22.7%）。
- 配对变化为 10 个失败转成功、4 个成功转失败。

按 goal 类型，Adaptive 的 SR/SPL 均高于 Route-only：image 为
84.62%/56.06%，description 为 76.47%/66.16%，object 为 73.68%/59.90%。

## 二、配对区间

以 scene 为 block 的配对结果：

- GOAT SR 差值为 +12.24 pp，95% 区间 `[0.00, 29.41]`；两个 scene 胜、一个持平。
- GOAT SPL 差值为 +10.27 pp，95% 区间 `[-1.52, 30.03]`；两个 scene 胜、一个负。
- Snapshot SPL 差值为 +12.44 pp，95% 区间 `[4.05, 24.21]`。

只有 3 个 scene，因此区间仍很粗。SR 下界恰好为零，SPL 区间跨零；这些结果支持
进入独立最终集确认，但还不支持声明稳定优于 Route-only。

## 三、机制和验证审计

- 43 次 REVISIT，31 次选择跨 goal 历史来源；重复证据选择为 0。
- 只产生 22 次终点验证：15 confirmed、3 rejected、4 uncertain。
- 15 个确认停止中 13 个满足距离成功，确认精度为 86.67%。
- 两个错误确认均为 description goal，最终距离分别为 6.98 m 和 1.05 m。
- 选择性验证任务计数为 19；uncertain 的额外视角使实际验证调用达到 22。
- Frontier intent 三步窗口到期 36 次。
- 已记录 346 个已知空间运动事件；没有模型请求错误或未分类运行错误。

相对全量 Verified-ablation，选择性验证将验证次数从 99 降至 22、总 VLM 调用从
697 降至 387、Step 从 629 降至 349、路程从 334.94 m 降至 190.37 m，同时 GOAT
SR/SPL 从 71.43%/34.38% 提升至 77.55%/61.05%。这说明按来源和 goal 类型限制验证
比全量验证更有效。

## 四、异常结束审计

4 个 exhausted 全部位于 `00062-ACZZiU6BXLz_1`：

- 1 个 image goal 达到 50 step 预算，最终距离 4.37 m。
- 1 个 object goal 选择了不可执行 Frontier，最终距离 9.80 m。
- 1 个 image goal 选择了不可执行 source，最终距离 7.43 m。
- 1 个 object goal 选择了不可执行 Frontier，最终距离 9.09 m。

其中三个 goal 在 Route-only 中同样距离失败；一个 image goal 在 Route-only 中成功，
是 Adaptive 的实际回退。不可执行选择被安全拒绝，没有绕过已知空间约束，也没有被
错误记录为完成。

## 五、开发集决策

冻结当前 Adaptive-hybrid 为批次 D 的候选，不继续针对这 3 个 scene 调参，以免开发
集过拟合。它尚不是正式默认版本；Route-only 继续保持当前正式状态，直到独立 12×2
清单的重复评测完成。

完整机器可读结果：

- `artifacts/hgr_rebuild/adaptive_hybrid_3x2_v1_summary.json`
- `artifacts/hgr_rebuild/adaptive_hybrid_3x2_v1_summary.md`

最终评测命令：

```bash
python scripts/run_hgr_experiment_plan.py \
  --phase final \
  --final-candidate adaptive-hybrid \
  --gpu 5 \
  --threads 16
```

批次 D 使用独立 12×2 清单和三次重复，成本较高；运行前应重新检查 GPU 状态。
