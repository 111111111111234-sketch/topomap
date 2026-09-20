# Adaptive-hybrid 单场景 smoke 结果

日期：2026-09-16。

本轮 `adaptive_hybrid_v1` 完成 Route-only、Memory-dedup 和 Adaptive-hybrid
三个配置。三项使用相同的单 scene、单 episode、8 goal 清单；所有任务均得到明确
结束状态，无模型请求失败、缺失结果或未分类错误。

## 主结果

| 配置 | GOAT SR | GOAT SPL | Snapshot SR | Distance SR | Step | 路程（m） | VLM 调用 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Route-only | 75.00% | 44.83% | 25.00% | 75.00% | 76 | 49.52 | 114 |
| Memory-dedup | 50.00% | 40.58% | 37.50% | 50.00% | 55 | 34.88 | 90 |
| Adaptive-hybrid | 87.50% | 46.23% | 25.00% | 87.50% | 74 | 45.55 | 79 |

相对 Route-only，Adaptive-hybrid 在本次 smoke 中：

- 多成功 1/8 个 goal，GOAT SR 增加 12.5 个百分点。
- GOAT SPL 增加 1.40 个百分点。
- Step 从 76 降至 74，路程减少 3.96 m（约 8.0%）。
- VLM 调用从 114 降至 79，减少 35 次（约 30.7%）。
- 8 个任务全部 completed；4 个使用普通 Snapshot 到达停止，4 个使用验证确认停止。

按 goal 类型，Adaptive 的 image goal 为 4/4 成功，object 为 1/2，description 为
2/2。相对 Route-only 的净新增成功来自一个 image goal。

## 机制审计

- 选择了 7 次 REVISIT，其中 6 次使用跨 goal 历史来源。
- 只对其中 4 个 image/description 历史 REVISIT 执行验证；object goal 未触发验证。
- 4 次验证全部返回 `confirmed → stop`，且 4 个任务按真实距离口径全部成功。
- 另外 3 个 REVISIT 按原停止链路提升为 APPROACH。
- Frontier 三步 intent 窗口到期 6 次，证明长期 Frontier 承诺已被限制。
- 重复证据选择为 0；74 个已知空间运动事件全部被记录。

Snapshot SR 仍只有 25%，因为 8 个 goal 中存在 4 个映射不可用，并且 Snapshot-ID
口径与距离口径不同。任务成功继续以 GOAT SR 为主，不用 Snapshot SR 否定这 7 个
距离成功任务。

## 结论与限制

这次结果支持继续评测 Adaptive-hybrid：它没有重现全量 Intent/Verified 的成本膨胀，
而且三个主目标——成功率、SPL、运行成本——在点估计上都没有劣于 Route-only。

但样本只有一个 scene。报告中的 scene-block 区间退化为该单场景的固定差值，不能
表示跨场景置信区间；模型运行也并非完全确定。因此当前结论是“smoke 通过、进入扩大
样本筛选”，不能声明 Adaptive 已可靠优于 Route-only 或已经是最终最优版本。

机器可读汇总位于：

- `artifacts/hgr_rebuild/adaptive_hybrid_v1_summary.json`
- `artifacts/hgr_rebuild/adaptive_hybrid_v1_summary.md`

下一步只运行 Adaptive-hybrid 的 3×2 开发清单即可先验证泛化；Route-only 和
Memory-dedup 已有同一清单的 C-lite 结果可作开发参考：

```bash
python scripts/run_hgr_stages.py \
  --stage adaptive-hybrid \
  --manifest cfg/manifests/goat_train_fast_3x2_seed77.json \
  --split all \
  --gpu 5 \
  --threads 16 \
  --matrix-id adaptive_hybrid_3x2_v1 \
  --resume \
  --skip-offline
```

若 3×2 结果仍同时满足 GOAT SR、GOAT SPL 和成本门槛，再决定是否在独立 12×2
清单上运行 Route-only 与 Adaptive 的正式重复评测。
