# Phase C 前置：稳定目标身份

2026-09-09 的历史前置修复报告，范围仅为 route-only 选中源续行。
后续 Phase C 已实现并跑完三次 smoke，但存在独立的 overlay 合并和状态生命周期缺口；
当前结论见 [三次 smoke 复核](HGR_PHASE_C_THREE_SMOKES_REVIEW.md)，
整体设计以 [主规格](HGR_DUAL_TOPO_GENERAL_DEVELOPMENT_SPEC.md) 为准。

## R5 逐例诊断

24 次 `hgr_source_mapping_disappeared` 全部发生在 snapshot commitment。将对应 trace 与同一 subtask 结束时的 scene graph 日志关联，23 次原 object ID 仍然存在，但代表图像已改变。对象 ID 在 episode 内单调分配，不复用；这支持这 23 次由 snapshot 重聚类替换代表图像、旧代码仅查询 `scene.snapshots` 导致错误释放的判断。

剩余 `00720-8B43pG641ff_0_2`、step 10、object 120 在结束图中不存在。历史记录没有逐步对象合并 lineage，不能区分其在释放时被合并、过滤删除，还是之后才消失。不能据此声称 24 次都能被修复。

逐例清单见 [r5_source_mapping_audit.json](r5_source_mapping_audit.json)。结束图不是释放时完整状态快照；本次没有做闭环反事实重放。

## 修复

- `SnapshotSourceCommitment` 在选择时复制原图像、观测位置和选中对象集，snapshot 重聚类不再使该证据失效；续行保留原 capture Place，避免换代表图像后偷偷换导航锚点。
- 感知对象真实合并时记录 episode 内 `旧 ID → 存活 ID`。续行只沿记录的合并链解析；不按类别、位置或 GT 猜测替代对象。
- 所有选中对象都必须有存活身份；部分集合丢失、合并链终点删除或链异常时明确释放。合并到同一对象的 ID 去重。
- 终端段、回退、释放和新 subtask 清理缓存。释放后清空 route execution 进度。
- 新增 `place_route_source_resolution`，记录原图像是否仍在 snapshots/frames、原 ID、执行 ID、合并链和具体原因。旧的 release 总类保留，增加详细 source_resolution。

历史图像作为选中证据保留，不等于其内容被当前观测重新确认；原 HGR 的停止和成功判定未改变。该层不承担 Phase C 的目标验证职责。

## 验证

在现有 hgr 环境中完整原测试集及新增 source tests 共 184 项通过；真实 merge producer 的两项额外测试通过，共 186 项。覆盖重聚类与 pose 保留、选择对象集合不扩张、链式合并、多对象合并去重、删除对象拒绝替代、环检测，以及可选 lineage 记录不改变原合并结果。

独立运行配置：

- `cfg/eval_goatbench_place_source_identity_qwen3vl_dashscope_train_smoke.yaml`
- `cfg/eval_goatbench_place_source_identity_qwen3vl_dashscope_train.yaml`

本次未启动在线评估，不能报告修复后的 SR/SPL 或实际释放减少数量。下一次 smoke 应检查新 resolution 事件、原 capture Place 保留和非预期循环，再做固定开发集配对评估。
