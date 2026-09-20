# HGR V2 R3：生命周期、路线复用与验收修复

首轮在线结果与最新 hypothesis pin 修复见 [R3 Smoke 复核](HGR_DUAL_TOPO_V2_R3_SMOKE_REVIEW_20260912.md)。

状态：第二次 pin 修复、完整回归和 C-persistent 在线复核均通过；自动 source audit 输出 `cascade_acceptance=passed`，active-intent hypothesis 生命周期缺口已关闭。

首轮 R3 在线运行已证明 persistent frontier 能到达验证终点，但暴露出 TSDF 每步清理旧 frontier 时也会删除 active intent 引用的 hypothesis DAG 节点。第一次补丁仅覆盖 belief-v2 deferred semantic 分支，而 R3 的 stable-only 配置实际走 eager 清理，所以第一次 `verify-r3` 的 13 次验证仍全部找不到节点。现已将 active hypothesis refs 直接传入 `update_frontier_map()` 的 eager 清理路径；意图验证或释放后才恢复普通清理。最新 C-persistent 的 7 次验证全部有效，并完成 7 次 falsification/retraction；自动审计确认候选身份、独立证据、hypothesis refs 和空间事实不变量全部通过。

## 行为变化

- persistent_intent 开启时，保存独立的探索 frontier 副本。实时 frontier 消失后保留意图和已认证终点，每次运动前仍检查并修复路线；不做邻近 frontier 的猜测合并。关闭时保留逐步选择。
- frontier 到达后保留 verification_pending 意图，重新采集终端 RGB/depth 并检测对象，完成验证与跨层撤销后再释放。观测失败、无检测对象时跳过，不把失败当负证据。
- 修正 SemanticCritic 把 semantic_class=None 当实际类别的问题。实测引用不会自动代表目标匹配，也不会自动为任意假设提供独立支持。
- 选择后复用原 RoutePlan；起点、相关边、栅格状态或体素尺度不匹配时重新认证。目前采用完整栅格哈希保守失效，后续可细化搜索域依赖。
- 缓存记录条目创建上下文，命中分成同决策、同一步、跨步、跨目标；记录哈希与实际搜索耗时，不用命中数推算节省时间。
- 审计分别报告正常运动契约、假设撤销覆盖和恢复覆盖；没有触发时输出 not_covered。撤销日志包含候选引用前后状态及空间事实保持检查。
- 图像目标保持原选择和停止协议，新增 v2_goal_diagnostic，关联最终实体选择、到达、原 task_success 和评测结果。

## 用户运行

在 hgr 环境及项目目录下，单条命令运行回归测试，再依次运行新的 B、C-shadow、C、C-persistent、D；D 结束后执行固定记录的缓存对照。失败即停止，每阶段新建结果目录。

```bash
bash scripts/run_hgr_v2_stage.sh fixes 7
```

最新 hypothesis pin 补丁已在线验收，无需再次运行 C-persistent 或 D。

仅运行测试：

```bash
bash scripts/run_hgr_v2_stage.sh tests 7
```

固定记录的缓存等价性检查可单独执行：

```bash
python scripts/hgr_v2_experiments.py cache-replay RESULTS/decision_records --output RESULTS/cache_replay_new.json
```

将 RESULTS 换为新 B/C/D 结果路径。旧记录必须另传 --observe-distance，取对应运行的 planner.final_observe_distance；A 记录不支持这个共享路线对照。输出文件必须不存在。

此命令重建固定空间图，在相同起终点比较缓存关闭、冷缓存、重复查询暖缓存；输出路径与成本等价性及耗时。这是计算缓存实验，不是空间记忆导航收益实验。

## 空间记忆对照

`hgr_v2_experiments.py commands --suite memory` 生成的四组 baseline/V2 × cold/warm 配置现在默认设置：

```yaml
controlled_memory:
  preserve_hgr_memory: true
  single_subtask: true
```

四组共享同一 checkpoint 中的 Scene、TSDF、原 HGR 历史和随机状态；warm 恢复 Place 图，cold 使用新 Place 图。每次只评测 checkpoint 后首个目标，防止前序行为造成起点漂移。
比较工具核对 checkpoint、起点、朝向和上述控制项；旧的全历史冷暖实验仍可运行，但不再被新比较工具接受为空间图独立收益证据。
已有检查点导出配置和命令生成接口继续使用；prefix 开销保留在元数据中。多个独立检查点、场景和重复才足以形成泛化结论。

## 验收重点

回归增加：持续 frontier 源消失/原对象被修改、路线起点和栅格依赖变化、缓存互斥归因、终点阻塞、新观测使用与采集失败、真实 SemanticCritic→hypothesis DAG→GoalState 级联且空间事实保持。

下一轮结果必须同时查看执行审计、verification/retraction 事件、cache 分类和原始成功率。正常 smoke 可能仍不包含证伪或动态阻塞；这种运行只能说明相应分支未覆盖，不能将未覆盖当作通过。固定纠错测试覆盖事务正确性，真实在线触发仍须另行确认。
