# HGR 双层拓扑分阶段实施报告

日期：2026-09-13。首版代码接入完成，全量离线回归 347 项通过。按用户要求，本次没有执行 smoke、simulator、模型请求或性能评测。

## 设计与实现

在现有工作区中增加独立链路，配置根为 `eval_goatbench_hgr_baseline_qwen3vl_dashscope_train.yaml`。保留旧实验代码，不继承 V2-D 的策略。参考 V6 的跨子任务记忆、C-strict 的保守历史复用、direct-first 与几何重绑定原则；语义选择仍由 HGR 提供，不额外建立全候选置信度重排系统。

| 阶段 | 配置（位于 `cfg/`） | 行为 |
| --- | --- | --- |
| 原版 | `eval_goatbench_hgr_baseline_qwen3vl_dashscope_train.yaml` | 因果对照；保留原版几何与停止行为 |
| Shadow | `eval_goatbench_dual_dynamic_shadow.yaml` | 只同步 L1/L2 与记录原选择 |
| Known-space | `eval_goatbench_dual_dynamic_known_space.yaml` | 原语义与停止，改为已观察自由空间执行 |
| Route-only | `eval_goatbench_dual_dynamic_smoke.yaml` | 原语义与停止，接入认证拓扑路线与真实 approach |
| Memory | `eval_goatbench_hgr_dual_topo_memory.yaml` | 源绑定 APPROACH / REVISIT / EXPLORE，保持基准决策节奏 |
| Intent | `eval_goatbench_hgr_dual_topo_intent.yaml` | 持续意图、增量实体验证后抢占、几何重绑定与恢复 |
| Verified | `eval_goatbench_hgr_dual_topo_verified.yaml` | 同一实体的新鲜终点验证与独立停止协议 |

后三阶段由 `HypothesisAwareNavigator` 统一拥有任务、意图与反馈。Memory 配置不会同时安装另一套记忆 intent。新配置启动前拒绝 stage/backend 不一致，以及 V2、Phase C、fusion、hierarchical 或旧 V7 策略混用。

| 文件 | 实现职责 |
| --- | --- |
| `src/dual_dynamic_navigation/unified_navigator.py` | 候选资格与授权、唯一 intent、源失效、pin、增量抢占、反馈、恢复 |
| `src/dual_dynamic_navigation/task_planner.py` | 三类任务与结果类型、意图数据、重访机会成本门控；首版调度逻辑位于 Navigator |
| `src/dual_dynamic_navigation/hgr_adapter.py` | 原 HGR 查询接口、精确候选集合与 crop 证据摘要 |
| `src/dual_dynamic_navigation/route_resolver.py` | 实际终端、新观察点、完整已知空间路线与成本 |
| `src/dual_dynamic_navigation/executor.py` | 已知自由空间证书、路线推进与实际运动审计 |
| `src/dual_dynamic_navigation/feedback.py` | 选中历史实体 crop 与新鲜 RGB 联合验证，错误与不确定性分离 |
| `src/dual_dynamic_navigation/scene_map.py` / `goal_graph.py` | 跨子任务场景事实、目标证据、alias、视角覆盖及正负反馈 |
| `src/hypothesis_graph.py` | pin 延迟移除；新链路中独立支持保留与失效支持撤销 |
| `src/hgr_experiment_state.py` | schema v4 backend/配置/策略契约，Open3D 数组持久化 |
| `run_goatbench_evaluation.py` | 观察、选择、设置终端、运动、反馈与恢复的实际接入 |

新 executor 向实际 `agent_step` 传入 `pathfinder=None`，终端设置使用 `known_space_only=True`。无已知路径时返回 blocked，不回退到完整导航网格。原版和 Shadow 保持原行为，比较时必须披露几何访问差异；离线接口测试不能代替 simulator 全调用边界审计。

历史重访要求未检查的新视角、已知可达路线、剩余预算及探索机会成本门控。授权绑定返回的具体 source/entity/evidence，候选 A 的返回不能放行 B。增量对象必须来自本次检测，再通过同一实体确认，才能抢占探索或重访。

终点 `confirmed` 才请求新版成功停止；`uncertain` 最多尝试两个视角，耗尽后保留 unresolved；`rejected` 否定目标关联；请求错误不产生负事实。陈旧 intent 和重复反馈被忽略。目标实体 alias 迁移覆盖记录；Frontier 重提取保留物理终端，替代绑定要求方向一致与已知连通。

## 静态与离线验证

全量回归命令：

```bash
cd /home/hdd/tangyuxin/projects/Hypothesis_Graph_Refinement
MPLCONFIGDIR=/tmp/hgr-matplotlib YOLO_CONFIG_DIR=/tmp/hgr-ultralytics \
  /home/tangyuxin/miniconda3/envs/hgr/bin/python -m unittest discover -s tests -q
```

结果：347 项通过。新增测试覆盖原查询同输入像素/映射/RNG 回放、真实 TSDF 局部运动与未知区域阻断、精确授权与预算、实际 runner 分支、增量确认、幂等验证、alias、pin、独立支持级联、恢复及包含实际 Open3D 几何的 checkpoint 往返。外部模型边界使用 mock；不包含在线动作序列等价或成功率证明。

源码快照位于 `artifacts/hgr_rebuild/20260913T120726Z/`。`source.tar.gz` 冻结本轮修改前工作区；`baseline_source.tar.gz` 冻结当前 sibling baseline checkout，均有指纹清单。它们不重建历史 V7 运行源码。runner 的 `--record_run_fingerprint` 记录实际运行溯源，`--run_name` 使用独立结果目录。

## 手动在线指令

### 一键执行离线回归与七阶段 smoke

新增执行入口 `scripts/run_hgr_stages.py`。以下命令会实际启动 simulator 并请求模型；本次编写脚本时没有执行在线评测。

```bash
cd /home/hdd/tangyuxin/projects/Hypothesis_Graph_Refinement
export DASHSCOPE_API_KEY='你的密钥'
python scripts/run_hgr_stages.py --gpu 4
```

默认依次执行：离线全量回归 → Baseline → Shadow → Known-space → Route-only → Memory → Intent → Verified。所有在线阶段共享同一 manifest、split 和 GPU，串行执行；某条命令非零退出时停止后续阶段。进程退出成功不等于阶段验收通过，仍需检查结果与导航轨迹。

仅预览全部命令（无需密钥，不执行测试）：

```bash
python scripts/run_hgr_stages.py --gpu 4 --dry-run
```

每个阶段的独立运行指令如下。在线命令中的 `--skip-offline` 表示已经单独完成离线回归，避免每次重复执行：

```bash
python scripts/run_hgr_stages.py --stage offline
python scripts/run_hgr_stages.py --stage baseline --gpu 4 --skip-offline
python scripts/run_hgr_stages.py --stage shadow --gpu 4 --skip-offline
python scripts/run_hgr_stages.py --stage known-space --gpu 4 --skip-offline
python scripts/run_hgr_stages.py --stage route-only --gpu 4 --skip-offline
python scripts/run_hgr_stages.py --stage memory --gpu 4 --skip-offline
python scripts/run_hgr_stages.py --stage intent --gpu 4 --skip-offline
python scripts/run_hgr_stages.py --stage verified --gpu 4 --skip-offline
```

脚本从任意目录启动都会以项目根目录为工作目录。默认评测解释器为 `/home/tangyuxin/miniconda3/envs/hgr/bin/python`，可用 `--python /absolute/path/to/python` 替换。日志和包含命令、退出码的汇总保存到 `artifacts/hgr_rebuild/runs/<UTC时间>/`；runner 结果使用各阶段独立的 `exp_hgr_rebuild_<stage>_<UTC时间>` 名称。

默认仍是 ACZZ 场景 episode 0 的单 episode smoke。扩大到现有 12×2 开发清单：

```bash
python scripts/run_hgr_stages.py --gpu 4 \
  --manifest cfg/manifests/goat_train_dev_12x2_seed77.json --split 1
```

`--split` 只运行清单指定的对应分片；其他分片需分别执行。这些命令覆盖当前已接入七阶段，不包含尚未实现的历史 C-strict 冻结对照、消融、完整动作回放或 P6 全部验收。

### 原有只打印入口

以下工具只打印命令，不执行评测；先在运行环境配置 `DASHSCOPE_API_KEY`：

```bash
cd /home/hdd/tangyuxin/projects/Hypothesis_Graph_Refinement
python scripts/hgr_rebuild.py commands --gpu 4
```

指定单个阶段：

```bash
python scripts/hgr_rebuild.py commands --stage verified --gpu 4
```

打印的命令均固定相同短程 manifest，并使用带 UTC 时间的独立目录。GPU 编号由运行时空闲情况决定。先执行原版/Shadow 对照，再按 Known-space、Route-only、Memory、Intent、Verified 顺序验收；短程指令不代表 P6 完整评测。

运行后审计包含 `known_space_motion` 的 JSONL 文件：

```bash
python scripts/hgr_rebuild.py audit /absolute/path/to/navigation_trace.jsonl
```

审计仅检查已记录运动的证书字段，不能证明没有漏记运动，也不替代完整任务、语义来源或性能验收。

## 仍待完成的验收与实现边界

- P0/P1：历史强版本源码溯源、完整 episode 同输入动作和请求序列等价回放。
- P2–P5：simulator 完整几何调用边界、运动日志完整性、在线重绑定/恢复和真实模型验证协议。
- P6：原版、历史 C-strict、新各阶段与关闭单项机制的冻结性能对照。
- 首版没有新增完整语义缓存；去重使用实体证据摘要及物理观察点覆盖。
- 首版恢复上限为一个 intent 内一次局部 corridor 修复加一次同终端拓扑重规划，成功后不逐帧重置次数。这比按每次阻塞事件重置更保守，后续应由在线恢复轨迹决定是否调整。
- 主循环接入完成，但 runner 尚未全面精简为纯 backend 调度器；旧分支仍作为历史对照保留。

代码接入不等于阶段验收通过。本报告不声称 SR/SPL 提升，也不使用旧 V2-D smoke 结果作为本轮证据。
