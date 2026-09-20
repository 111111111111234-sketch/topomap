# Phase C：Place 图上的目标导航层

更新：2026-09-10。状态：C-R1/C-R3/C-R4 smoke 均已跑完，性能与整体正确性验收未通过。
后续目标覆盖修正与连续 route guidance 已接入代码，见 [最新实现说明](HGR_ROUTE_GUIDANCE_IMPLEMENTATION.md)。
以下“实现缺口”记录的是修改前的 C-R4 审阅，在线历史结果保持不变。
本文件保留现有实现/历史命令；下一版以 [主规格](HGR_DUAL_TOPO_GENERAL_DEVELOPMENT_SPEC.md)
与 [连续执行设计](HGR_TOPO_GUIDED_CONTINUOUS_DESIGN.md) 为准。

## 完整 smoke 后的结论与实现缺口

详见 [三个 smoke 复核](HGR_PHASE_C_THREE_SMOKES_REVIEW.md)。C-R4 的 SR/SPL 为
50.00%/23.90%，耗时 49:38；C-R3 为 62.50%/36.64%，耗时 84:40。
B-R5 smoke 为 87.50%/47.82%，耗时 17:37。速度改善伴随质量下降，不能作为完整方法交付验收。

此前“已完成重构”的说明超过了实际实现范围，以下为当前明确限制：

- R4 关闭 `(intent, entity, Place)` 动作后没有相关证据/几何版本失效与重新激活；
  新增其他 Place 可能提供新动作，但不等于原动作因新证据恢复。
- 主流程 checked 重绑定取了 tuple 的 intent 字段而非 entity；直接测试 helper 没有覆盖此问题。
- 执行失败也进入 checked，模型错误与语义 uncertain 尚未充分区分。
- revision guard 返回 None 后主入口会 break；尚未具备明确的 Continue/Retry 协议。
- Place 状态是候选的投影；未完成独立增量证据更新/批量评分，仍对全量候选构造选择输入。
- R4 两次选择解析失败提前结束；六次 matched 中两次 Distance 失败，需进一步视觉/几何诊断。

下一轮先做正确性修复，再将连续 route guidance 作为固定目标/停止协议下的执行器实验。
目前没有 corridor 实现，也不能将所有非 Phase C 计时归给 Hypothesis Graph。

## R2 撤回与 R3 等价计算优化

R1 smoke 出现单个对象重复选择 15 次、候选接近 60 个的情况。此前把日志中
39–58 秒的选择阶段全归为 API 延迟并不准确：该区间还包括候选路线搜索、图像编码、
联系表生成，以及部分其他模型调用。现有日志只能确认重复选择增加总工作量，不能精确划分这些耗时。

R2 的永久实体淘汰、不确定验证次数限制、候选 top-k、8 次调用上限、缩小/合并图像及
省略当前视角均已撤回。这些改变会影响模型选择与 SR/SPL，不能当作等价加速。
R2 启动配置已删除；历史运行结果保留。

R3 仅做以下计算复用：

- 图像缓存按实际像素、形状和 dtype 做内容寻址；联系表缓存包含全部标签、顺序和图片编码。
  仍展示全部候选、每个 Verify 的 context 与 crop、当前视角以及原尺寸联系表（12 tile/页）。
  每个 goal 的缓存上限为 64 MiB 编码内容；像素或裁剪改变会重新编码，源身份每次重新校验。
- 路径结果只在一次同步候选构建中复用；下一次构建清空缓存，避免跨地图/边更新沿用旧路线。
- `cache_paths`、`cache_visuals` 可分别关闭以做等价对照。
- trace 增加 `view.planning_metrics`、`selection.timings`、`verification.timings`。
  `api_seconds` 包含请求处理和重试等待；`visual_prepare_seconds` 包含提示准备。
  汇总脚本报告次数、总量、中位数；planning 的 search_seconds 是 build_seconds 的子集，不能相加。

缓存不减少上传内容或模型调用次数，也不消除 R1 重复验证；实际总时长改善需在线测量。
缓存开关两侧完整请求一致不意味着随机模型输出必然相同，SR/SPL 仍需运行验收。
R3 使用独立输出目录；已经运行的 Python 进程需重新启动才会加载新代码。

R3 smoke 的同一首个 image subtask 从 R5 的 17 steps / 4 分 24 秒增加到
50 steps / 20 分 40 秒。它执行了 39 次 Phase C 决策，验证结果为 18 rejected、
17 uncertain，并完成 3 次 Explore。Phase C 路径、图像准备、选择和验证的记录耗时
合计约 5 分 28 秒；更多总耗时来自额外 step 重复执行原视觉建图和 HGR 流程。

## R4：Place 上的目标动态覆盖层

R4 将第二层显式表示为 `GoalTopologyOverlay`。它只复用第一层的 Place ID 和 valid 边，
不创建第二套空间坐标或连通图。每个 Place 保存当前 goal 的 Verify 实体、Explore frontier、
证据引用、已完成动作和搜索状态。候选是覆盖层在当前时刻产生的动作投影，不是长期地图本体。

验证动作从连续网格点改为稳定的 `(intent, entity_id, approach_place_id)`：

- 同一对象在同一 approach Place 只执行一个 Verify；局部安全环仍用于寻找该动作的最短可达终端点；
- rejected/uncertain 只关闭该 Place 上的动作，不降低对象或整个 Place 的目标支持；
- 同一 Place 内的相邻网格点不再生成连续重试；新增的可达 approach Place 可以产生新动作；
- frontier 同样按 `(frontier, approach Place)` 记录，目标切换会创建全新覆盖层；
- 设计意图为稳定对象合并时一起重绑定证据与已检查动作；当前 checked 主流程存在上述字段索引缺口。

覆盖层维护单调 revision guard。选择入口比较新 goal、验证/执行反馈或候选、证据、Place/边、
路线发生变化后决策；相同 revision 再次进入选择器会返回
`no_goal_topology_change`，不发送模型请求。一个 intent 选定后仍由
`PlaceRouteExecution` 跨局部 step 和 Place waypoint 持续执行。

R4 trace 在原字段之外记录 `goal_topology`、是否启用 topological actions 和事件驱动选择。
审计会拒绝同一 subtask 内重复执行相同拓扑动作，以及未改变 goal-topology revision 就再次调用模型。
R3 的两个开关保持关闭，可作为原候选循环对照；R4 使用独立配置和输出目录。

## 实现范围

主规格为 `HGR_DUAL_TOPO_GENERAL_DEVELOPMENT_SPEC.md` 第 5、6 节与阶段 C。
新模块 `src/place_goal_navigation.py` 保存当前 goal 的支持证据、检查视角、执行意图和反馈；
`src/query_place_goal.py` 调用现有同模型 VLM 做候选选择与当前画面验证。
第一层继续使用 `PlaceTopology`，路线继续使用 `PlaceRouteExecution`，未增加独立空间图。
旧 V7 dynamic/region 模块保留为历史对照，不叠加启用。

Object、Description、Image 共用同一候选生成、图搜索、状态机和验证接口；输入差异仅经现有
`format_question` 适配，image goal 附带用户目标图。运行策略不读取 GT 目标映射、成功距离或官方最短路径。

## 候选与证据

- Verify：按当前稳定 object ID 去重，从真实 snapshot 对象裁剪获取代表证据。保留图像和 capture pose；
  导航目的地为对象周围已知自由网格上的安全观察点，**不是把拍摄点当成对象位置**。
- Explore：真实 frontier 加其已验证 approach Place。没有真实源、图路线或终端已知路径时不进入可选集合。
- 支持分数只来自模型选中候选时给出的 [0,1] 目标匹配/探索相关性，不当作校准概率。
  未评分候选显式标为 null；不凭空为所有对象制造正支持。
- 同一 entity 的相近 capture pose/朝向只留最大支持；跨独立视角也只取最大值，不累加。
  证据记录 source observation、capture Place、pose、look-at、模型分数与 step。
- 不传播跨 Place 概率，不把检测缺失、验证不确定或一次动作失败解释成整个 Place 没有目标。
- 实际对象合并链同步重投影证据与检查视角。新 subtask 创建全新目标层，空间图继续保留。

选择器把全部有可执行视觉源的候选及图路径、connector/terminal 成本交给同一个 VLM。
图像按每页 12 个 tile 拼成联系表，保留稳定 candidate ID；Verify 同时展示上下文和对象 crop。
没有 top-k Region 裁剪。缺少图像/裁剪的候选明确列入 excluded，并记录实际 presented IDs。
联系表缩小图像可能影响细粒度辨识，应在在线结果中观察，不能假定与原始 HGR prompt 等价。

## 路线与状态机

`候选 → 选定 Explore/Verify → valid Place 边逐跳执行 → 局部终端段 → 新观察/验证 → 完成或重新选择`。

路线成本拆为：当前位姿到 origin Place 的已知 connector、valid 图边长度、approach 到安全点的已知局部段。
同 Place 从当前实际位姿计算局部段。终端连接限于 `2 × place_spacing_m + final_observe_distance`，
避免用全局网格路径悄悄绕过 Place 图。此数值为已知图路线估计；实际 HGR 局部轨迹仍须单独测量。

安全点使用 `final_observe_distance`（继承配置 1.5m）附近的自由网格环，允许 0.75 voxel 离散误差。
R4 从环上选取当前 `(entity, approach Place)` 动作的最低成本终端，不把其他相邻环点视为新动作。
`view_spacing_m=0.5` 与 45° 桶继续用于证据去重和 R3 对照。没有类别/场景专用阈值。

选中 intent 在局部运动期间保持；新源映射失效或已知路径不再可行时释放并重新选择。
每次安装下一跳前及移动中的下一轮观测后复查已知网格路径。只在有明确 occupied endpoint
证据时使执行边失效，未知或一次 controller 失败不等于永久断连。
当前版本沿用 HGR 执行步预算，不新增 Region TTL、失败黑名单或多个无进展计时器。

终端到达时读取新的 RGB，并保存为 `object_observations/phase_c_observation_STEP.png`：

- Explore：记录已观察视角，下一轮观察更新空间图与候选。
- Verify：同一个 VLM 比较目标输入、原选中 crop 和当前 RGB，输出 matched / rejected / uncertain。
- 只有 matched 才允许原有 task_success 结束分支。到达 graph waypoint、历史匹配、API/解析失败都不能结束。
- rejected / uncertain 记录对象的已检查视角和反馈，下一次不能原样重试相同观察点；可以尝试其他可达视角。
  不降低整个 Place 的支持。模型请求失败保守记 uncertain，并保留具体错误原因。

本次还修正了原 HGR query 返回选择时使用 `np.empty(3)` 的问题：现在复制原 snapshot 的真实 obs_point，
使此前的 source identity 缓存也有正确拍摄位姿。

## 记录与对照

新事件：`phase_c_decision`、`phase_c_source_resolution`、`phase_c_execution_feedback`、`phase_c_subtask_summary`。
包含候选/裁剪原因、支持证据、路线成本、选择原因、实际验证结果和新观察路径。
原 `place_route_planned` 继续记录每一跳；Phase C 不再把其选择标成 hgr_choice_preserved。
VLM telemetry 的新增 purpose 为 `place_goal_select` 和 `place_goal_verify`，计入现有子任务成本。

`cfg/manifests/place_goal_phase_c_r5_reference.json` 固定 R5 已完成结果与任务清单的 SHA256。
审计脚本在比较前校验基准未变。这冻结的是**历史结果文件**；不能把有未提交修改的当前工作区当作历史 R5 源码归档。
R5 的 Distance SR/SPL 为 64.16%/35.69%，173 subtasks。

评估保持原 Distance/Snapshot 指标定义。Distance 成功是最终位置的评测结果，不等同于模型 matched；
在线报告需同时看 verification trace 与原 SR/SPL，不能混淆两种判定。

## 验证与启动

2026-09-09 的历史离线结果为 210 项通过，包括 R3/R4 行为隔离、Place 状态投影、同 revision 禁止重复请求、
同一拓扑动作不可重复、对象合并后的动作重绑定、缓存关闭/冷缓存完整请求逐字节一致、像素/裁剪更新失效、
路线缓存跨构建失效，以及新目标隔离、重复证据不累加、合并续接、多跳不能提前成功、
uncertain 后换观察点、未知空间/断连排除、长距离网格不能绕过图、同模型三种目标类型选择和验证、
非法模型输出不成功、明确障碍才失效边、Explore 到达后的检查记录，以及审计拒绝未经验证的成功。原有兼容测试继续通过。
配置和主入口通过语法/加载检查；审计脚本对 R5 自比较为 173 个相同 ID、零差值。
这些测试未覆盖本次发现的主流程缺口；本次文档更新未重跑代码测试。

以下为复现历史 C-R4 的命令（线程限制为 16），不是下一版设计启动指令。
当前目录已有完成结果；复跑应先设置独立 exp_name，避免跳过已完成 episode 或混淆历史输出：

```bash
cd /home/hdd/tangyuxin/projects/Hypothesis_Graph_Refinement
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 NUMEXPR_NUM_THREADS=16
CUDA_VISIBLE_DEVICES=7 /home/tangyuxin/miniconda3/envs/hgr/bin/python \
  run_goatbench_evaluation.py \
  -cf cfg/eval_goatbench_place_goal_phase_c_r4_qwen3vl_dashscope_train_smoke.yaml --split 1

/home/tangyuxin/miniconda3/envs/hgr/bin/python scripts/summarize_place_goal_traces.py \
  results/exp_dev_goatbench_place_goal_phase_c_r4_qwen3vl30b_dashscope_train_smoke \
  --expected-subtasks 8
```

验收应检查有效 Explore/Verify、至少一条完整验证链、无未经验证的 task_success，以及
模型失败/无候选分母和重复执行情况。脚本的零 violation 仅是必要条件；如果某机制未触发，不能视作通过该项。

smoke 验收后跑完整开发集：

```bash
for split in 1 2; do
  CUDA_VISIBLE_DEVICES=7 /home/tangyuxin/miniconda3/envs/hgr/bin/python \
    run_goatbench_evaluation.py \
    -cf cfg/eval_goatbench_place_goal_phase_c_r4_qwen3vl_dashscope_train.yaml \
    --split "$split" || break
done

/home/tangyuxin/miniconda3/envs/hgr/bin/python scripts/summarize_place_goal_traces.py \
  results/exp_dev_goatbench_place_goal_phase_c_r4_qwen3vl30b_dashscope_train \
  --expected-subtasks 173
```

GPU 编号按实际空闲卡调整，API 环境沿用原 DASHSCOPE_API_KEY。
12×2 已反复用于开发，不是独立验证集。后续还需要重复运行、按 scene/episode 配对区间和独立场景测试，
不能据离线测试宣称完整双层方法已获得泛化收益。
