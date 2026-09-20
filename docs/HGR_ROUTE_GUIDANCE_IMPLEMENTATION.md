# 目标覆盖修正与连续 route guidance：实现说明

2026-09-10。已接入可运行代码，在线 smoke 尚未运行；历史输出未改动。

## 实际接入

- `goal_observation_coverage.py`：以实体周围的八个方向区域记录检查覆盖，复用已有 45° 离散。
  同一实体同方向跨 Place 去重；同一 Place 的其他方向允许观察。安全环只用于寻找该区域可达点。
  这是有限方向近似，尚不是从 RGB-D 精确恢复的遮挡/可见面聚类；`uncertain` 仍不能作为整个区域不存在目标的证明。
- 检查版本分别保留独立来源位姿桶和目标附近自由网格摘要。文件重命名、时间戳或地图变化
  不重新开放语义已检查方向；uncertain 仅在新增独立来源视角时重开，rejected 保存该方向的
  矛盾记录，其他方向仍可验证。自由网格摘要用于执行失败恢复；几何开放新可达、未检查方向
  时仍可生成新动作。无有效 Verify 时保留 Explore；重复来源不累加 support。
  所有当前 snapshot 的来源视角增量登记，模型 support 仍只在被选择时取得；未评分仍为 null。
- 执行错误单独保存为当前版本的 failure；不进入语义 checked，也不永久删对象或图边。
  合并时通过实际 alias 迁移证据和覆盖，修正原 checked tuple 的实体索引。
- 选择器对 malformed JSON/非法 ID 等语义输出最多尝试 3 次；底层已有 transport 重试，返回 None 后不再倍增。
  记录尝试数、失败阶段和截断响应。验证请求耗尽返回 `request_failed`，显式结束为错误，不伪装成 uncertain。
- 选择返回 `action=selected/continue/no_action/error`；active intent 继续执行，同 revision 的已选响应可复用，
  复用前重新校验真实视觉源。重试在请求适配器内完成，不消耗额外导航步。
- `route_guidance.py`：在当前已知自由网格恢复有序边 anchor 的路径证据，限制边连接的局部搜索范围。
  认证路径及对角边的必要自由格组成保守窄 corridor；前视点只从路径前缀上选择，执行直线必须通过 supercover
  检查，并保持 transition 顺序。最大前视长度使用原 HGR 动作步长；一次运动可穿过多个 anchor，不要求节点停顿。
- `TSDFPlanner.agent_step(route_guidance=...)` 真正消费 guidance 并输出通过认证的直线段，
  不调用 `get_distance`/Habitat Pathfinder，不再在认证后用独立位置调整把终点移出路径。
  到达真实终端才返回 `target_arrived`，保留终端朝向及每步感知。
- 阻塞先尝试原 corridor 内局部恢复，再保持目标重求路线；无法认证的边仅在本次查询中排除，
  不永久修改空间图。全部候选路线都失败时记录 `corridor_unavailable`，返回目标层。
- 无损度二链 `structural_route_view` 保留完整边展开，frontier 接入和分支不收缩。
  它是第一层路线查询/展示视图；当前 Dijkstra 仍使用原图，未更改 Place 采样或宣称图搜索加速。

## 诊断与实验边界

新增 `route_guidance_installed`、`route_guidance_local_repair`、`route_guidance_step`，
保存认证路径、transition、进度和实际执行段。现有 `place_route_planned` 增加 execution_mode，
其 next_hop 字段仍表示图路线，不表示 guidance 在安装单点 waypoint。
审计重放执行段是否落在认证 corridor 内；此审计不能替代在线真实碰撞/可见性验证。

Verify trace 增加选中候选、实际位置/yaw 和终端误差。历史 matched/Distance 不一致未因代码改动
被证明解决，必须观察新运行的 current RGB、对象身份与轨迹；在线禁止使用 GT 修补判断。

`correctness` 与 `route_guidance` 配置共用目标层、模型、图片输入、Verify/停止协议；只切换执行器，
用于隔离 guidance 的效果。R3/R4 配置不启用新 coverage 或 guidance；公共解析诊断和合并索引修复仍共享。

宽 corridor 优化器、由视觉精确估计独立观察面、全量/增量 VLM 批量打分，以及几何生成 doorway 节点
仍为后续独立消融，不混入当前配置。当前实现不预先保证 SR/SPL 或总耗时提升。

## 启动与验收

先比较以下两个独立 smoke，均为同一 manifest 的 8 个 subtask。完整开发配置后缀为 `_train.yaml`，
仍使用固定 12×2。新目录避免覆盖历史 C-R4。

```bash
cd /home/hdd/tangyuxin/projects/Hypothesis_Graph_Refinement
conda activate hgr
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 NUMEXPR_NUM_THREADS=16
# 当前终端的 DASHSCOPE_API_KEY 沿用已有设置；GPU 编号按运行前实际空闲情况选择。
CUDA_VISIBLE_DEVICES=7 python run_goatbench_evaluation.py \
  -cf cfg/eval_goatbench_phase_c_correctness_smoke.yaml --split 1
CUDA_VISIBLE_DEVICES=7 python run_goatbench_evaluation.py \
  -cf cfg/eval_goatbench_route_guidance_smoke.yaml --split 1

python scripts/summarize_place_goal_traces.py \
  results/exp_dev_goatbench_route_guidance_smoke --expected-subtasks 8
```

最新 Phase C 修复完整离线测试 `python -m unittest discover -s tests`：226 项通过，线程为 16。
新增回归覆盖局部几何不解封语义检查、rejected/uncertain 区别、执行失败恢复、
无关空间图更新不触发语义 revision，以及验证机会耗尽后保留 Explore。观察预算、停止协议
和历史运行结果未修改；本轮在线 smoke 尚未运行。
离线验证包括真实 agent_step 无 Pathfinder、跨 portal 不提前停止、墙/未知空间/对角穿角禁止、
同目标绕行且不永久删边、覆盖更新与实际合并调用链、有限解析重试、请求错误不是语义观察、
以及篡改轨迹被审计拒绝。在线收益必须另报，不能由离线通过代替。
