# V2-A：对象 approach 与完整路线成本

当前重建主线见 [双层动态 Topo Map 导航重设计与开发计划](HGR_DUAL_DYNAMIC_TOPO_NAVIGATION_REDESIGN.md)。
新主线使用 `navigation_backend: dual_dynamic`，本文件以下 V2-A/R4 内容作为几何组件与历史实验依据保留。

后续统一架构已新增独立 `hierarchical_navigation` 模式，见
[统一分层导航器实现与运行说明](HGR_HIERARCHICAL_NAVIGATOR_IMPLEMENTATION.md)。该模式将
语义硬门控、拓扑任务规划、同 Place TSDF 直达、跨 Place 认证执行和事件触发终点验证纳入
一个状态机；旧 V7 与 V2-D 配置保持不变。在线性能尚待按说明中的顺序验收。

后续开发：新增统一路线与目标层的独立 B/C/D 配置，详见
[开发与用户运行手册](HGR_DUAL_TOPO_V2_BCD_IMPLEMENTATION.md)。
本轮遵照用户要求不运行测试或在线评测；以下 R2 待验收状态未因此改变。

## R2：有效路线续行修正

首轮失败后已修复：已有 approach 优先重新认证，只有不可认证才搜索其他 approach；
成本浮动不再自动更换接近 Place。对象中心/终端实际改变时仍允许重新解析。
Frontier 的已安装跨 Place hop 在真实源存在、边有效、当前到原安装 anchor 的已知网格路径
有效时，不被 choose_every_step 清空；hop 到达才推进执行进度。源消失/边失效/局部路径
阻断仍允许恢复重规划。原 same-place frontier 选择频率保持；没有将完整探索改成新的事件驱动策略。

新增 `v2_hop_validation` trace（retained、seconds）、`v2_object_approach.geometry_query_seconds`，
approach 结果记录 route_event 与 route_query_seconds；日志 `V2 candidate geometry` 记录
候选构建总时间和其中路线查询时间。后者是子集，不能和总时间相加；耗时字段不进入模型输入。
这些计时尚不覆盖全部感知、图片准备与局部执行，不能宣称完整互斥性能剖析。

16 线程完整回归 242 项通过，包含实际主循环重置分支、有效 hop 保留/阻断释放、
更便宜替代不抢占、原 approach 失效恢复、执行已达 Place 与观测 Place 分离。
R2 在线尚未运行；新配置 `cfg/eval_goatbench_hgr_dual_topo_v2_a_smoke_r2.yaml` 使用独立目录。
下述时间戳命令中的 extends 可改为该 R2 配置名；旧进程不会自动加载修改，首轮结果保留。

在线更新：首轮 smoke 已完成，见 [复核](HGR_V2_A_SMOKE_REVIEW.md)。Distance SR 62.5%，
SPL 43.90%，耗时 34:51；当前闭环验收不通过，先修有效 frontier 路线的续行稳定性。
下文“未运行”是开发时的离线状态，不代表最新在线进度。

实现状态：代码已接入，16 线程完整 unittest 239 项通过；本轮在线 smoke 未运行。
本轮不启用 V2-B 连续 guidance、V2-C 目标状态或新的停止协议。

`src/object_approach.py` 直接复用 HGR 的 `get_proper_observe_point`，将输入自由空间
限制为 island、unoccupied 与非 occupied 的交集。纯查询无导航状态写入、不调用 Pathfinder，
不会生成按方向逐次 Verify 的候选。生成一个对象终端后，比较可用 approach 的完整已知路径。
每条候选路线沿 B-R5 实际使用的 edge target anchors，计入当前位姿到首 anchor、
中间 anchor 连接和真实终端段；每段只计一次，保存 certified_path 和 segment_distances_m。
比较的是各 approach 对应图最短路线的认证完整成本，并非所有可能几何路线的全局最优解。
局部终端段沿用 `2 × place_spacing_m + final_observe_distance` 限制；跨边段限制复用
相关边长度和 Place 尺度，不能把全岛连通当作无限局部捷径。

原 HGR query 为每个真实 crop 对象计算结果；原预筛选/图片预算后，重新绑定 Object 标签，
提示只展示对应对象的结果，完整网格路径不上传。frontier 同样使用完整终端路线估计。
没有增加模型请求、图片或候选筛选规则；成本和目的地变化可能改变模型选择，不能声称选择输出不变。

选中对象携带同一 approach 结果，主入口在执行前复查；对象中心未变时保留已选终端，
多跳续行根据执行已达 Place 重新认证，不因观测 Place 归属不变重走已完成边。
真实 alias 合并后沿已有 source resolution 取得新 ID，再重绑定终端查询。
到 approach 后安装该对象终端 override，不再恢复“先到 capture Place 再生成另一个终端”。
若中心估计变化则重新生成终端；若无认证路线则显式回到原同目标局部执行，
不生成 capture Place 路线。`v2_object_approach` 事件保存结果、路径与 fallback 标记；
原 waypoint 失败回退诊断仍保留。

边界：当前 HGR 局部执行仍可能调用原 Pathfinder，认证路径是成本估计和路线规划证据，
不是对其实际轨迹的 corridor 保证。未知目标观察点、多个对象的非标准 Snapshot 或
不可认证几何保留原局部回退。观察点来自已知自由空间不等于已经证明 RGB 可见/目标匹配。
历史 task 3/5 用于诊断；本轮未用缺失的历史已知网格或未来地图伪造冻结重放。
尚未证明在线消除折返、提升 SPL 或减少耗时。

## R4：目标身份优先保护

单场景 memory pilot 的三重复暴露了选择层混淆：V2 warm 三次均选择路线更短的错误
toilet，而 baseline warm 有 2/3 次选择正确 towel；错误对象仍落在 distance 阈值内，导致
旧 performance gate 被误判为通过。修复后，Snapshot/Object 的完整认证路线仍在运行时保留，
供选中后的同一路线执行使用，但选择提示只显示对象标签对应关系与 route status，不再显示
对象的 `total_distance_m`、route places、terminal 或 approach Place。Frontier 路线信息继续显示，
只在没有 Snapshot 能识别目标时辅助探索。提示明确规定视觉/语义身份为硬约束，禁止用更短路线
替换更可信的身份匹配。

比较报告新增 `snapshot_sr` 与 `snapshot_spl`；performance gate 现在同时要求 distance SR 和
snapshot identity SR 的 95% CI 下界不低于 -0.02。用新口径重算旧 R3 后，snapshot SR 为
0.667 对 0，gate 从错误的 `true` 修正为 `false`。定向与完整回归共 288 项通过。

R4 三次在线复测均恢复正确 towel 193，snapshot SR 达到 3/3，证明路线成本污染身份选择的
修复生效。认证路线均以两段、0.6236 m 完成且零违规；相对 baseline warm 均值，身份指标改善，
但 distance SPL、路径和耗时退化，新 gate 仍为 `false`。这一区分了“语义选择修复完成”和
“整体性能已经提升”；后者仍需多场景验证，不能由本单场景宣称。

## 新配置与启动

`eval_goatbench_hgr_dual_topo_v2_a_train.yaml` 继承融合 train，仅开启
`hgr_topology_fusion.object_approach`；smoke 使用相同八任务 manifest。
原选择协议、观察预算、停止标准、模型/图像配置和历史输出保留。

```bash
cd /home/hdd/tangyuxin/projects/Hypothesis_Graph_Refinement
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 NUMEXPR_NUM_THREADS=16
run_name="exp_dev_goatbench_hgr_dual_topo_v2_a_smoke_$(date +%Y%m%d_%H%M%S)"
run_cfg="cfg/${run_name}.yaml"
cat > "$run_cfg" <<EOF
extends: eval_goatbench_hgr_dual_topo_v2_a_smoke.yaml
exp_name: ${run_name}
EOF
CUDA_VISIBLE_DEVICES=7 /home/tangyuxin/miniconda3/envs/hgr/bin/python \
  run_goatbench_evaluation.py -cf "$run_cfg" --split 1
```

GPU 按空闲卡调整，沿用已有 API 环境。先核对有效 approach、fallback 分母、
crop/对象 ID 与终端一致性、多跳结束后的实际终端，再对比 baseline/旧融合的全部八任务。
保留失败、起点差异及请求成本，不能只报告成功任务或减少的 waypoint 数。
