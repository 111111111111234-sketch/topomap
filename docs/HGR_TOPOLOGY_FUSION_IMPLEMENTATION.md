# HGR 与 Place topo 融合

2026-09-10。原 HGR 选择器消费 Place 路线信息，B-R5 执行同一图上的路线；
关闭独立 Phase C Explore/Verify 循环。原 HGR 候选、假设、视觉输入预算和终端协议保留。
历史模式默认不添加新的提示内容。

选择上下文在 `query_vlm_goatbench.py` 中从当前图生成，原预筛选/图片预算之后，
在 `format_explore_prompt` 中按真正展示的 Snapshot/Frontier 编号加入纯文本。
图路线仅用 valid 有向边，frontier approach 与执行器共用选取函数；
Snapshot 使用相同的 capture Place。图不连通标为 no_known_route，不能作为目标不存在证据。
日志保存 `HGR topology selection context` 和实际提示；`place_route_planned` 增加
`hgr_topology_fusion` 标记，继续记录路线、失效和同目标执行进度。

这不是全程 corridor 执行。当前 B-R5 跨 Place 使用边 anchor，HGR 执行最后局部段，
保留原局部安全路径。图距离不是完整运动成本，Snapshot 拍摄点不是对象位置；
这些限制已写入选择提示。在线速度和导航质量尚未验证。

离线验证：以 16 线程运行 `python -m unittest discover -s tests`，231 项通过。
新增 5 项覆盖选择/执行路线一致、断边、筛选及图片预算后的编号、三种目标输入共用原入口、
融合前后图片内容和原提示保持、配置预算继承。已有多跳续行与停止回归继续通过。

## 运行

使用独立时间戳配置和目录，避免覆盖历史输出；沿用已有 API 环境，GPU 按空闲情况调整。
旧进程不会自动切换策略；不要把同一运行前后不同策略的结果拼接为完整实验。

```bash
cd /home/hdd/tangyuxin/projects/Hypothesis_Graph_Refinement
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 NUMEXPR_NUM_THREADS=16
run_name="exp_dev_goatbench_hgr_topology_fusion_smoke_$(date +%Y%m%d_%H%M%S)"
run_cfg="cfg/${run_name}.yaml"
cat > "$run_cfg" <<EOF
extends: eval_goatbench_hgr_topology_fusion_smoke.yaml
exp_name: ${run_name}
EOF
CUDA_VISIBLE_DEVICES=7 /home/tangyuxin/miniconda3/envs/hgr/bin/python \
  run_goatbench_evaluation.py -cf "$run_cfg" --split 1
```

在线验收：实际提示候选编号正确，包含有效路线；原 HGR 选择和终端协议执行，
没有 `place_goal_select` / `place_goal_verify` 独立请求；跨 Place 保持同一目标到真实
终端，不在中间 waypoint 停止；断连/局部回退单独统计。对比原 HGR、B-R5 route-only
和融合配置的 SR/SPL、路径、步数、请求及耗时。smoke 通过后再运行固定 12×2 开发集，
不能由离线通过宣称性能提升。
