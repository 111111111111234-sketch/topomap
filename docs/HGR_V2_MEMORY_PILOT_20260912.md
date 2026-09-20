# HGR V2 单场景空间记忆 Pilot（2026-09-12）

## 目标与状态

该 pilot 只验证 baseline/V2 × cold/warm 四组是否从相同 checkpoint、位姿、朝向和首个后续子任务恢复，以及 difference-of-differences 报告能否完整生成。单场景结果不用于宣称性能提升或泛化。

工程准备已完成：

- memory export smoke 固定为 `ACZZiU6BXLz` episode 0，第一子任务作为共同探索前缀。
- 四组均设置 `preserve_hgr_memory=true`、`single_subtask=true`，只评测 `ACZZiU6BXLz_0_1`。
- 运行脚本在启动前检查 `DASHSCOPE_API_KEY`，避免完成 GPU 初始化后才失败。
- checkpoint 现在通过 pickle persistent ID 将 Open3D PointCloud/AABB/OBB 显式编码为 numpy，加载时恢复为 Open3D geometry；checkpoint schema 升级为 v2，同时仍可读取 v1。
- checkpoint 先完整写入同目录临时文件，再以不可覆盖方式原子发布，序列化失败不再遗留空的正式文件。
- Open3D checkpoint 往返、失败原子性和历史 frontier 图像回退已纳入回归测试；全量 286 项测试通过。
- 为保留失败现场并避开不可覆盖的 provenance 文件，下一次 export 使用新的 `_r4` 输出目录；共享 checkpoint 路径不变。
- export 已成功生成 198,307,524-byte checkpoint；SHA-256 为 `fbb0f6643bba820a08ecf304a520657b5e3ee7a971e3121db5451ac21e6d2e20`。
- 第一轮 baseline-cold 已成功恢复 checkpoint 并完成 VLM 选择，但日志可视化错误地从新结果目录读取历史 frontier PNG，因 `8_1.png` 不存在而停止。现在缺失的 frontier 文件会回退到 checkpoint 中的内存图像，纯可视化缺失不再中断评测。
- 为避开第一轮已生成的 provenance，四组 retry 配置已更新到 `v2_memory_pilot_aczz_20260912_r2`。

## 配置与产物

- export 配置：`cfg/eval_goatbench_hgr_dual_topo_v2_memory_export_smoke.yaml`
- checkpoint 目录：`results/hgr_v2_memory_checkpoints_smoke_aczz`
- 四组配置和脚本：`cfg/generated/v2_memory_pilot_aczz_20260912_r2`
- 预期报告：`results/v2_memory_pilot_aczz_20260912_r2_comparison.json`

沙箱内首次尝试因不可访问 GPU/EGL 停止；沙箱外第二次尝试已进入场景和感知，但在首个 VLM 请求前因缺少 API 环境变量停止。用户终端的第三次尝试暴露 Open3D CUDA PointCloud 无法由标准 pickle 处理的问题，现已修复。该次遗留的 0-byte 文件已保留为 `00062-ACZZiU6BXLz_ep_0.pkl.failed-zero-byte-20260912`，不计入 pilot。

## 待运行命令

在已经设置 `DASHSCOPE_API_KEY` 的终端执行：

```bash
cd /home/hdd/tangyuxin/projects/Hypothesis_Graph_Refinement
bash cfg/generated/v2_memory_pilot_aczz_20260912_r2/run.sh
```

四组结束后运行：

```bash
/home/tangyuxin/miniconda3/envs/hgr/bin/python scripts/hgr_v2_experiments.py compare \
  --manifest cfg/manifests/goat_train_v7_1_d_smoke_aczz_ep0_seed77.json \
  --start-subtask 1 \
  --baseline results/v2_memory_pilot_aczz_20260912_r2_baseline_warm_r1_s1 \
  --candidate results/v2_memory_pilot_aczz_20260912_r2_v2_warm_r1_s1 \
  --baseline-cold results/v2_memory_pilot_aczz_20260912_r2_baseline_cold_r1_s1 \
  --candidate-cold results/v2_memory_pilot_aczz_20260912_r2_v2_cold_r1_s1 \
  --output results/v2_memory_pilot_aczz_20260912_r2_comparison.json
```

## 验收

- 四组 checkpoint SHA-256、`first_subtask=1`、起点和朝向完全相同。
- 每组只有一个相同 subtask，指标均为有限值。
- 报告包含 SR、SPL、路径长度和 wall time 的 memory interaction。
- pilot 只判定实验链是否有效；是否扩大到 12×2 × 3 repeats 在结果复核后决定。

## R2 单场景结果

四组均完整结束，且 checkpoint SHA-256、`first_subtask=1`、起点和朝向一致；memory provenance 验证通过。V2 cold/warm 的在线 trace contract 均通过，分别记录 1/2 次运动确认，路线违规均为 0。本次没有 hypothesis 验证或撤销事件，因此 cascade/recovery 为 `not_covered`，不视为失败。

| 组别 | distance SR | distance SPL | 路径 (m) | 时间 (s) | 最终 snapshot 选择 |
|---|---:|---:|---:|---:|---|
| baseline cold | 1.0 | 0.1671 | 0.5385 | 33.11 | towel 193 |
| baseline warm | 1.0 | 0.2846 | 0.3162 | 32.50 | toilet 190 |
| V2 cold | 1.0 | 0.3182 | 0.2828 | 36.12 | toilet 190 |
| V2 warm | 1.0 | 0.1443 | 0.6236 | 54.31 | towel 193 |

报告中的 warm 主比较为：SR 差 0，SPL 差 -0.1403，路径增加 0.3074 m，时间增加 21.82 s。memory difference-of-differences 为：SR 0、SPL -0.2913、路径 +0.5631 m、时间 +18.80 s，因此单次 performance gate 未通过。

这不是可推广的负面结论。仅一个 scene、一个 subtask、一个 repeat 时 bootstrap CI 退化为点值；而没有行为差异设计的 baseline cold/warm 已分别选择 towel/toilet，直接证明 VLM 随机波动足以改变本次路径。该 pilot 能确认的是 checkpoint、cold/warm 隔离、指标汇总和 trace 审计链可运行，不能确认 V2 提升或退化。

产物：

- `results/v2_memory_pilot_aczz_20260912_r2_comparison.json`
- `results/v2_memory_pilot_aczz_20260912_r2_v2_cold_trace_audit.json`
- `results/v2_memory_pilot_aczz_20260912_r2_v2_warm_trace_audit.json`

下一步先做同一 checkpoint 的 3 repeats 稳定性检查，而不是立即进入昂贵的 12×2 × 3 正式评测。若方向仍不稳定，再固定/缓存 VLM 决策；若均值方向稳定且结构审计继续通过，再扩到 `goat_train_fast_3x2_seed77.json`。

已生成稳定性检查脚本：

```bash
bash cfg/generated/v2_memory_pilot_aczz_20260912_r3_repeat3/run.sh
```

完成后汇总：

```bash
/home/tangyuxin/miniconda3/envs/hgr/bin/python scripts/hgr_v2_experiments.py compare \
  --manifest cfg/manifests/goat_train_v7_1_d_smoke_aczz_ep0_seed77.json \
  --start-subtask 1 \
  --baseline results/v2_memory_pilot_aczz_20260912_r3_repeat3_baseline_warm_r1_s1 results/v2_memory_pilot_aczz_20260912_r3_repeat3_baseline_warm_r2_s1 results/v2_memory_pilot_aczz_20260912_r3_repeat3_baseline_warm_r3_s1 \
  --candidate results/v2_memory_pilot_aczz_20260912_r3_repeat3_v2_warm_r1_s1 results/v2_memory_pilot_aczz_20260912_r3_repeat3_v2_warm_r2_s1 results/v2_memory_pilot_aczz_20260912_r3_repeat3_v2_warm_r3_s1 \
  --baseline-cold results/v2_memory_pilot_aczz_20260912_r3_repeat3_baseline_cold_r1_s1 results/v2_memory_pilot_aczz_20260912_r3_repeat3_baseline_cold_r2_s1 results/v2_memory_pilot_aczz_20260912_r3_repeat3_baseline_cold_r3_s1 \
  --candidate-cold results/v2_memory_pilot_aczz_20260912_r3_repeat3_v2_cold_r1_s1 results/v2_memory_pilot_aczz_20260912_r3_repeat3_v2_cold_r2_s1 results/v2_memory_pilot_aczz_20260912_r3_repeat3_v2_cold_r3_s1 \
  --output results/v2_memory_pilot_aczz_20260912_r3_repeat3_comparison.json
```

## R3 三重复稳定性结果

12 个短评测全部完整结束，所有 run 均恢复同一个 checkpoint。六个 V2 run 的 trace contract 全部通过，共 7 次运动确认、0 路线违规；本轮仍未触发 hypothesis 验证/撤销，因此不新增 cascade 覆盖。

warm 主比较的三重复均值为：distance SR 1.0 对 1.0；baseline/V2 SPL 为 0.2063/0.3182；路径为 0.4644/0.2828 m；时间为 31.69/40.04 s。difference-of-differences 为 SPL +0.0188、路径 -0.0395 m、时间 +0.89 s。自动 distance-based performance gate 显示 `true`，但该结果不能作为性能通过结论。

原因是 image-goal 的目标身份指标与距离指标冲突：baseline warm 有 2/3 次选择正确 towel 193，V2 warm 为 0/3，三次都选择 toilet 190；但 toilet 的终点仍落在 GT 距离阈值内，因此 distance SR 仍为 1，并因路径更短获得更高 SPL。V2 prompt 中 towel 路线约 0.6414 m、toilet 路线约 0.2828 m，稳定选择较近的错误对象，说明路线代价正在干扰视觉身份判断，而不只是随机波动。

因此当前决策为：

- checkpoint、cold/warm 隔离、路线执行和审计链验收通过；
- V2 性能验收不通过，不能扩到 3×2 或 12×2；
- 下一修复应把 image/description 目标的语义身份判定设为路线代价之前的硬优先级，路线只在语义等价候选之间决胜；
- 后续 gate 必须同时报告并约束 `success_by_snapshot`，不能允许较近的错误对象仅凭 distance SR 通过。

## R4 语义优先修复

已完成两层修复：选择提示隐藏 Snapshot/Object 的路线成本并声明视觉/语义身份为硬约束；
比较报告新增 snapshot SR/SPL，performance gate 同时约束 identity regression。旧 R3 用新口径
重算后 snapshot SR 为 0.667 对 0，gate 正确变为 `false`，报告保存为
`results/v2_memory_pilot_aczz_20260912_r3_repeat3_comparison_identity_gate_v2.json`。

最小在线验证只重跑 V2 warm 三次，复用 R3 baseline warm：

```bash
bash cfg/generated/v2_memory_pilot_aczz_20260912_r4_semantic_guard/run.sh
```

### R4 在线结果

三次 V2 warm 全部完成，均选择正确 towel 193，snapshot SR 由旧 V2 的 0/3 恢复为 3/3；
baseline warm 参考为 2/3。三次 V2 路线完全一致：2 steps、0.6236 m、distance SPL 0.1443，
平均 54.07 s。与 baseline warm 三重复均值相比，snapshot SR +0.3333、snapshot SPL
+0.0329，但 distance SPL -0.0620、路径 +0.1592 m、时间 +22.38 s，因此新 gate 仍为
`false`：语义回归已修复，但尚无总体性能提升。

三条在线 trace 均通过，共 6 次运动确认、0 路线违规。认证路径从 `[46,60]` 到 `[40,59]`
包含已知自由空间中的转折，执行器分两段确认到达；不能为了匹配 legacy 的单步短路径而跳过
可见性认证。当前应把额外路径/感知时间视为安全路线的待评估代价，而不是直接放宽路线约束。

正式报告：`results/v2_memory_pilot_aczz_20260912_r4_semantic_guard_comparison.json`。
