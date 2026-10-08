# 480×270 机器人参数更新与验收

> 历史工程验收记录（2026-09-30），不代表导航学习有效。旧训练已停止，当前配置、失败诊断与清理情况以 [2026-10-01 训练审计](training_audit_20261001.md) 为准。
本轮按用户指定的传感器和机体参数重新训练；原始实验保留，不混用两套传感器的训练/验证曲线。

## 实际生效的参数

| 参数 | 数值 |
|---|---:|
| RGB 宽 × 高 | 480 × 270 |
| HFOV | 120° |
| 相机高度 | 0.88 m |
| 相机俯仰 | 0° |
| 机体高度 | 0.88 m |
| 机体半径 | 0.18 m |

配置入口为 `configs/config.yaml`；训练、验证、可视化共用。仿真器 RESET 返回实际 sensor/agent/navmesh 参数，真实集成测试检查其数值。视觉预处理完整保留原始 RGB，上下各补 9 行到 480×288，135 个合并视觉 token；不把宽屏图像拉伸成方形。

提供的原导航网格实际按照 1.5 m 高、0.10 m 半径生成，AgentConfiguration 的尺寸并不会改变它。本次按 0.88 m / 0.18 m 重建并检查 pathfinder 网格，文件存入 `runtime/cache/navmesh`，使用跨进程文件锁和原子发布复用。未改写提供的场景文件。四动作、120° FOV 和该机体碰撞设置属于自定义机器人配置，不声称与官方 Habitat benchmark 设置等同。

## 实测效率与选择

两张 GPU 均为 48 GiB RTX 4090。使用真实 Habitat rollout，固定连续序列长度 4、16 rollout steps、2 replay epochs；每个候选先 warmup，再测 2 次 update。此处 FPS 为优化阶段 replay frames/s，包含前后向和 optimizer，不能当作整个实验的在线环境采样速率。调优过程中没有启动合成 GPU 保活负载。

| 并行环境 | 序列 batch | 梯度检查点 | 优化 FPS | GPU 总显存峰值 |
|---:|---:|---|---:|---:|
| 8 | 4 | 开 | 10.11 | 20,342 MiB |
| 8 | 8 | 开 | 16.01 | 21,392 MiB |
| 8 | 16 | 开 | 21.44 | 24,430 MiB |
| 16 | 32 | 开 | 21.49 | 35,797 MiB |
| 16 | 64 | 开 | 19.46 | 48,478 MiB |
| 8 | 8 | 关 | 26.36 | 39,101 MiB |
| 8 | 16 | 关 | OOM | 已排除 |
| **8** | **8** | **关，批量梯度统计** | **28.11** | **39,100 MiB** |

选择最后一行：在已测候选中最快，且为可视化和场景切换留出约 10 GiB。相对于本轮最初的 batch 4 配置，优化吞吐提升约 2.78 倍；调优采样中的平均 GPU 利用率从约 27.1% 提升至 46.3%，并非承诺 GPU 始终满载。更大的 batch 没有带来更高实际效率，不能仅用显存占满程度判断性能。

采用 fused AdamW 和 foreach 梯度范数统计，关闭 activation checkpoint 重算。BF16 compute、FP32 master parameters/moments 与全模型联合训练保持不变。多 GPU 加载时显式设置目标 CUDA device，修复 FLA 初始化误在默认 GPU 上分配小张量、导致另一张 GPU 快满时加载失败的问题。

在 1/8/16 帧推理批量中，recurrent 均快于 chunk，故保留 recurrent inference，训练仍用可求导的 chunk kernel。验证改成每 split 3 个环境并行，episode 结束后从活动批量移除；串行/并行测试的逐 episode 导航结果一致。推理等待时间和整批吞吐分别记录，避免混淆指标。

原始报告：

- `runtime/reports/sensor_training_tuning.json`
- `runtime/reports/sensor_training_unfused.json`
- `runtime/reports/sensor_training_largebatch.json`
- `runtime/reports/sensor_training_no_checkpoint.json`
- `runtime/reports/sensor_training_final_candidate.json`
- `runtime/reports/sensor_inference_tuning.json`

`tools/tune_training.py` 支持重复性能调优；其输出明确标记 `benchmark_not_training`，不会保存调优模型为训练成果。

## 验证证据

22 项 CPU 单元/回归、4 项 GPU 回归、2 项真实 Habitat 集成，共 **28 项通过**；ruff、format、mypy 同样通过。集成验证包括实际相机/机体/导航网格参数、真实 oracle 到达、vision/KDA/actor/critic 梯度、矩形图像联合优化，以及 2 环境并行评估与串行评估逐 episode 一致性。3 个 episode 用 2 个环境运行，覆盖不足一个完整批量的结尾情况。

- `runtime/logs/sensor_release_validation.log`：完整检查与真实训练保存、视频验证。
- `runs/sensor_smoke`：新参数下的 2 环境训练、checkpoint、3 环境短程评估与 MP4。
- `runs/sensor_resume_smoke`：恢复 optimizer/scheduler/RNG/sampler 后继续第 2 次 update。
- `runtime/logs/sensor_resume.log`：真实断点恢复日志。
- `runtime/logs/sensor_viewer_verified.log`：Playwright 验证模型单步、480×270 原图和页面参数；无 JavaScript 错误。
- `docs/assets/viewer.png`：新视野的真实交互截图。

500 步策略基准使用 480×270 随机 RGB、转换初始化权重、BF16，在 GPU 1 上独立测量；同时另一张 GPU 做测试，未启动保活负载。`runtime/reports/latency_sensor480x270_500.json` 记录：prefill 26.0 ms，P50/P95/P99 34.29/37.47/41.68 ms，28.78 Hz；递归 state 固定 32,120,832 bytes，末/首 50 步延迟比 1.0065，峰值 allocated VRAM 1,847,518,720 bytes。此基准不含渲染/RPC，不能用于声称模型导航成功。

## 新训练与可视化

新主实验目录：`runs/streamnav_sensor480x270`，GPU 0。

新固定 IL/PPO 混合对照：`runs/streamnav_sensor480x270_fixed_mix`，GPU 1。

两者从 KDA 转换初始化重新开始，默认 8 环境、16 rollout steps、sequence batch 8、10,000 updates。第 1 次更新保存并验证，之后每 25 updates 保存、每 50 updates 对三组各 3 episodes 验证，单 episode 上限 500。渲染跟随各自 learner GPU。PID、有效配置、指标以各 run 文件为准。

两组均已完成 update 1 的三组验证，并继续更新。首次三组 SR/SPL 均为 0，仍是训练早期；主实验三个 split 的 SoftSPL 为 0.24631 / 0.25653 / 0.05145。训练热身后记录约 37–40 replay frames/s；一次同步采样的 GPU 利用率为 77% / 78%，这是训练阶段快照，不是全程平均值。`runtime/reports/sensor_restart_status.json` 保存进程存活、有效参数、实际指标、GPU 与网页 checkpoint 的最终核验快照。

旧主实验 update 50、旧固定混合对照 update 72 已保存退出。新旧目录与曲线完全分开；恢复代码拒绝混用不同相机或机体配置。小样本快速验证只用于监控，不是完整 benchmark；新模型仍需持续训练。

网页 `http://127.0.0.1:8765` 显示真实 480×270 场景和生效参数，支持模型单步/自动运行与手动控制。视频也保持原图比例，附加状态栏；浏览器图像不拉伸。服务加载的权重版本以 `/health` 或页面为准，重启后才能切换新 checkpoint。启动/恢复/SSH 转发命令见项目 README。

本轮交付时，网页已加载新主实验的 `update_0000001`，Playwright 再次验证模型单步成功且没有 JavaScript 错误，日志在 `runtime/logs/sensor_viewer_main_verified.log`。

训练启动器继续默认启用 GPU 监控和独立保活守护。保活只在 learner GPU 连续低于 50% 达 2 小时时触发 15 秒合成矩阵计算，日志单独标记，不计入上面的优化收益或模型指标。
