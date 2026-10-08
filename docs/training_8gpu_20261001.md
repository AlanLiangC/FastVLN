# 2026-10-01：8 卡重启、教师诊断与持续监督

本报告记录 revision 3。主实验 `runs/streamnav_8gpu_20261001/ealm` 使用 GPU 0–3；固定混合对照 `fixed_mix` 使用 GPU 4–7。传感器保持 480×270、HFOV 120°、相机 0.88 m / 0°、机体 0.88 m / 0.18 m。模型仍只接收 RGB 与目标文本。

## 旧训练为何没有解决问题

上轮恢复实验分别运行到 update 332 / 313，随后都因课程 warmup 内未捕获的 `OracleUnavailableError` 退出。共同失败 episode 为 HM3D-v2 的 `00401-H8rQCnvBgo6` / `5269`；当时的全局异常处理只覆盖 rollout，不覆盖 reset 内 warmup。该路径现已修复：恢复原始合法起点，记录 fallback，不生成假的 oracle 标签。真实 Habitat 故障注入测试验证了恢复后的像素、距离和 frame_id 与原始 reset 一致。

进一步发现教师在极近目标处反复转向。使用同一机器人和同一批 episode，仅改变 success / follower goal radius，做配对审计：

| 集合 | 半径 | 成功数 | 平均动作数 |
|---|---:|---:|---:|
| HM3D-OVON train | 0.10 m | 15 / 24 | 216.29 |
| 同一批 train | 0.25 m | 24 / 24 | 43.58 |
| HM3D-OVON val_seen | 0.10 m | 16 / 24 | 193.83 |
| 同一批 val_seen | 0.25 m | 24 / 24 | 43.25 |

旧半径下，失败的最终距离为 0.1008–0.2069 m，持续到 500 步，许多轨迹包含数百次左右旋转而碰撞很少。离散前进步长是 0.25 m，0.10 m 的教师终止范围会使 follower 在目标附近无法可靠结束。这会将大量无效旋转作为监督，并减少有效 STOP 标签。以上是 48 个配对 episode 的诊断结果，不应外推为全量 oracle 成功率，也不是训练后策略成绩。

原始逐 episode 证据：`runtime/eight_gpu_20261001/oracle_gpu{4,5,6,7}.jsonl`；脚本 `oracle_audit.py` 位于同目录。审计完成后加入 64 步无 geodesic 进展保护，避免继续收集数百步重复方向标签。

为排除“只改阈值就得到好成绩”，对旧主模型 update 325 使用 0.25 m 重新做了 72 个原始起点自主验证：三个 split 各 24 个，**SR 仍全部为 0**，仍然没有 STOP。unseen 只有 1 / 24 曾进入目标区域。结果在 `runtime/eight_gpu_20261001/old_policy_025/`。因此本轮从转换初始化重训，不继承这些失败策略的优化器。

旧恢复实验的 `EXPERIMENT_STATUS.json` 已标记不可恢复、不可混入当前协议比较。负面指标和崩溃证据保留；此前错误碰撞配置的实验已清理，详见首次审计。

## 与 OVSegDT 的可比范围

审阅了 [OVSegDT 官方仓库](https://github.com/CognitiveAISystems/OVSegDT)，本地副本为 `runtime/vendor/OVSegDT`，commit `646a3d53e7eae5879a4ce28d3a4dbaacca52d2b7`。

- [transformer_dagger_ppo_no_segm_loss.yaml](https://github.com/CognitiveAISystems/OVSegDT/blob/646a3d53e7eae5879a4ce28d3a4dbaacca52d2b7/config/experiments/transformer_dagger_ppo_no_segm_loss.yaml) 使用 0.25 m 成功半径；即使配置名叫 no_segm_loss，仍启用 GT segmentation mask 输入。
- 官方策略包含 SigLIP、GPS / compass、前一动作和持续提供的目标 embedding；其 Transformer 与本项目转换的 Qwen/KDA 不同。官方配置训练预算为 1B steps。
- [机器人任务配置](https://github.com/CognitiveAISystems/OVSegDT/blob/646a3d53e7eae5879a4ce28d3a4dbaacca52d2b7/config/tasks/objectnav_stretch_hm3d.yaml) 的视场、分辨率、相机和机体也不同。本轮保留用户指定的机器人与 RGB-only 约束。

因此可借鉴教师和监督机制，不能把现有配置称作 OVSegDT 的严格复现，也不能承诺相同成功率。

## 当前训练与验证协议

每实验四个 learner，同步 DDP，每卡八个训练环境。每 update 新采集 512 transitions，序列长度 4，每卡 minibatch 8 条序列，重放两遍。两组共 64 个训练环境，采用不同 rank seed 扩大采样；两组使用相同 seed 规则、机器人、数据配比和预算，主要对照因素是 EALM / 固定 alpha 0.5。

成功 / oracle 半径为 **0.25 m**。同时报告：

- `success`：在该半径内执行 STOP。
- `success_strict_0_1`：在 0.10 m 内执行 STOP。
- `oracle_success`：自主轨迹曾进入 0.25 m 范围，即使没有正确 STOP；不是教师接管成功率。
- `min_distance_to_goal`：自主轨迹到目标的最近 geodesic 距离。

必须在图表中注明半径，不能将旧 0.10 m SR 与新 SR 拼为同一训练曲线。所有自主验证使用原始起点、argmax、无 oracle 接管，三个 split 各 48 episodes，跨场景固定采样，按 rank 分片后汇总。第 1、25、50、100、200… updates 验证，每 25 次保存，每个 split 导出一条视频。这是诊断子集。

首次 update 1 的三个 split 均为 SR 0，原样保留。随后从同一 checkpoint 切换梯度同步实现以减少通信等待，参数、优化器、每 rank RNG / sampler 状态均恢复；没有用一次无效验证换掉指标。GPU 上显示 100% 也可能包含 NCCL 等待，应结合 update 耗时和实际新样本量解释。具体最新状态以后台监控文件为准。

## Update 25 的自主验证结果

2026-10-01 10:35 UTC 左右完成；每个 split 为原始起点的固定 48 episodes，覆盖 36 个场景，无教师接管。

| 实验 | seen SR / SPL | synonyms SR | unseen SR | 严格 0.10 m SR |
|---|---:|---:|---:|---:|
| EALM 主实验 | **2/48 = 4.17% / 2.71%** | 0/48 | 0/48 | 三组均 0 |
| 固定混合 | **1/48 = 2.08% / 1.96%** | 0/48 | 0/48 | 三组均 0 |

主实验的两个成功样本是 seen 列表 #14（tv）和 #28（couch），分别执行 8 / 13 步，最终距离为 0.133 / 0.115 m，均无碰撞。couch 轨迹曾到达 0.046 m，但最终 STOP 时不满足严格阈值。这说明已经能在部分真实样本中自主接近并停止，仍不能视为长距离导航或开放词汇泛化已经有效。样本很少，不能据此宣称 EALM 显著优于对照。

模型与逐 episode 记录均保留。`checkpoints/best` 指向三组平均自主 SR 更高、同分时平均 SPL 更高的检查点；只选择正 SR，始终保留所有失败记录。默认最近三份 checkpoint 之外，额外保护 best。当前两组 best 均为 update 25。

随后在 update 27 保存并恢复，启用了完成 episode 的并行重置，并增加 `rollout_seconds` / `reset_seconds` / `optimization_seconds` / `global_update_fps` 计时。该改动经过真实轨迹逐项等价性验证，未改变采样 seed、目标或优化目标。后续训练与第 50 次验证继续由后台进程推进。

## Update 50：仍未稳定，继续监督

| 实验 | seen SR / SPL | seen 严格 0.10 m SR | synonyms SR | unseen SR |
|---|---:|---:|---:|---:|
| EALM | 0 / 0 | 0 | 0 | 0 |
| 固定混合 | **2/48 = 4.17% / 2.66%** | **1/48 = 2.08%** | 0 | 0 |

固定混合的严格成功是 couch 样本 #28，最终距离 0.069 m，但耗时 **235 步、碰撞率 94.5%**。因此非零 SR 不等于导航质量已经好；该案例暴露了撞墙和 STOP 过晚的问题。EALM 最新一轮退回零，主实验 best 保持 update 25；固定混合 best 更新为 50。所有负面结果保留，不以 best 替代最近训练曲线。

两组在 update 50 的完整验证后安全保存，从同一检查点继续，所有训练 worker 已使用去除重复渲染 / 距离计算的环境实现。后续第 100 次验证仍使用同一批原始起点。每 30 秒监控不仅检查全零，也明确列出仍为零的 split；seen 出现少量成功不会把 unseen 失败隐藏为健康状态。

## 验证与可复查产物

- `make check unit`：ruff、format、mypy 通过；35 个 CPU / 结构回归测试通过。
- GPU 数学与状态测试：4 项通过。
- 真实 Habitat 集成测试：6 项通过，覆盖 warmup 异常恢复、64 步停滞拒绝及 reset 恢复，以及并行重置与串行重置的 RGB / 动作 / reward / done 逐项精确一致。
- 8 卡 NCCL all-reduce：各 rank 得到预期求和 36。
- 两卡完整批次参考校验：flat all-reduce 和 DDP 的最大梯度误差均为 1.49e-8；包含 rank 间部分未用参数、全局 advantage、停止标志和更新后参数一致性。
- 四卡真实训练、分片验证、保存成功；四卡 checkpoint 从 update 2 恢复到 3 成功。
- DDP 四卡真实 Habitat 连续两次更新、分片验证与保存成功。日志与测试产物在 `runtime/eight_gpu_20261001/` 及 `runs/ddp_smoke_20261001/`。
- 浏览器服务已接主实验 best；真实 reset / policy step 的机器人参数、权重路径和动作响应保存在 `viewer_probe.json`。

DDP 使用 FP32 梯度分桶和 bucket views，通信与反向计算重叠；参见 [PyTorch 2.10 DDP 文档](https://docs.pytorch.org/docs/2.10/generated/torch.nn.parallel.DistributedDataParallel.html)。保留 `distributed.gradient_sync=flat_allreduce` 作为数学参考路径。数学校验可通过 `tools/check_distributed.py` 复跑。

## 持续监督与浏览器查看

```bash
source scripts/env.sh
python tools/training_status.py
cat runs/streamnav_8gpu_20261001/ealm/health_status.json
cat runs/streamnav_8gpu_20261001/fixed_mix/health_status.json
```

两个独立后台 monitor 每 30 秒记录主进程、各 rank、更新推进、全零 SR、动作坍缩、GPU 利用率和显存。非有限 loss、worker 退出或 3,600 秒不推进会要求停止；update ≥500 后最近五轮完整验证全零，也会在保存后停下，要求重新检查，不无限重启。监控还校验 worker 所属 launcher，防止重启时旧 PID 文件造成误停。

浏览器：`http://127.0.0.1:8765`。远程可用 `ssh -L 8765:127.0.0.1:8765 <服务器>`。点击“加载 / 重置”可读取新发布的最佳验证权重；页面也显示最新训练和监控记录。当前 viewer 使用 GPU 7，与对照训练共享少量显存。服务重启命令见 README。

监控进程是实际留在服务器的持续监督机制，不能在 Docker 或驱动退出后继续工作，也不代表自主策略已经达标。后续判断应以新模型的独立验证和实际轨迹为准。


## 性能检查与真实复现视频

初期逐秒采样 120 秒（10:29–10:31 UTC），各卡平均利用率约 32%–59%，显存峰值约 42–45 GiB；没有把短时 NCCL 100% 当成持续有效计算。记录见 `runtime/eight_gpu_20261001/gpu_util_summary.json`。分项计时显示 reset / 课程移动占用了明显时间；该结果用于继续优化，不能宣称 8 卡已达到峰值效率。

环境 step 原来重复渲染 post-action RGB，oracle 又重复计算同一位置的 geodesic 距离。本轮去除重复渲染，在 oracle 查询和原地转向 / 完全碰撞时复用精确位置对应的距离。优化前后的隔离 Habitat 进程在 5 个真实 episode、168 个动作上逐项比较，RGB SHA256、oracle 动作、reward、done、距离和全部 metrics 完全一致，包含课程 warmup。结果见 `env_equivalence.json`。同一进程同时持有两个 Simulator 的首个测试发生 native crash，已弃用该测试布局；生产环境继续采用每个 Simulator 独立进程。计时受同时进行的训练影响，不作为独立吞吐基准。

浏览器已实测 replay seen #14 / #28，两次均自主成功，且步数、距离、SPL 与周期验证一致。视频位于主实验 `evaluation/update_0000025/viewer_replays/`，可通过当前服务访问：

- [TV，8 步](http://127.0.0.1:8765/videos/update_0000025/viewer_replays/split0_sample14.mp4)
- [Couch，13 步](http://127.0.0.1:8765/videos/update_0000025/viewer_replays/split0_sample28.mp4)

这是明确挑选的两个成功复现示例，失败样本保留在页面和完整验证记录中。每条视频旁边的 JSON 保存了实际 checkpoint、每步概率、执行动作与最终指标。


当前进程、配置和监控快照保存在 `runtime/eight_gpu_20261001/final_status.json`，仅代表文件中的时间。长期真实状态请读取两个实验目录的 `health_status.json` 或运行 `tools/training_status.py`；不要把本文快照当作持续增长的实时结果。
