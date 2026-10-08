# 初始工程验收记录（2026-09-30）

> 历史工程验收记录（2026-09-30），不代表导航学习有效。旧训练已停止，当前配置、失败诊断与清理情况以 [2026-10-01 训练审计](training_audit_20261001.md) 为准。
本文件保留最初 224×224 / HFOV 79° 配置的证据；当前 480×270 / HFOV 120° 参数与新训练状态见 [传感器更新记录](sensor_update_20260930.md)。

本记录对应真实 RTX 4090、Qwen3.5-0.8B 权重和 HM3D Habitat-Sim 环境，不使用合成导航环境替代集成测试。训练仍在早期进行，工程通过不代表导航模型已经收敛。

## 已完成的闭环

| 项目 | 证据 |
|---|---|
| 原始 Qwen 权重转换 | `checkpoints/qwen35_0p8b_kda/kda_layout.json`；替换 6 层，保留 18 层 GDN |
| 实际模型 forward/backward | `runtime/logs/model_probe.log`；vision/KDA 梯度非零 |
| 真实 Habitat oracle | `runtime/logs/habitat_probe.log`；41 steps，success=1，SPL≈0.9397 |
| 单环境联合训练 + 保存 + 评估视频 | `runs/smoke/` |
| 两环境 rollout、episode 截断与重置 | `runs/batch_smoke/`，2 updates |
| optimizer/scheduler/sampler 断点恢复 | `runs/resume_smoke/`，从 update 2 完成 update 3 |
| 完整混合数据训练 | `runs/streamnav_mixed/`，已完成至少 30 updates，持续运行 |
| 固定 IL/PPO 对照 | `runs/streamnav_fixed_mix/`，已完成至少 50 updates，持续运行 |
| 定期评估 | 两个实验已完成 update 10、20 的三组 OVON 验证，主实验另完成 update 30；JSONL 与 MP4 保留 |
| 浏览器真实交互 | Playwright 加载 episode、模型单步；无 JavaScript error；`docs/assets/viewer.png` |
| 保活守护 | 2 秒诊断 burst 完成 1,709 次 GEMM；单独标记 synthetic keepalive，未计入模型测试 |

测试结果：**20 项 CPU 单元/结构回归 + 4 项 GPU 回归 + 2 项真实 Habitat 集成 = 26 项通过**。此外 ruff、format check、mypy（51 个 source 文件）、shell 语法检查和 uv lock 校验通过。GPU 的 PyTorch JIT deprecation warning 不影响测试；没有跳过要求真实环境的两项集成测试。

日志：`runtime/logs/release_checks.log`、`tests.log`、`integration_final.log`、`meta_loader_test.log`。Meta checkpoint loader 的非持久 rotary buffer 已显式重建，测试验证加载后 episode 隔离；避免重新随机初始化整个 0.8B 模型造成的启动开销。

## 500 步系统基准

`runtime/reports/latency_500.json` 保存完整逐步记录，设备 RTX 4090，BF16，224×224 RGB，转换初始化权重，随机图像。保活诊断发生在这份基准完成之后。

| 指标 | 实测 |
|---|---:|
| Prefill | 26.585 ms |
| P50 / P95 / P99 | 34.361 / 36.613 / 39.655 ms |
| Vision P50 | 8.501 ms |
| Recurrent P50 | 25.587 ms |
| Actor/Critic P50 | 0.219 ms |
| Decision rate | 28.828 Hz |
| State bytes，全部 500 步 | 32,120,832 |
| 末 50 / 首 50 步延迟比 | 1.02135 |
| Peak allocated VRAM | 1,841,356,800 bytes |

这是策略计算基准，不包含 Habitat 渲染/RPC，不是任务成功率测试。GPU kernel 测试另行比较 chunk/recurrent/reference 数值、跨 chunk 的初始 state 梯度和 cache boundedness。

## 数据检查与真实修复

发现未过滤 HM3D-v1/v2 训练标签与 OVON unseen 的交集为 `plant`。混合训练 manifest 显式排除两者合计 1,369,961 条 episode，剩余三源总计 16,709,509 条。过滤前原始数据保留；过滤后训练场景并集与全部验证场景无交集，目标词表与 OVON unseen 无交集。

持续混合训练首次在 greedy oracle 的离散可达性边界报错。已修复为稳定目标视点、记录后尝试其他合法目标视点；全部失败时显式截断并 bootstrap 当前 episode，不生成错误监督标签。对应测试检查了失败发生在普通前缀和上一 episode 已终止这两种情况，防止把新 episode value 加到上一 episode 的终止奖励上。失败运行保留在 `runs/streamnav_mixed_oracle_failure/` 供追溯。

DAgger/PPO 概率按实际执行的 mixture action 计算，修正设计伪代码把未执行 policy action 与 oracle reward 配对的问题。FP32 optimizer master parameters、BF16 compute、functional recurrent state 和连续序列 minibatch 均在真实训练中使用。

## 初始验证结果及限制

主实验 update 10，在 OVON 三组各前 3 个固定 episode 上：

| Split | SR | SPL | SoftSPL | 平均最终距离 |
|---|---:|---:|---:|---:|
| val_seen | 0 | 0 | 0.22977 | 3.89857 m |
| val_seen_synonyms | 0 | 0 | 0.23979 | 3.68978 m |
| val_unseen | 0 | 0 | 0.08173 | 4.50304 m |

这些 episode 都跑到 500 步上限。当前检查点还没有稳定自主到达并 STOP 的能力；不要把 oracle 成功、DAgger 混合 rollout 成功或非零 SoftSPL 当作自主导航成功。固定混合对照的初始三组 SR/SPL 同样为 0。

主实验 update 20、30 的三组 SR/SPL/SoftSPL 均为 0，平均最终距离分别为 4.59972、3.88168、4.40623 m。初期指标没有持续改善，后续必须根据定期验证判断模型是否真正学会导航。最终后台配置为每 25 updates 保存、每 50 updates 验证，每组快速验证 3 episodes。

快速验证只用于检查训练趋势，采用固定顺序的少量 episode，不能替代完整 benchmark。使用 `eval.episodes=null` 执行全量验证，并报告四动作/224×224 输入与官方六动作/640×480 设置的差别。模型转换没有蒸馏，不保证保留原 VLM 全部能力；这也是需要继续训练与研究验证的部分。

## 运行与复现产物

- Qwen revision：`2fc06364715b967f1860aea9cf38778875588b17`。
- 原始权重 SHA256：`04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696`。
- Transformers 5.3.0、FLA/fla-core 0.4.2、Torch 2.10.0+cu126、Habitat-Sim 0.3.3。
- 每个 run 保存 config hash、dataset manifest hash、KDA layout hash、git SHA/dirty 状态和源代码哈希；新的检查点包含 `source.zip` 源代码快照。
- 真实进程 PID 以各 run 的 `training.pid`、`monitor.pid`、`keepalive.pid` 及 `runtime/viewer.pid` 为准；这些值可能因恢复而改变。
- 新依赖、缓存、数据与产物都位于用户指定共享目录下的项目路径；既有 `/root` Torch 环境只读复用。
