# 2026-10-02：8 卡续训与监督

这次继续运行两个四卡 DDP 实验：GPU 0–3 为 EALM，GPU 4–7 为固定 0.5 混合对照。机器人仍为 480×270、HFOV 120°、相机 0.88 m / 0°、机体高 0.88 m / 半径 0.18 m。学习算法、数据划分、奖励和验证成功距离均未更改。

## 中断检查与恢复

接手时 8 张 RTX 4090 均为空闲，原 learner、monitor 和 viewer 都已退出。旧监控最后写入时间为 2026-10-02 09:47:22 UTC，仍错误保留 `process_alive=true`。原日志末尾没有异常栈，当前 cgroup 的 `oom` / `oom_kill` 为零；这些证据不足以判定先前进程的确切退出原因。

| 实验 | 原日志最后 update | 完整恢复检查点 | 新目录 |
|---|---:|---:|---|
| EALM | 4035 | 4025 | `runs/streamnav_8gpu_20261002/ealm` |
| fixed_mix | 4050 | 4050 | `runs/streamnav_8gpu_20261002/fixed_mix` |

2026-10-02 11:22:51 UTC 启动两个 torchrun。`runs/streamnav_active` 指向本次根目录，便于查看状态和启动网页。每卡 8 个训练环境，每组每次更新采集 512 个新 transitions。恢复优化器、LR / DAgger 调度、各 rank 的 RNG 与 sampler，物理 episode 和递归状态重新初始化。

`tools/prepare_resume.py` 创建独立续训目录，将历史指标继承到检查点对应的 update，保存 `continuation.json`。主实验原先 4026–4035 的未落盘模型对应指标仍在原目录，没有删除、覆盖或算作续训的新数据。历史最佳权重与验证视频通过只读用途的符号链接继承，因此旧目录仍是依赖，不能删除。新检查点在本次目录中独立发布和保留。

正常停止后可以执行：

```bash
source scripts/env.sh
STREAMNAV_RESUME=1 bash scripts/start_eight_gpu_training.sh
python tools/training_status.py
```

如果意外退出导致日志 update 晚于 latest，必须再次建立分支，不可直接截短原日志：

```bash
python tools/prepare_resume.py \
  --source runs/streamnav_8gpu_20261002/ealm \
  --target runs/my_continuation/ealm
python tools/prepare_resume.py \
  --source runs/streamnav_8gpu_20261002/fixed_mix \
  --target runs/my_continuation/fixed_mix
STREAMNAV_RUN_ROOT=runs/my_continuation STREAMNAV_RESUME=1 \
  bash scripts/start_eight_gpu_training.sh
```

只有在原训练全部停止后才能更改 active 别名。启动器将 run_dir 解析为实际路径，避免之后更换别名导致日志写入不同实验。

## 学习效果检查

原 update 4000 的每个 split 有 48 个固定、跨场景的自主 episode，使用原始起点、argmax，无专家接管。主成功距离仍为 0.25 m，并额外保留严格 0.10 m 指标。

| 实验 | seen SR | synonyms SR | unseen SR | 宏平均 SR | 历史最佳宏平均 SR |
|---|---:|---:|---:|---:|---:|
| EALM | 2/48 | 0/48 | 5/48 | 4.86% | 8.33%，update 1200 |
| fixed_mix | 5/48 | 1/48 | 3/48 | 6.25% | 6.25%，update 2800，SPL 用于同 SR 时选择 |

这已经有非零成功，但远不足以说明导航效果良好。两组每组约 2.05M 个训练 transitions；训练 success 含 DAgger 接管，不能当作自主成绩。不能仅因数据预算比 OVSegDT 小，就排除实现或学习方法的问题。

逐 episode 检查显示，EALM 的 137 个失败中，102 个在 100 步以内提前结束，15 个达到 500 步上限；各 split 失败终点的目标距离中位数约 4.77–5.25 m。固定混合的 135 个失败中，95 个在 100 步以内结束、9 个达到上限。失败轨迹的平均碰撞比例约为 27%–43%，多数失败并非仅仅在成功边界少走一步。相关逐 split 统计保存在 `runtime/resume_20261002/*_failure_analysis.json`。

检查了 rollout / sequence replay、DAgger 行为概率、PPO ratio、全局 advantage 与 IL 归一化、梯度裁剪、原始 RGB 和目标 prefill 路径。当前没有发现新的数值非有限、动作映射或奖励对错动作问题。梯度范数偏大但有限，IL 已优于类别先验；这些都不等于已经学会目标导航。没有根据低 SR 擅自删掉真实失败记录，也没有更改评测阈值来提高成绩。

## 本次修复

1. 状态查询重新读取 `/proc`，核验 launcher、worker 命令及父进程，排除 zombie 和明显 PID 复用。监控过期或消失会明确显示异常，防止容器整体退出后误显示“训练中”。网页 `/training` 采用同一逻辑。
2. 学习监控增加长时间无 SR 提升、最近五轮宏平均 SR 均低于 10% 的告警。保留原有非有限 loss、worker 退出、更新停止、持续全零和单动作坍缩的安全停训规则。小样本 plateau 只告警，不无依据自动删除或重启实验。
3. 支持双实验显式续训、日志分支与沿革记录。新目录继承历史 best，新的较差模型不会覆盖已验证的最佳模型。
4. 网页恢复历史最佳模型，读取当前续训监控；继承的历史视频链接也能正常访问。默认脚本、状态工具和 README 指向 active 实验。
5. 修复推理精度不一致：训练内验证使用 FP32 master + BF16 autocast，原网页和独立评测却先将权重转成 BF16。现在网页及独立评测保留 FP32 权重，并显式使用相同的 BF16 计算上下文。独立评测也读取 checkpoint 的真实 update，不再全部标记为 update 0。此项不改变正在运行的训练数学或已有批量验证记录。

统一精度后的第一次网页回放出现了 viewer OOM，8 个 learner 未退出。原因是 autocast 为整模型缓存额外的 BF16 权重副本；关闭网页 autocast 权重缓存后完成了完整回放。热切换权重时将旧模型先移至 CPU，避免双份 FP32 GPU 权重。OOM 日志和不完整视频独立保存在 `runtime/resume_20261002`，没有混入正式 episode 结果。

后台监控每 30 秒检查一次。它会报告异常并按规则请求安全保存停止，不会自动编写代码，也无法跨容器整体退出继续执行。

## 验证与可视化

使用 `.venv` 的 `python -m pytest`：41 项 CPU 单元 / 回归检查通过；Ruff 检查与格式检查通过，mypy 检查 54 个源文件通过。直接调用系统 `pytest` 曾因其解释器缺少 hydra 在收集阶段失败，改为项目 Python 后通过，未额外向系统安装依赖。

真实浏览器已通过加载 Habitat 场景、模型单步、480×270 图像和传感器参数核验，没有 JavaScript 错误。截图为 `docs/assets/viewer_resume_20261002.png`，报告位于 `runtime/resume_20261002/viewer_browser_test.log`。

```bash
bash scripts/start_viewer.sh
# 本机浏览器：http://127.0.0.1:8765
# 远端使用：ssh -L 8765:127.0.0.1:8765 <服务器>
```

网页当前加载 EALM 最佳 update 1200，默认使用 GPU 3。下拉列表展示全部 48 个诊断样本及成功 / 失败标记，支持自动运行、模型单步和明确标记的手动控制。新 best 发布后，在重置 episode 时加载。

完整真实交互回放保存在 `runs/streamnav_active/ealm/evaluation/viewer_replays_20261002`，同时包含视频和每一步的动作 / 概率 / 指标：

| seen 样本 | 交互回放结果 | 与训练内批量验证的差异 |
|---|---|---|
| 0，couch | 500 步失败，最终距离 6.73 m，碰撞比例 90.6% | 原批量验证为 59 步成功 |
| 46，tv | 313 步成功，最终距离 0.221 m，碰撞比例 87.5% | 原批量验证为 25 步成功 |

这些差异仍然存在，不能声称精度路径统一后轨迹已逐步复现。进一步用相同 JPEG 帧、相同目标、独立持久状态比较 batch 1 / batch 2：30 帧中 2 帧 argmax 不同，动作概率最大差异 0.260；batch 2 中两个完全相同的输入输出一致。报告为 `runtime/resume_20261002/batch_precision_probe.json`。长期导航稳定性仍需继续改进，正式成绩与交互回放分别记录。

随后用同一检查点、同一真实帧、克隆的目标初始状态，对视觉编码器及全部 24 个 decoder 层逐层比较 batch 1 / batch 2，定位到 BF16 数值差异被深层放大：

| 计算配置 | 视觉输出相对 RMS 差异 | 第 24 层相对 RMS 差异 | 动作概率最大差异 |
|---|---:|---:|---:|
| 当前 BF16 | 0.01574 | 0.36614 | 0.04341 |
| BF16，禁用 reduced precision GEMM reduction | 0 | 0.14267 | 0.03245 |
| FP32 视觉 + BF16 decoder，禁用该 reduction | 0.00000264 | 0.13737 | 0.06588 |
| 全 FP32 计算 | 0.00000264 | 0.00006001 | 0.00001012 |

诊断代码为 `tools/probe_batch_precision.py`，数据为 `runtime/resume_20261002/layer_precision_options.json`。这支持“当前混合精度路径存在较强数值敏感性”的判断，不支持归咎于某一个 KDA kernel：FP32 下同一递归实现的批量差异很小。局部提高精度并未一致改善动作概率，尚未验证全 FP32 训练的显存、吞吐及自主 SR，因此没有在正在运行的对照实验中偷偷切换训练精度。后续应在明确标记的实验中验证精度稳定性与导航收益。

目标敏感性诊断在原 BF16 权重网页上使用 3 条真实轨迹、相同帧但不同目标文本。第 10 帧后的平均动作概率 total variation 约 0.070–0.075，argmax 差异约 6%–13%。目标信息并非完全丢失；这不是正确目标识别的证据，也不应被当作 SR。原始报告为 `runtime/resume_20261002/goal_conditioning.json`，明确标注其旧推理精度背景。

120 秒采样中，8 卡平均利用率分别约 70.2%、56.4%、65.9%、62.0%、62.2%、54.8%、58.3%、62.9%，包含训练、通信、仿真与网页诊断，不能全部当作有效反向计算。未触发合成保活。采样位于 `runtime/resume_20261002/gpu_util_120s.jsonl`。

## 本轮监督结束时的状态

截至 2026-10-02 11:49 UTC，两组已完成第 4100 次更新的全部验证，并重新进入训练。主实验已超过 4100，对照已超过 4125；两个 launcher、8 个 worker、两个外部监控和 viewer 均运行。每 25 updates 保存、每 100 updates 验证的后台流程继续执行。

| 实验，update 4100 | seen | synonyms | unseen | 宏平均自主 SR | 严格 0.10 m 宏平均 SR |
|---|---:|---:|---:|---:|---:|
| EALM | 3/48 | 1/48 | 5/48 | 6.25% | 2.08% |
| fixed_mix | 3/48 | 2/48 | 2/48 | 4.86% | 2.08% |

本轮各分集都有自主成功，历史最佳仍为 EALM update 1200 的 8.33% 和 fixed_mix update 2800 的 6.25%。本轮波动不能被解释为稳定改进，也没有达到 OVSegDT 水平。两个监控均保留长期无提升和持续低成功率告警，当前未触发停训规则。

新训练日志未出现 learner OOM、非有限 loss 或异常栈。续训的更新数、累计新 transitions、loss 范围、每轮耗时、完整第 4100 次验证指标、最佳检查点和真实进程快照保存在：

- `runtime/resume_20261002/final_learning_report.json`
- `runtime/resume_20261002/final_status.json`
- `runtime/resume_20261002/final_checks.log`
- `runtime/resume_20261002/operations_manifest.json`

本轮修改的服务 / 监控代码哈希记录在 operations manifest；正在运行的 learner 使用其启动时归档的训练源码，更新数学未在进程中途替换。后台监控能检查和按规则停训，不能在本次交互结束后自动改写代码；下一次迭代应继续依据验证和这些诊断数据进行。
