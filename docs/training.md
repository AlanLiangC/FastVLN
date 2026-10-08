# 训练与恢复

训练过程始终为 online rollout + oracle IL + clipped PPO + value regression + entropy bonus。EALM 从当前四分类策略 entropy 得到 detached 权重，`entropy_low=0.2`、`entropy_high=1.2`，超出范围截到 `[0,1]`。高熵时 IL 权重大，低熵时 PPO 权重大。

## DAgger 与 PPO 的一致性

rollout 同时保存 sampled policy action、executed action、expert label、policy log probability 和实际行为分布 log probability。实际行为分布是：

```text
mu = (1 - beta) * pi + beta * one_hot(expert)
```

PPO 在固定 rollout beta 下重算 `mu(executed_action)`。奖励对应环境实际执行的动作，因此不使用未执行 policy action 的概率配上 oracle reward。`beta=0` 退化为普通 PPO；`beta=1` 时该行为分布不依赖 actor，PPO 项的 actor 梯度为零。

Buffer 为预分配 CPU uint8 RGB 和 `[T,N]` tensor 字段。minibatch 只采样连续片段；每个片段起点保存 detached state，重放期间保留片段内 BPTT。片段内 reset 会重新生成目标 prefill；PPO 多 epoch 不修改原 snapshot。

真实 STOP 切断 bootstrap。时间上限与 oracle 不可用造成的截断，从同一 episode 的最后 RGB/state bootstrap，同时切断 GAE 跨 episode 的递推。绝不能从 reset 后的新 episode value bootstrap。

## 本次实验

当前目录为 `runs/streamnav_active/ealm` 和 `runs/streamnav_active/fixed_mix`，指向 2026-10-02 续训目录；恢复点分别为旧实验 4025 / 4050。详见 [本次续训记录](training_resume_20261002.md)。两者同一 backbone、数据配比、seed 和更新预算，对照实验仅使用 `trainer.ealm.enabled=false`、`trainer.ealm.fixed_alpha=0.5`。运行时有效参数以各目录 `resolved_config.yaml` 为准。

```bash
# 固定混合对照，GPU 4–7
STREAMNAV_RUN_DIR=runs/new_fixed_mix \
STREAMNAV_GPU_OFFSET=4 STREAMNAV_WORLD_SIZE=4 \
  bash scripts/start_distributed_training.sh trainer.ealm.enabled=false

# 其他文档中的 objective ablation
python -m streamnav.training.trainer trainer.ealm.enabled=false trainer.ealm.fixed_alpha=1.0
python -m streamnav.training.trainer trainer.ealm.enabled=false trainer.ealm.fixed_alpha=0.0 \
  trainer.dagger.beta_start=0.0 trainer.dagger.beta_end=0.0
```

对照实验应单独设置 `run_dir`，不要混写主实验日志。当前未实现 explicit-history 额外 backbone 或多模型 registry，符合唯一模型约束。

新参数实验使用 480×270、HFOV 120°、相机 0.88 m / 0°、机体 0.88 m / 0.18 m，并从原始 KDA 转换初始化重新开始。旧长训练已标记失败，保留诊断证据；错误碰撞配置实验已删除。每卡 8 environments × 16 steps、sequence batch size 8，每实验 4 卡，共 512 个新 transitions/update、关闭 gradient checkpointing、fused AdamW；第 1 次更新先保存并验证，之后每 25 updates 保存、在 25、50、100、200… updates 对三个 split 各 48 episodes 做跨场景并行验证。详见 [8 卡审计与实测](training_8gpu_20261001.md)。

评估的 `latency_p*` 是一个 observation 在当前批量中等待策略计算的时间，`decision_hz` 是对应的每流策略频率；`policy_decisions_per_second` 是整个批量的决策吞吐。字段 `evaluation_batch_size` 明确并行数；两者均不包含仿真器/RPC/视频编码。

## 检查点

单个训练 checkpoint 约 9.7 GiB（FP32 权重和 AdamW moments），保留最近三份，并额外保护 best 所指的最佳自主验证模型。写入完成前目录名以 `.writing-` 开头；之后原子 rename，最后原子更新 `latest` symlink。不要把半成品目录用于服务或恢复。

恢复会读取优化器、LR scheduler、DAgger update、每个 rank 的 Python/NumPy/Torch/CUDA RNG、各数据 sampler。为保证可审计性，训练数据/配比/过滤规则、图像尺寸、相机/机体参数、head 结构、成功距离、world size 和 DAgger schedule 改变会被拒绝。Simulator 进程与物理状态不序列化，恢复后以新 episode 起步。这是 update 级恢复，不是完整 rollout 状态复现。

`SIGTERM` 和 `SIGINT` 请求在当前 update 后安全保存。程序错误会明确退出；后台启动器不无限自动重启错误训练。查看 `training.log`、`train_metrics.jsonl`、各 worker 的 `runtime/logs/habitat-*.log`，排除原因后用 `checkpoint=.../latest` 恢复。

## 看哪些日志

- `vision_grad_norm`、`kda_grad_norm`、`actor_grad_norm`、`critic_grad_norm` 确认各分支参与学习。
- `greedy_action_histogram` 检查实际自主 argmax 行为；`policy_action_histogram` 是随机采样分布，不能单独用来排除坍缩。`expert_action_histogram`、`oracle_class_recall`、`il_gain_over_prior` 辅助判断是否只在拟合类别先验。
- `oracle_skipped_episodes` 非零表示 greedy oracle 在候选视点恢复后仍失效；这些位置不生成伪标签。
- `success`/`spl` 是当前 rollout 内完成 episode 的混合策略统计。没有完成 episode 时这两个日志值为 0，需同时读取 `episodes_completed`。
- `eval_metrics.jsonl` 才是自主模型验证成绩；快速子集的统计波动较大。
- `gpu_metrics.jsonl` 记录实际利用率；初始化、首轮 Triton 编译、checkpoint I/O 和验证都可能造成暂时下降。
- `keepalive_events.jsonl` 单独标记按用户资源保留要求触发的矩阵占用计算，不能作为训练吞吐或模型性能证据。守护进程跟随 learner 生命周期，默认连续低于 50% 达 2 小时才触发 15 秒占用。

## 2026-10-01 恢复配置与监督

旧训练有约 5,600 updates、675 条 split 验证记录，但自主 SR 全为零。训练 success 含 oracle 接管，不能作为自主导航效果。详见 [审计报告](training_audit_20261001.md)。

revision 3 延续第二版的主干 / 视觉 / head 学习率设置为 3e-6 / 1e-6 / 1e-4，STOP IL 权重设置为 8（其余为 1，按全局 minibatch 实际权重均值归一化），false STOP penalty 为 0.2。前 2,000 updates，50% 新训练 episode 从 oracle 预先行进后的较近位置开始，距离门槛从 1.5 m 线性增加到 6 m；不向模型传递距离或位姿。课程 warmup 的真实仿真步数记为 `curriculum_warmup_steps`，不算 learner transitions。原始起点仍占 50%，所有正式验证保留原始起点。数据分布因此变化，结果必须标注 revision 3，不能与旧实验直接混合。

`health_status.json` 是进程外监控结果，每 30 秒更新；任何仍为零的 split 都会明确列出，部分成功不等于整体健康。默认规则：

- loss 非有限或 3,600 秒无新 update：请求停止。
- update ≥500，最近 5 次完整验证中三个 split 的 SR 全为零：请求停止。
- update ≥500，最近 50 updates 平均 greedy 单动作比例 ≥95%，且 IL 对类别先验的平均改善 <0.05：请求停止。

监控发送 SIGTERM，训练完成当前 update（正在验证时会先完成该轮验证）后保存退出，状态为 `halted_for_review`；不无限自动恢复。默认每 100 updates 验证，另在 update 1、25、50 提前检查，因此连续失败可在约 update 500 截止，而不会再跑到 5,600 才发现。门槛可以在 `trainer.supervision` 中调整，但提高预算前应先检查失败原因。

查看：

```bash
cat runs/streamnav_8gpu_20261001/ealm/health_status.json
cat runs/streamnav_8gpu_20261001/fixed_mix/health_status.json
tail -f runs/streamnav_8gpu_20261001/ealm/monitor.log
```

已有 run_dir 禁止无 checkpoint 重启；从旧 update 分支必须用新目录。配置 `trainer.revision` 改变后拒绝恢复旧优化器；本轮两实验从转换初始化开始。


## 多卡同步与第三版 oracle 修复

`torchrun` 启动每组四个 learner，`distributed.gpu_offset` 分别为 0 / 4。rollout、随机数与 sampler 按 rank 独立；sequence replay 作为一个完整 DDP forward，使递归片段内的梯度先累积，再分桶同步。FP32 梯度在裁剪、AdamW 之前平均；global advantage 与 IL 权重分母也跨 rank 归一化。NCCL 通信与反向计算重叠，检查点只由 rank 0 发布。完整代码、配置和数据哈希可从各 checkpoint 的 source.zip / manifest.json 追溯。

验证固定全局 episode 顺序，再按 index % world_size 分片，汇总时按实际 episode 数求均值，不能直接平均各 rank 的 SR。原始逐 episode 文件位于 rank_XXX 子目录，汇总文件位于 update_*/ 根目录。每个 split 的视频来自全局第一个 episode。

第三版的主成功半径 / follower goal_radius 改为 0.25 m，保留额外的 `success_strict_0_1` 指标。0.1 m 小于 0.25 m 离散前进步长，48 个真实配对 oracle 测试中，原阈值只有 31 个成功，0.25 m 则 48 个全成功。原教师的长时间近目标旋转会污染方向标签；该结论来自小样本配对诊断，不代表全量 oracle 成功率。64 步无 geodesic 改善会报告 oracle 失效，已有截断恢复逻辑处理；课程 warmup 的 oracle 异常会恢复原始起点，并记录 `curriculum_fallbacks`，不伪造标签。旧模型在放宽阈值后的 72 个自主验证 episode 仍然全零，说明只是修改判定不会修好已经失败的策略。

多卡数学校验（包括 rank 间部分未使用参数、全局 advantage、同步停止标志）：

```bash
PROBE_GPU_OFFSET=0 python -m torch.distributed.run --standalone --nproc_per_node=2 tools/check_distributed.py
```

请在对应 GPU 空闲时运行额外验证。监控会确认 worker 属于本次 launcher，避免恢复时旧 PID 文件触发误停；不要只看 training.pid 或 GPU 占用判断训练正常。

## 2026-10-02：实时状态与评测精度

运行 `python tools/training_status.py` 会重新核验实际进程，`health_status.json` 本身只是监控快照。续训分支工具会保留原日志，并继承检查点之前的历史和最佳权重，不覆盖未落盘 update 的记录。

网页与独立评测现在保留 FP32 master 权重，使用 BF16 autocast，与训练内验证一致。网页关闭 autocast 权重缓存以避免额外驻留整份 BF16 权重副本；新 best 切换时先将旧权重移到 CPU。独立评测从 checkpoint manifest 读取真实 update 编号，不再一律输出到 update 0。批量与单流推理仍可能因浮点差异在长轨迹中分叉，单条交互回放不能替代正式批量验证成绩。
