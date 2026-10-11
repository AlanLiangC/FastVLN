# 训练与监控

入口为 `scripts/start_lightnav_training.sh`，默认配置 `configs/lightnav_kda.yaml`。八卡、每卡两个环境，rollout 100 步，每 update 共 1600 条 transition。BPTT 长度 20，跨片段保留缓存；全部片段完成后统一同步梯度并更新一次参数。

`trainer.replay_mode: packed` 将同一片段的 20 帧批量回放，保留逐帧局部 attention、因果 KDA、episode 重置与实际 APOS/OPOS/action token。视觉使用 BF16，语言与输出头使用 FP32，允许 TF32 加速矩阵乘法；单 token 无梯度解码使用 fused recurrent KDA。没有缩短采样或 BPTT，也没有减少损失项。数值误差超过 0.05 的片段自动用顺序路径重算，记录 `packed_replay_error_max` 与 `replay_fallback_fragments`。`sequential` 可用于对照；执行方式、计算精度与解码内核属于可恢复的运行选项，不改变底座或优化配方。

```bash
source scripts/env.sh
bash scripts/start_lightnav_training.sh
python tools/training_status.py --root runs/lightnav_kda_active
tail -f runs/lightnav_kda_active/training.log
```

前 50 updates 逐渐减少教师执行概率，动作 CE 持续学习，PPO 系数为零；第 51 update 开始完全执行学生策略，PPO 在后续 100 updates 从 0 升至 0.5。DAgger 的 oracle 动作标签始终保留，但 PPO 不使用教师强制执行的 transition。

损失包含动作 CE、三 token 联合策略 PPO、value loss、冻结 LightNav 的点位 KL、动作和点位熵。critic 在 APOS 生成前读取；GAE 在 episode 边界断开，并对时间上限和不可用 oracle 的截断做 value bootstrap。每四步蒸馏两个点位分布，补偿稀疏频率后系数为 0.1。

每个训练 update 都验证预更新联合 log-prob 重放误差和梯度是否有限。日志记录 IL/PPO/点位损失、PPO 系数和实际 token 梯度、动作分布、oracle 分布、rollout 与反向耗时和显存。该检查验证训练的一致性，不代表导航成功率已经提升。

`training_progress.json` 在每 20 步采样及每个回放片段完成时原子更新，提供当前 update、阶段、完成数和耗时；`training_status.py` 同时显示这些进度。完整训练指标仍在 update 结束时写入。

每 100 updates 在 val_seen、val_seen_synonyms、val_unseen 各评估 48 条固定 episode，最多 500 步；每 split 保存六个代表 case，视频显示鲜明的 APOS/OPOS 标记。路径为 `evaluation/update_*/index.html`。默认入口也评估训练起点。本轮已经进行独立八卡短测，正式训练用 `--skip-initial-eval` 跳过重复起点评估；首次完整 144-case 评估作为正式学习曲线起点，短测结果不能与之直接比较。

第 1 update 和此后每 25 updates 保存一次可训练参数、Adam 状态、各 rank 的 RNG 与数据采样状态。保留初始化、最近两个及已评估的最佳检查点；底座固定并单独存放，检查点不重复保存整个 4B 模型。

```bash
bash scripts/stop_training.sh runs/lightnav_kda_active
bash scripts/start_lightnav_training.sh --skip-initial-eval --resume runs/lightnav_kda_active/checkpoints/latest
```

停止信号在共同的 BPTT 边界处理，完成已收集片段的优化和保存。恢复要求同一底座、参数集合、训练配方与卡数；保留 optimizer、更新数、采样器和 RNG，环境重新创建 episode。改变训练目标 update 数和评估频率可以恢复，改变优化配方需新建实验。

旧版必要最终结果归档于 `runtime/archive/legacy_streamnav_20261011`，旧实验、更新流水和冗余权重已经清理。新训练使用独立目录。

优势归一化后，更新前 PPO loss 的全局均值可以接近零，而策略梯度仍非零。`ppo_token_gradient_norm_max` 用于区分这种正常情况与 PPO 分支未参与优化。

真实 Habitat 20 帧同权重前向/反向对照：FP32 顺序回放 33.76 秒，批量回放 2.94 秒，约 11.5 倍，联合概率误差 0.000016，最低参数组梯度余弦相似度 0.99999989；Torch 峰值显存约 19.7 GiB。允许 TF32 后，同一段批量回放约 2.18 秒，概率误差约 0.006，仍低于原阈值。此测试包含 cache 初值与 episode 重置，排除首次编译；它不是整轮加速倍率或导航成绩。整轮耗时还受环境采样与评估影响，以实时日志为准。报告为 `runtime/reports/lightnav_efficiency_20261011/fp32/replay_comparison.json` 和 `tf32_comparison.json`。

正式目录为 `runs/lightnav_kda_20261011`，别名为 `runs/lightnav_kda_active`。底座为 4B，不能直接与旧 0.8B 模型比较速度。

八卡完整 1600-transition 实跑，update 21 用时 99.50 秒，其中采样 76.64 秒、优化 22.88 秒；相对原约 447 秒/update 加速约 4.5 倍。80 个回放片段均未触发重算，最大联合概率误差 0.0088，Torch 峰值显存约 20.4 GiB。此结果不包含评估耗时，仍需以后续日志监控长期速度及重算比例。
