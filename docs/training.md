# 当前训练配方与运行

活动训练使用 revision 13 的 KDA-Qwen 感知分支、HM3D-OVON train 和 OVSegDT 风格的 on-policy IL／PPO／EALM。完整有效配置见 [活动 resolved_config.yaml](../runs/streamnav_active/ealm/resolved_config.yaml)。不能用通用默认 YAML 代替。原无感知实现和成绩冻结于 [版本节点](current_version.md)，新增分支见 [轻量感知监督](perception.md)。

## 每一步与每次更新

一个环境步输入该环境当前 RGB、目标文本和已有递归状态，策略输出一个动作，Habitat 执行动作并返回下一帧、奖励与结束标记。教师同时为当前状态提供 IL 动作标签。目标文本相同也会在每帧输入，递归状态带着更早历史。

一次训练 update 在八卡上各采集四个环境的 100 步，再进行两次 minibatch 优化，每批每卡两条完整 100 步序列、一个 epoch。每次 update 共 3,200 transitions，更新中包含两个 Adam steps。100 步同时是采集长度与 BPTT 截断长度，是当前工程配方，不是 KDA 的固定上下文长度；一个 episode 最多 500 步，可跨 update 延续状态。

| 项目 | 当前值 |
|---|---|
| GPU／每卡环境 | 8／4 |
| rollout_steps／sequence_length | 100／100 |
| sequence_batch_size／update_epochs | 2／1 |
| 语言主干／actor／critic LR | 1e-5／2.5e-4／2.5e-5 |
| 感知分支 LR | 2.5e-4 |
| Adam epsilon／weight decay | 1e-5／0 |
| 全局梯度裁剪 | 0.2 |
| 参数／计算精度 | FP32 master 参数和 Adam moments／BF16 autocast |
| 视觉编码器 | 冻结并缓存 rollout 视觉输出 |
| 当前训练目标 | 从原 update 2500 初始化，新增 2500 updates／800 万环境步 |
| 检查点间隔／轮转 | 每 50 updates／最近两份加 best，退出时完整保存 |
| 自主评估间隔 | 首次 update 50，随后每 100 updates |

当前分支从已训练模型初始化，取消 critic-only 首次 warmup。语言 embedding、KDA/GDN/MLP、actor、critic 和新增感知参数均参与更新，视觉编码器冻结。原分支的 Adam moments、各 rank RNG／sampler／EALM 保留，新感知参数建立自己的优化状态；分支 update 从零计。旧 NAV 参数不在当前读出路径中。

## 数据、教师与监督

仅使用 OVON train，验证的三个 split 不参与采样。按上游 worker 分配规则，将打乱场景轮流分给每卡四环境，每个环境至少分配 16 场景。Episode 惰性读取，按 episode 数加权安排场景顺序，场景内无放回打乱；每场景连续 50,000 步 ±20%，在 episode 边界轮换。随机流不保证与上游逐 episode 一致。

DAgger beta=0，所有动作从当前策略采样，教师只提供当前位置的监督标签。当前教师是 ObjNavExplorer＋`collision_safe` 可执行性修复，细节见 [架构](architecture.md#仿真与教师)。IL 按跨 rank 有效标签归一化，六动作类权重均为 1；实际阻塞前进监督用 `il_mask` 过滤。辅助 IL 环境数为零，没有 STOP 加权或近目标课程。

教师无法导航时记录原因，当前有效前缀显式截断并重置，连续三次失败终止训练。异常不伪造动作标签，不计作成功，不与普通 500 步超时混同。

## 损失、EALM 与奖励

```text
loss = alpha × IL + (1-alpha) × PPO + 0.5 × value - 0.01 × entropy
       + 0.05 × affordance + 0.10 × object_position + 0.10 × arrival
alpha = clip((previous_entropy_EMA - 0.35) / (0.75 - 0.35), 0, 1)
```

EALM 初始 alpha=1，EMA decay=0.95。每个 minibatch 用上一次平均熵 EMA 形成权重，成功完成 optimizer step 后才更新 EMA；拒绝的重放检查不改变它。各 rank 的 EMA 随检查点恢复。没有强制 PPO 权重下限。若 EMA≥0.75，alpha=1，PPO 策略梯度系数为零，但 value 与 entropy 仍训练；这也是当前监控中 `ppo_policy_inactive` 的含义。

当前分支初始化也恢复原 EMA。感知损失独立于 EALM，带置信度、空间邻域平滑和全局类别平衡；它们在 PPO 策略权重为零时仍参与优化。

PPO ratio 为 π_new(a)/π_old(a)，clip epsilon=0.2；gamma=0.99，GAE lambda=0.95，advantage 不归一化。Value loss 按上游 old value ±0.2 截断并屏蔽越界梯度，再算 0.5×MSE。正常结束和 500 步上限都切断 bootstrap，oracle 异常前缀另作显式截断。

奖励为成功 +5、每步 -0.001、碰撞 -0.003；无距离进度奖励或 false STOP penalty。成功要求主动 STOP 且到目标视点距离 <0.25 m；另记录严格 0.1 m 成功和曾到过目标邻域的 oracle_success。教师和奖励使用的位姿／目标信息不进入策略。

## 加速与数值保护

当前启用冻结视觉缓存、一次 replay 内 token embedding 复用、最多 64 帧因果合并、独立环境变长批处理、activation checkpointing 和 fused Adam。梯度仍覆盖完整 100 步，没有缩短序列或冻结更多语言层。

更新前用真实优化精度检查 rollout/replay 的 log-prob 差，所有 rank 最大误差必须≤0.05。超过时全 rank 同步降低合并数量，必要时退到逐帧／逐环境路径；仍失败或 loss 非有限则拒绝更新并保存退出。阈值和 rollout 旧概率不修改。优化后的 log-prob 差不能代替更新前校验。

短八卡测试中，预热后 update 从约 119.4 秒降至 20.9 秒，见 [效率验证摘要](../runtime/baselines/current_20261010/evidence/efficiency_validation.json)。实际长训练还包含 fallback、场景加载、评估及约 9.6 GB 的完整检查点写入，壁钟时间明显高于纯 update 时间；不要直接用短测速度估算总完成时间。

## 评估、case 视频与监控

每 100 updates 在 seen、seen_synonyms、unseen 各运行 48 条固定分层 episode，原始起点，argmax，最多 500 步。按 episode 数汇总，各 split 写入 `eval_metrics.jsonl`。这是诊断子集，成绩不能当作官方完整 benchmark；训练随机采样成功率也不能替代自主评估。

感知分支另在 update 50 提前评估。评估关闭深度标注，视频和 trace 展示模型预测的可走点、目标点与 P(ready)，用于诊断动作和停车决策。

每 split 保存最多六条代表视频，共 18 条；包含固定对照和成功、到达未停车、错误停车、高碰撞、持续转向等结果。不足时补充不同场景与目标。视频选择发生在评估后，距离只用于诊断。`evaluation/update_*/index.html` 是播放页，`video_cases.json` 记录选择理由，case `.jsonl` 保留动作概率、距离和碰撞。当前训练的全部既有评估记录及视频保留。

后台监控每 30 秒检查真实 worker、loss、更新和评估。非有限值、worker 丢失、3,600 秒不推进或达到审查预算后的持续全零／坍缩会请求当前 update 后保存退出，不自动反复重启。具体原因写入 `health_status.json` 和 `monitor.log`。

| 指标 | 用途 |
|---|---|
| `preupdate_replay_log_prob_error_max` | 更新前重放一致性，阈值 0.05 |
| `kda_grad_norm`／`actor_grad_norm`／`vision_grad_norm` | 语言主干和动作头更新，视觉应为零 |
| `entropy_ema`／`ealm_alpha`／`ppo_policy_coefficient` | PPO 策略实际参与情况 |
| `oracle_class_recall`／STOP 标签数量 | 有效监督与停车拟合 |
| `oracle_skipped_episodes`／`oracle_navigation_repairs` | 教师失败和修复 |
| `eval_metrics.jsonl` | 固定协议的自主 SR／SPL／转向循环与停车 |
| `update_seconds`／GPU 状态 | 训练计算效率，需另考虑保存与评估开销 |

## 启动、停止与恢复

先在 Bash 中 `source scripts/env.sh`。运行中的任务无需重新启动。

```bash
python tools/training_status.py --root runs/streamnav_active
cat runs/streamnav_active/ealm/health_status.json

# 需要暂停时：当前 update 完成后完整保存并退出
bash scripts/stop_training.sh
```

同目录恢复要求最后一条日志 update 与检查点一致，且配方／world size 相同。以下命令仅在任务已停止后执行，直接读取最新检查点保存的完整配置：

```bash
streamnav_resume_checkpoint="$(realpath runs/streamnav_active/ealm/checkpoints/latest)"
STREAMNAV_KEEPALIVE=0 STREAMNAV_WORLD_SIZE=8 \
  STREAMNAV_RUN_DIR=runs/streamnav_active/ealm \
  bash scripts/start_distributed_training.sh \
  --config-path "$streamnav_resume_checkpoint" --config-name resolved_config \
  "checkpoint=$streamnav_resume_checkpoint" trainer.num_updates=2500 \
  'trainer.fork_recipe_changes=[]'
```

恢复包括模型、Adam、调度、八 rank RNG／场景与 episode sampler／entropy EMA。仿真物理状态和正在执行的 episode 不保存，恢复从新 episode 开始，不保证逐动作续接。数据、机器人、模型结构与核心配方变化会拒绝恢复；允许的配方实验必须显式 fork 到新空目录。

创建同配方、从同一导航 checkpoint 增加感知分支的新实验，可使用活动配置；目录名必须独立：

```bash
STREAMNAV_KEEPALIVE=0 STREAMNAV_WORLD_SIZE=8 \
  STREAMNAV_RUN_DIR=runs/my_experiment/ealm \
  bash scripts/start_distributed_training.sh \
  --config-path "$PWD/runs/streamnav_active/ealm" --config-name resolved_config \
  checkpoint=null 'trainer.fork_recipe_changes=[]'
```

固定节点 update 2240 用于长期保留和分支复现，日常续训使用活动 latest。节点和空间规则见 [当前版本](current_version.md)。
