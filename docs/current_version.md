# 当前版本节点

节点 **current_20261010** 固定 revision 12 的实现、配置和完整检查点。后续实验通过活动入口运行，固定节点不会随 latest/best 轮转。源码、节点信息和精简验证证据位于 `runtime/baselines/current_20261010/`。当前感知分支的配置与限制见 [轻量感知监督](perception.md)，它不属于此固定快照。

## 节点与保留路径

| 项目 | 路径或值 |
|---|---|
| 活动入口 | `runs/streamnav_active/ealm` |
| 节点建立时训练目录 | `runs/streamnav_executable_teacher_long_20261009/ealm` |
| 节点训练目标 | update 5000，八卡，每卡四环境，100 步序列 |
| 固定完整恢复节点 | [checkpoints/current_20261010/resume](../checkpoints/current_20261010/resume/)，update 2240 |
| 固定最佳评估模型 | [checkpoints/current_20261010/best](../checkpoints/current_20261010/best/)，update 2000 |
| 当前教师分支的对照起点 | [revision12_ablation_update1000_20261009](../checkpoints/revision12_ablation_update1000_20261009/)，update 1000 |
| 导航初始化 | `checkpoints/qwen35_0p8b_kda_temporal_contrast_20261008` |
| 基础转换／GPU 测试权重 | `checkpoints/qwen35_0p8b_kda`、`checkpoints/qwen35_0p8b_kda_calibrated_20261008` |
| 节点记录／完整配方 | [node.json](../runtime/baselines/current_20261010/node.json)、[resolved_config.yaml](../runtime/baselines/current_20261010/resolved_config.yaml) |
| 整理后的源码与文档 | [source.zip](../runtime/baselines/current_20261010/source.zip)，哈希见节点记录 |
| 清理清单 | [cleanup.json](../runtime/baselines/current_20261010/cleanup.json) |

固定检查点保留 model、actor/critic、Adam、调度器、八 rank RNG／sampler／entropy EMA、tokenizer、配置和实际训练源码。大文件通过硬链接固定，不额外复制权重；禁止原地修改固定文件。日常续训使用活动实验 latest。较早节点不能直接接在已有更晚的日志后；需要独立分支目录。

本节点是工作区快照，不是新的 Git commit。基础 commit、工作区状态和源码 SHA256 写入 node.json；训练时的真实源码另保留于各检查点的 source.zip。

## 节点实现

策略仅读取 RGB＋目标文本。Qwen3.5-0.8B 的六个 full-attention 层转换为 KDA，18 个 GDN 保留，每帧输入官方 user 图像／目标消息并读取 assistant 前缀。递归状态大小固定。视觉冻结，语言主干、actor 和线性 critic 训练；采集与 BPTT 均为 100 步。

导航使用 on-policy 教师标签、EALM、PPO/value 和 entropy。当前训练教师显式启用 `collision_safe`：保留上游探索目标，用实际机器人离散动作修复不可执行的前进标签，另过滤实际无位移碰撞前进的 IL 监督。策略不读取修复用的特权信息。辅助 IL、STOP 加权、近目标课程、进度奖励和 PPO 权重下限均未启用。细节见 [架构](architecture.md) 和 [训练配方](training.md)。

## 评估与验证

以下均为同一固定 144 条 episode 的自主 argmax 诊断评估，不是完整 benchmark；表中数值冻结于节点建立时。

| 模型 update | 成功数／144 | SR | SPL |
|---|---:|---:|---:|
| 对照起点 1000 | 20 | 13.89% | 0.0802 |
| 节点最佳 2000 | 27 | 18.75% | 0.1098 |
| 节点建立时最近完整评估 2200 | 22 | 15.28% | 0.0865 |

最佳模型按宏平均 SR、再按 SPL 选择。结果仍有明显波动，到达目标后不停车与持续左右交替尚未解决。节点建立时 EALM alpha 仍为 1，PPO 策略梯度未参与，value 与 entropy 项仍存在。不能仅凭 IL loss、训练采样成功率或低碰撞率判断自主能力改善。

保留的证据包括 [最佳评估](../runtime/baselines/current_20261010/evidence/current_best_evaluation.json)、[评估记录](../runtime/baselines/current_20261010/evidence/evaluation_at_snapshot.jsonl)、[训练诊断](../runtime/baselines/current_20261010/evidence/training_analysis_update2000.json)、[进度图](../runtime/baselines/current_20261010/evidence/training_progress_update2000.png)。原起点的 [视频](../runtime/baselines/current_20261010/reference_evaluation/update_0001000/index.html) 保留用于对照，活动实验全部评估视频保留。

完整八卡、100 步、两次更新验证中，各 rank 参数哈希相同、语言主干／actor 有梯度、视觉梯度为零，更新前最大重放误差约 0.02217，原阈值为 0.05。见 [八卡验证摘要](../runtime/baselines/current_20261010/evidence/distributed_canary.json)。同一 32 条训练 episode 的教师执行验证由 7 次成功变为 22 次；这是教师质量检查，不是策略 SR。见 [教师验证摘要](../runtime/baselines/current_20261010/evidence/teacher_validation.json)。这些是已完成验证的留存证据，本次文档整理不重新占用 GPU 测试。

## 空间保留规则

活动训练保留最近两份检查点和 best，全部 train/eval 指标及代表视频保留。固定版本节点、当前初始化、update 1000 对照、基础转换及测试权重单独保留。旧训练分支、失效节点、失败初始化、重复诊断输出和旧更新流水已清理，具体路径与空间统计见清理清单。

`src`、`services`、`configs`、`tools`、`tests`、`scripts` 保留实现与复现能力。`runtime/data`、`.venv`、`runtime/habitat-env`、`runtime/vendor`、上游源码和正在使用的编译／navmesh 缓存是运行依赖。`.config` 凭据不进入快照。后续诊断输出放到 `runtime/reports`，运行状态写入活动实验，不在项目文档中追加日期更新流水。
