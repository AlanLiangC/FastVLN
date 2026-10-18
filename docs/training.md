# 训练与评估

当前配置为 [pointcot.yaml](../configs/pointcot.yaml)，模型与监督接口分别见 [架构](architecture.md) 和 [Point CoT](perception.md)。当前节点为 `checkpoints/versions/qwen35_pointcot_20261017`：navigation 固定 update 5155，best 固定 update 4400。活动运行继续写入 `runs/qwen35_pointcot_opos_hint`。

## 导航训练

```bash
source scripts/env.sh
# 当前八卡运行恢复；保留权重、Adam、数据游标和 PPO 进度
bash scripts/start_eight_gpu_training.sh \
  --resume runs/qwen35_pointcot_opos_hint/checkpoints/latest --skip-initial-eval
# 从固定节点恢复到新目录
STREAMNAV_RUN_DIR=runs/navigation_from_node bash scripts/start_eight_gpu_training.sh \
  --resume checkpoints/versions/qwen35_pointcot_20261017/navigation --skip-initial-eval
# 保存完整 update 后暂停
bash scripts/stop_training.sh runs/qwen35_pointcot_opos_hint
python tools/training_status.py --root runs/qwen35_pointcot_opos_hint
```

| 项目 | 当前设置 |
|---|---|
| GPU / 环境 | 默认 GPU 0–7；每卡专家、greedy DAgger、随机 PPO 各 1 个环境 |
| rollout / fragment | 100 步 / 20 步 |
| update | 2400 条转移，2 次同步 mini-batch optimizer step |
| 预算 | 累计 update 10000 |
| PPO | 系数 0.25，clip=0.2；仅随机学生参与 PPO 和 critic |
| 动作 IL | 实际预测点前缀与标注点前缀各 50%，三路环境都接受教师动作 CE |
| 辅助项 | point=0.25，未来动作=0.25，grounding=0.5 |
| grounding 混合 | 每 rank 每 mini-batch 4 条，八卡每 update 64 条 |
| 优化器 | fused AdamW，梯度同步后裁剪到范数 1 |
| 保存 / 评估 | 每 5 / 100 update，最近 2 个检查点及 best、配方初始检查点保留 |

LoRA LR=1e-5，KDA / point LR=5e-5，action / critic LR=1.5e-4，planning LR=3e-4，hint LR=1e-4。PPO 原计划为预热 50 update，再用 200 update 升至 0.25；当前延续的 `schedule_origin_update=1403` 已进入完整 PPO 阶段，恢复不重新预热。

一个环境 step 是一次当前帧决策及仿真动作；一个 update 汇集全部环境的 100 步转移后更新参数。100 不是 episode 长度：episode 最多 500 步，可跨 update 延续。KDA 只在反向 fragment 起点 detach，20 步内保留梯度。六卡每 update 为 1800 条转移，跨卡数比较使用累计 `frames`。

动作 CE 对碰撞恢复、STOP 样本的加权在整个分布式 rollout 上归一化。PPO 使用实际 APOS、OPOS、当前动作的联合概率，专家覆盖动作与 greedy DAgger 不使用伪造的 on-policy 概率。四步规划只用专家连续轨迹后续三步作辅助监督，跨 STOP、episode 或 rollout 边界的标签屏蔽；部署执行当前一步后重新观测规划。

## 场景、起点与提示

使用 HM3D-OVON train 和官方 HM3D-Semantics 实例资产，路径与准备方式见 [数据文档](datasets.md)。缺少实例资产或整次 rollout 的 OPOS 有效标签为零时拒绝训练。机器人 `allow_sliding=true`，高 0.88 m、半径 0.18 m，前进 0.25 m、转向 30°，成功距离 0.25 m。

训练 reset 按 episode 选择 50% 原始起点、35% 距目标合法视点 geodesic 0.5–3 m 的 near 起点、15% 距离 0–0.2 m 的 ready 起点。near 有 50% 概率朝向目标；ready 朝向目标并检查可见性。采样失败保留原始起点。比例针对 reset，不代表转移或 STOP 标签比例。评估只用原始起点。

每 rank 的 sampler 按约 2000 次环境 step 驻留场景，带 20% jitter。约一半 episode 全程 NULL，其余用训练实例点模拟提示，并加入 20% 缺失、±1 格内部扰动、1% 随机误报。NULL 只表示没有外部信息，目标可见时仍监督 OPOS 像素位置。训练深度与实例图仅用于生成标签，不进入策略输入。

## 计算与回放校验

每卡环境共享批量视觉编码与 APOS → OPOS → action 推理；padding 不进入有效 recurrence，各环境缓存独立。关闭常规语言梯度检查点，实际与教师前缀分别前向、反向，累积到同一组梯度后才执行原有 optimizer step。监督、归一化和混合比例保持一致；浮点累加顺序可以产生微小差异。

推理启用 CUDA Graph，按形状、序列边界和 LoRA 状态缓存，最多 16 个图。缓存满时先释放旧图；输出状态复制到各环境独立存储。FLA 序列索引在图内生成，避免图引用被释放的临时缓存。图只用于无梯度推理，有梯度回放保持原计算路径。

文本模板、token ID 和图像网格位置按指令缓存，LRU 各 256 条；每帧仍使用当前 RGB 编码。视觉编码与教师 / 实例标签 RPC 重叠，标签返回后填 hint token；所有 RPC 完成后才 step/reset。教师不可用导致重置时丢弃预取特征。

第一次参数更新前，完整 rollout 的行为概率回放误差必须 ≤0.05。打包误差超限时依次尝试逐步回放和原三环境布局的批量回放，保持采样 token、padding、独立缓存与重置。后备批量路径使用逐帧 activation checkpoint，释放推理图缓存以控制显存。仍不一致时保存诊断并退出，不放宽阈值或跳过错误样本。NaN/Inf 梯度在 optimizer step 前拒绝。

第二个 mini-batch 使用新策略对旧行为概率计算 PPO，允许合法的概率变化。关键字段为 `replay_log_prob_error_max`、`differentiable_replay_error_max`、`policy_log_probability_change_max`、`ppo_clip_fraction`、`optimizer_steps`。原生 GDN 与 KDA 的采样和回放使用一致的 chunk 方程。

验证工具包括 `tools/validate_behavior_replay.py`、`validate_inference_graphs.py`、`validate_plan4.py` 和 `validate_replay_efficiency.py`，输入样本位于 `runtime/fixtures/navigation/`。样本是工程验证数据，不是导航性能结果；输出写入 `runtime/reports/`。

## 保存、恢复与变更卡数

检查点包含 FP32 adapter、Adam、每 rank 的 RNG、导航 / grounding 采样状态、累计 frames、配方、基座 revision、数据哈希及源码快照。仿真与 KDA 状态恢复时重新创建，不能保证逐帧复现中断前的轨迹。超过恢复检查点的未保存日志会移入 `interrupted/` 并记录 `resume_events.jsonl`，避免同名 update 的两条分支混在曲线中。

`--resume` 检查原数据与监督配方。需要修改监督时使用 `--fork-from CHECKPOINT --keep-schedule`，保持 Adam 和 PPO 进度，并记录配方分叉。`--initialize-from` 仅加载适配权重，重新初始化导航优化器与采样器。

```bash
# 六卡检查点扩为八卡，或其他显式卡数迁移
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/start_eight_gpu_training.sh \
  --resume runs/qwen35_pointcot_opos_hint/checkpoints/latest \
  --allow-world-size-change --skip-initial-eval
# 缩为六卡
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 STREAMNAV_WORLD_SIZE=6 \
  bash scripts/start_pointcot_training.sh \
  --resume runs/qwen35_pointcot_opos_hint/checkpoints/latest \
  --allow-world-size-change --skip-initial-eval
```

卡数变化保留参数、Adam、update、frames 和 PPO 计划；存续 rank 保留各自 RNG / 两个 sampler，新增 rank 使用基种子、update 和 rank 派生独立种子。迁移写入 `world_size_changes.jsonl`，首个完整 update 立即保存新 rank 集合。之后同卡数恢复无需再次加迁移参数。当前运行沿用原 update 预算，卡数变化不自动缩放学习率。

## 评估与视频

每次评估在相同原始起点分别运行 `no_hint` 与 `simulated_hint`。每 split 日常固定 12 条，每 500 update 扩为 48 条，最多 500 步；每卡最多 8 个并行环境，日常最多 2 个。模拟提示只在当前目标可见且实例标签有效时给干净内部点，其余为 NULL；它不代表真实检测器性能。

无提示指标写 `eval_metrics.jsonl`，作为 best / health 主成绩；有提示写 `eval_hint_metrics.jsonl`。SR 采用 geodesic 距离 <0.25 m 且执行 STOP，`success_strict_0_1` 是另外的 0.1 m 口径。`oracle_success` 表示这条模型轨迹曾进入成功距离，不是另跑专家的成功率。

较大评估包含同样的日常前 12 条，单列 `selection_success`；best 按这组固定 anchor 的三个 split 平均 SR 选择。比较总体成绩时区分 36 与 144 条；已有小样本 best 不等于更大评估上的最优模型。训练混合近目标起点的成功率不能替代原始起点评估。

每个设置 / split 保存两个固定视频，共 12 个，目录为 `evaluation/update_XXXXXXX/{no_hint,simulated_hint}/`，包含 MP4、trace、播放页和汇总。点位与标注口径见 [部署文档](deployment.md)。独立评估应指定新的 `run_dir`，保留活动训练的原始评估结果。

monitor 每 30 秒更新 `health_status.json`、GPU 与 worker 进展。非有限数值、worker 异常、长时间停滞和持续感知 / 行为退化会触发检查或保存后停止。它只向当前 launcher 的存活 worker 请求正常退出，不直接中断 checkpoint I/O。完整指标和轨迹是判断收敛的依据。

## Grounding 阶段

已完成运行保留在 `runs/pointcot_grounding_refined`，best 为 800、最后一次为 2000。可用其 best 初始化新的导航运行：

```bash
STREAMNAV_RUN_DIR=runs/navigation_new bash scripts/start_eight_gpu_training.sh \
  --initialize-from runs/pointcot_grounding_refined/checkpoints/best
# 从冻结基座重新做 grounding，使用新目录和显式预算
STREAMNAV_RUN_DIR=runs/grounding_new bash scripts/start_grounding_training.sh --updates 2000
```

Grounding 默认 GPU 0–5、每卡 batch 32，全局 192。`--updates` 优先于 `grounding.pretrain_updates`；仅两者都未设置时才按 `pretrain_epochs` 换算。带来源权重的采样不代表每张图恰好遍历同样次数。恢复已有 grounding 使用 `--resume`；`--initialize-from` 只载入 adapter，并重新建立 optimizer、sampler 与学习率曲线。

静态图片每次从空缓存开始，只监督 OPOS；APOS 无标注时使用当前预测作为前缀并屏蔽 APOS loss，没有动作或 STOP loss。来源、区域监督、缺失采样和防泄漏见 [数据文档](datasets.md)。当前小学习率配方前 25 update warmup，再 cosine 降至 10%，梯度范数裁剪 1。

每 200 update 用固定分层的 1536 条查询验证，保存 32 个案例与全部预测。best 按真实点来源的宏平均可见目标定位选择，明确缺失准确率须 ≥85%；弱区域指标单列。图像保持比例、不裁切，批量冻结视觉编码，FP32 参数、语言 FP32/TF32，关闭常规梯度检查点。

## 项目保留规则

活动导航与已完成 grounding 的训练曲线、评估、源码、必要检查点保留；固定版本位于 `checkpoints/versions/`。数据、模型基座、安装环境、凭据、navmesh 与编译缓存不作实验垃圾清理。`runtime/reports/` 只保留当前验收、数据审计、恢复和性能分析；验证输入独立保存在 `runtime/fixtures/`，历史 canary、失败尝试、一次性对照输出不作为长期依赖。
