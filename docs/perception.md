# 轻量感知监督

模型可选启用 affordance、object position 和三状态到达判别。主干仍为流式 KDA-Qwen，策略输入仍只有 RGB 与目标文本。两通道 pointing 使用 48×27 画面网格；标签和可视化均以原始 RGB 尺寸为坐标，越界点不会压到边缘。

## 模型与动作

共享 assistant-prefix hidden 预测两个点位分布和到达分布。可走点词表含空值、1296 个位置、左转、右转、STOP；目标词表含不可见及 1296 个位置。到达状态为未到达、已到附近仍需调整、教师允许 STOP。

预测分布的期望 embedding、画面坐标及特殊状态概率进入动作修正分支，与原有六动作 actor 的 logits 相加。修正分支初始化为零，保留初始化 checkpoint 的动作行为；critic 继续读取原有 hidden。训练和推理都使用模型自己的预测，不输入教师点位，不设置强制停车规则。实现保持每帧一次主干前向和固定大小递归状态，并支持现有 100 步 BPTT 与合并重放。

这是面向现有 IL/PPO 接口的双通道监督实现，不是 LightNav 的原始 LM-head 自回归输出，也不包含 RVQ 轨迹模块。

## 标注来源与限制

训练仿真可选开启与 RGB 同位姿的 metric-depth sensor。深度留在 Habitat 进程，用于生成标签，不传给策略。采集当前观测、调用教师并生成标签，然后执行动作；episode uid 和 frame id 必须匹配。传感器读出缓存随真实 STEP 更新，标注不执行额外仿真动作。

- 可走点沿教师当前 frontier／BEELINE 的可执行局部路线选择，结合 navmesh、相机投影和深度检查。不可投影的转向使用对应特殊状态；合法终点的教师 STOP 使用 STOP 状态。
- 目标位置为已有目标三维 anchor 的投影，经局部深度中值检查得到 **弱标签**，置信度 0.5。所有 anchor 明显离开视野时使用置信度 0.25 的不可见标签。遮挡、无效深度和边缘不确定性屏蔽，不当作不可见。
- 目前资产缺少 semantic mesh 和逐帧实例标注；几何深度一致不保证物体身份或完整可见性。大型、透明物体以及同类多实例仍可能产生噪声，应通过标注视频核查，必要时增加离线视觉教师。
- 到达使用现有合法目标视点距离阈值。进入阈值但教师仍调整朝向标为“附近”；进入阈值且教师输出 STOP 才标为“允许 STOP”。被实际碰撞验证为无效的教师前进，同时屏蔽对应 affordance 标签。

## 优化与诊断

感知损失独立于 EALM 的 IL/PPO 权重。位置标签在相邻网格内平滑，特殊状态保持精确类别；pointing 的位置与特殊状态分组、到达的三个状态按全局有效样本密度平衡。置信度降低弱标签的影响。没有有效标签的 rank 保留零梯度连接，不引入 DDP 未使用参数。

默认系数为 affordance 0.05、目标位置 0.10、到达 0.10；感知参数单独使用学习率 2.5e-4。训练记录标签覆盖、点位覆盖、到达各类比例、损失、准确率及感知梯度。定期评估禁用深度标注，视频展示模型预测的亮绿色圆形可走点（APOS）、亮品红菱形目标点（OPOS）、P(ready) 和原有 P(STOP)，逐帧预测写入 case trace。点位加粗黑白轮廓，标签带黑色底框；转向、Stop 和目标不可见预测仅显示状态文字。

配置入口为 `model=qwen35_0p8b_kda_pointing`、`trainer=ovsegdt_kda_pointing`，训练需增加 `+habitat.perception_labels=true`。实际训练应保存并使用完整 resolved config，以免覆盖机器人和教师设置。

从已有导航模型增加分支时使用 `checkpoint=null`、`model.checkpoint=<完整训练 checkpoint>`。可开启 `trainer.initialize_optimizer_branches=true`，按原优化器分支恢复已有 Adam moments、各 rank 的 RNG／采样器及 EALM；新增感知参数建立新状态，实验 update 从零计。初始化要求相同数据、机器人、教师与分布式 world size。后续恢复使用普通 `checkpoint=<新分支 checkpoint>`，完整恢复全部状态；禁止恢复时增删感知架构。

```bash
source scripts/env.sh
python tools/audit_perception_labels.py \
  --config runs/streamnav_active/ealm/resolved_config.yaml \
  --output runtime/reports/pointing_labels --scenes 4 --steps 32
```

此工具的目标视点起点只用于检查标注，不能作为导航成功率。模型改善需对照同一初始化、相同训练交互量的自主评估，关注成功率、到达后未停车、错误停车和连续转向。
