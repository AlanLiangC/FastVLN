# 训练与评估

配置见 [pointcot.yaml](../configs/pointcot.yaml)。Stage I grounding 已完成，Stage II 导航从改进训练的最佳 update 800 权重初始化。

## Stage I：grounding

```bash
source scripts/env.sh
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 STREAMNAV_WORLD_SIZE=6 bash scripts/start_grounding_training.sh --initialize-from runs/pointcot_grounding/checkpoints/update_0004400
```

当前使用 GPU 0–5，GPU 6、7 留给其他任务；grounding 启动脚本默认选择这六张卡。每卡 batch 32，全局 batch 192。当前配置从保留的 4400 权重进行 2000 次改进更新，约 38.4 万次查询采样；update 从 0 重新计数。`--initialize-from` 只加载 adapter，优化器、采样器和学习率曲线重新初始化，来源权重和数据可显式变化；`--resume` 则严格恢复原配方和状态，二者互斥。完整基座预训练需要另设预算和学习率。带来源权重的采样不是每张图恰好遍历固定次数。输入、来源数量、采样比例和防泄漏规则见 [数据文档](datasets.md)。

基座冻结，OPOS loss 为 20% canonical 点概率损失 + 80% 实例有效区域概率损失；只有点标注时区域分支使用半径 0.025 的网格邻域。LoRA、KDA 和点表参与更新；APOS 未标注时使用当前模型的预测前缀并屏蔽 APOS loss。没有动作或 STOP loss。弱框使用区域概率质量，权重 0.25。新增动作/critic 在这一阶段没有有效监督。

使用八线程图像读取、批量冻结视觉编码、FP32 + TF32 high 和 fused AdamW，关闭 grounding 梯度检查点。视觉 FP32 减少 BF16 编码随批次形状产生的数值差异，图片仍保留比例且不裁切。视觉像素预算最多 114,688，避免极端长宽比图片产生过长序列。每张图片独立重置缓存，batch padding 不进入 KDA 的有效序列。

LoRA/KDA LR=1e-5、point LR=5e-5，当前改进训练前 25 update 线性 warmup，随后 cosine 降到初始 LR 的 10%；梯度范数裁剪 1。每次同步更新前汇总所有 rank 梯度。日志记录来源数量、缺失和弱区域样本数量、loss、学习率比例、最慢 rank 的更新耗时及最大显存。

启动先保存、评估 update 0，并将其纳入最佳权重选择，避免改进训练退化后覆盖初始化基线。之后每 200 update 保存和验证，固定验证 1,536 条查询，按来源和标签类型分层；完成预算后再验证。分别报告可见点的精确格命中、归一化 5% 半径命中、明确缺失命中、弱区域命中及分来源结果。最佳权重按四个真实点来源的宏平均可见目标定位选择：有实例 support 时检查预测 token 是否在实例内部；只有点标注时使用 5% 距离。旧的 canonical 点距离指标保留用于跨版本比较，弱区域与缺失准确率不能抬高这一主指标。新最佳权重还须满足明确缺失准确率 ≥85%，防止通过总是预测存在来抬高可见目标成绩；通过门槛后，缺失命中用于同分比较。

每次 eval 使用实际确定性 APOS 前缀计算 OPOS 的混合 loss、canonical NLL 和 support NLL，分来源记录；同时报告可见目标误判缺失率与缺失目标误判存在率，不额外运行第二次视觉编码或模型推理。每次 eval 保存 32 个案例：12 个分层固定，其余按预测结果选择。复用同一验证预测，目录为 `runs/pointcot_grounding_refined/grounding_visualizations/update_XXXXXXX/`，包括原图、标记图、`overview.jpg`、`index.html`、`cases.json` 和全部查询的 `predictions.jsonl`。洋红菱形为预测 OPOS，青绿圆圈为标注点，绿色轮廓为真实实例有效网格，橙框为弱区域；没有真值的 APOS 不绘制。标记有白边，原图亮度保持不变。

```bash
make viewer VIEWER_RUN_DIR=runs/pointcot_grounding_refined
# http://127.0.0.1:8765/grounding_visualizations/
```

训练日志、逐更新指标和验证指标分别为 `training.log`、`train_metrics.jsonl`、`grounding_eval.jsonl`。`training_progress.json` 和 `progress/rank_*.json` 记录加载、训练和评估进度；后台 monitor 每 30 秒更新 `health_status.json` 与 GPU 状态。grounding 监控使用点指标，非有限 loss/梯度、worker 异常退出或长时间无进展才建议暂停，不套用导航成功率的停止规则。使用 `python tools/training_status.py --root runs/pointcot_grounding_refined` 查询。

检查点保留最近两个与最佳权重，包含优化器、每 rank RNG/采样状态、配方、数据哈希和源码快照。`tools/watch_grounding_visualizations.py` 仅用于已有 checkpoint 补图，新训练直接在 eval 内输出可视化。

同一批 COCO 样本的单卡测试：串行视觉编码、batch 8、梯度检查点开启时约 14.4 查询/秒；批量视觉编码、batch 32、关闭检查点时约 34.0 查询/秒，batch 64 显存不足。八卡混合来源的实际吞吐与该单卡受控测试分别记录在 `runtime/reports/grounding_training_launch.json` 和 `grounding_benchmark_*.json`；评估及保存耗时另计，不将短程更新吞吐直接当作全程速度。

## Stage II：DAgger/PPO + grounding

```bash
bash scripts/start_pointcot_training.sh --initialize-from runs/pointcot_grounding_refined/checkpoints/best
```

导航启动脚本默认使用 GPU 0–5 的六卡训练，GPU 6、7 保留。每卡两个环境，全局每 update 收集 1200 条导航转移，正式训练预算为 10000 update。

默认每 rank 两个环境，一个持续执行专家动作，一个学生环境。DAgger 对两路访问状态都做教师动作 CE，混合实际点前缀与标注点前缀各 50%。point CE 使用实际预测点历史的回放 logits，对齐真实 rollout 的缓存；APOS 另加系数 2 的类型 CE，比较 NONE、像素路点、LEFT、RIGHT、STOP，像素路点使用 1296 个格子的概率总和。未知标注屏蔽这两项 loss，推理保持 flat token 解码。学生早期可混合教师动作；PPO 在 warmup 结束后才启用，此时学生环境完全按随机学生策略行动。专家环境和确定性 DAgger 环境不参与 PPO。

默认 rollout=100、fragment=20。一个环境 step 是一次当前帧决策和仿真动作；一个 update 收集 `100 × num_envs × world_size` 条转移，进行一次同步参数更新。100 不是 episode 长度：episode 最多 500 步，可跨 update 延续。KDA 缓存在 fragment 起点 detach，20 步内反向传播。

每卡环境共享一次批量视觉编码和一次批量自回归决策，保持 APOS → OPOS → action 顺序。不同指令长度的 padding 不进入原生 delta recurrence 或 KDA；每个环境独立保存、重置 cache。评估复用同一批量推理路径。默认关闭导航梯度检查点：20 帧回放的单卡峰值约 32 GiB，换取较少的重复计算。该计算开关允许在恢复训练时变化，权重、Adam、采样器和已完成 update 仍恢复；模型与监督配方的校验继续保留。

PPO 系数在前 100 个 update 为零，随后 200 个 update 线性升到 0.5。使用固定 ramp，避免低成功率使 PPO 一直关闭。策略使用两点和动作的联合概率，clip=0.2。优势只在随机学生样本上归一化。默认超时不 bootstrap，与 OVSegDT 任务截断口径一致。

每 rank 每个 update 继续混入 4 条 grounding 样本，全局 24 条，系数 0.5。默认 LoRA LR=2e-5、KDA LR=1e-4、point LR=1e-4、action/critic LR=3e-4；LoRA 和点表使用较小步长保护 grounding 初始化，KDA、动作与 critic 继续学习导航。梯度范数裁剪 1。所有 fragment 回放误差必须小于 0.05，打包误差过大则回退逐步回放，再次失败会拒绝更新。

## 混合起点

每次训练 reset：50% 保留原始起点，35% 采样 geodesic 距离 0.5–3 m 的近目标起点，15% 采样距离 0–0.2 m 的到达区域起点。这里距离指到合法目标视点集合的 geodesic，成功阈值为 0.25 m。near 起点 50% 朝向目标附近、50% 随机朝向；ready 起点朝向目标并校准相机俯仰，检查目标锚点未被遮挡，避免空缓存、背对目标的图像被直接标为 STOP。

采样在当前机器人 navmesh 上直接选位置，检查楼层和实际距离；失败则保留原始起点并记录 fallback。ready 样本仍按真实成功条件标注 STOP。评估严格使用原始起点，禁止 curriculum metadata；仿真启用 `allow_sliding=true`，机器人高度/半径为 0.88/0.18 m。

## 保存、暂停和恢复

```bash
# 保存后暂停 grounding
bash scripts/stop_training.sh runs/pointcot_grounding_refined
bash scripts/stop_training.sh runs/qwen35_pointcot_active
bash scripts/start_pointcot_training.sh --resume runs/qwen35_pointcot_active/checkpoints/latest
```

checkpoint 包含 FP32 adapter、优化器、每 rank RNG 和两个数据源的采样状态、配置、基座 revision 与源码快照。导航恢复需同一 world size、数据和训练配方；仿真进程与缓存重新创建，不能保证复现中断前的逐步轨迹。`--initialize-from` 仅加载权重，重新初始化导航优化器。grounding 可通过其启动脚本的 `--resume` 恢复。需要缩减卡数时显式加 `--allow-world-size-change`，保留存续 rank 的 RNG/采样游标、权重、Adam 状态和累计样本数。训练终点与学习率曲线按原采样预算换算，避免换卡数后进度跳变；增加卡数仍会拒绝恢复。检查点与 resolved config 记录卡数变化。

导航调整监督配方时用 `--fork-from CHECKPOINT`，保留权重、Adam 与采样状态，并在 `recipe_forks.jsonl` 记录来源和原配方。同目录继续须从该目录最新检查点开始；首个新配方 update 即保存，以便后续严格 `--resume`。当前 APOS 配方从 update 119 继续，首次新配方保存为 update 120。

默认每 25 update 保存，每 100 update 评估，保留近期权重和最佳导航权重。导航日志记录各流成功率、碰撞、动作分布、教师 STOP 比例、起点类型、loss、PPO 系数及回放误差。`training_status.py --root DIR` 查询状态。

APOS 的几何监督看 `apos_geometry_labels`、`apos_geometry_point_labels` 和左右/STOP 标签数量；`apos_loss` 按全部转移归一化，`apos_loss_per_labeled_frame` 按有标签的转移归一化。新增采样/greedy 格命中率与 `apos_parameter_gradient_norm`（同步、裁剪后的点表梯度范数）。`point_teacher_frames` 及对应旧 teacher agreement 字段仅指旧版视觉教师，不表示几何监督覆盖率；没有视觉教师时它们为 0/null。rollout 分项记录 oracle、批量策略、环境/重置和 bootstrap 耗时，各分项的全局值取最慢 rank，不能直接相加当作同一 rank 的总耗时。

像素路点与 LEFT/RIGHT/STOP 的 greedy 成绩分开记录；像素路点同时报告精确格与归一化 5% 距离命中，避免转向分类掩盖真正的路点定位能力。真实导航帧的监督与批量推理验证使用 `tools/validate_navigation_batching.py`；单卡两环境完整推理实测约快 1.81 倍，六卡热身后的完整 update 从约 57 秒降至约 33 秒。启动、冷缓存、存盘和完整评估另计，不能把单卡推理倍数当作整个训练的倍数。

`apos_type_loss` 记录类型 CE，`apos_teacher_prefix_loss` 仅诊断真值历史下的点 CE，不参与当前 point loss；类型 loss 也按全部转移归一化。`tools/validate_navigation_supervision.py` 使用已保存的真实帧检查实际前缀 CE、APOS 类型梯度、联合 PPO 梯度和基座冻结。

## 测评和视频

每 split 固定分层抽 48 条原始 episode，最多 500 步，不使用教师动作、深度标签或近目标采样。三个 split 分别报告 SR、SPL、距离、碰撞与吞吐，完整 split 与诊断子集不能混为一个成绩。

```bash
python -m streamnav.evaluation.runner checkpoint=runs/qwen35_pointcot_active/checkpoints/best
make viewer
```

每个 eval 保存代表案例视频和 trace。APOS 荧光绿、OPOS 洋红，带白色边框，并与模型决策的原始帧对齐。失败视频与成功率应同时分析，不能用辅助专家流成功率代替学生评估。

短程 canary 只验证数据、更新、PPO、保存和视频链路。短程 grounding 和少量导航更新不足以判断收敛或导航上限；应以正式 grounding 验证与导航原始起点评估判断效果。
