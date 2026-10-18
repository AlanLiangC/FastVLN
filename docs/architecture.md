# 架构

唯一活动模型族为 `qwen35_pointcot`，实现见 [policy.py](../src/streamnav/models/point_cot/policy.py)，配置见 [pointcot.yaml](../configs/pointcot.yaml)。它直接加载固定 revision 的官方多模态 [Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B)。

## 当前帧与历史

24 层语言网络保留原有 18 层 Gated DeltaNet、6 层 gated attention、MLP、norm 和多模态位置编码。视觉编码器冻结，沿用官方 processor。图像保持比例缩放、不裁切；点坐标定义在原始图像的归一化坐标中。

所有原生 token mixer 的状态只用于一次决策。当前帧和目标文本构成前缀，随后自回归生成 APOS、OPOS、action。原生 GDN 的卷积与矩阵状态、attention 的 KV 在这一块内延续，下一帧清空。

6 层 attention 另加 KDA 分支，使用共享 Q/K/V/O 和小门控参数。它使用未施加当前帧 RoPE 的 Q/K，按 token 顺序更新跨帧矩阵。残差系数从零初始化，保留基座的单帧计算。跨帧只保留六个矩阵，不保留历史 RGB、历史 KV 或逐帧特征列表。

每个 episode 的长期缓存为 6 × 8 × 256 × 256 个 FP32 元素，共 **12 MiB**，不会随轨迹长度增长。新 episode 必须通过 `start_episode` 重置缓存。最终 action token 也进入记忆，使后续帧可以利用执行过的动作。

## 可训练参数

默认 LoRA rank=8、alpha=16，覆盖 attention/GDN 投影与 MLP。新增低秩 KDA 门控、两组 tied point 输入/输出表、六动作表与 critic 均训练，原始视觉与语言参数冻结。

当前实测总参数 861,964,413，可训练参数 **8,978,493（约 1.04%）**。四步规划分支为 529,536 个参数，可选目标点输入新增 2048 个参数。APOS/OPOS 是可训练 token，需要 grounding 和导航实例监督。

每帧前缀末尾固定追加一个目标点提示 token。有点时复用 OPOS 坐标表并加输入来源向量；没有信息时使用独立可学习 NULL，不复用输出端表示不可见的 OPOS_0。该 token 不增加视觉编码或输出解码次数，参与当前帧 prefill 和 KDA，必须随实际帧保存在 rollout 中并原样回放。NULL 不提供目标可见性的真值。

基座、视觉与新增表以 FP32 保存，语言 / KDA 使用 FP32 并启用 TF32 high；视觉前向遵循调用处的 autocast，导航外层为 BF16 autocast。参数 dtype 不等同于每个算子的执行 dtype。批量视觉编码与逐图编码保持同一图像、空间位置和 prompt，并用工程验收检查数值一致性。最大视觉像素预算 114,688，输入保留比例且不裁切。无梯度批量策略使用 CUDA Graph 减少调度，训练回放保留完整梯度，见 [训练文档](training.md)。吞吐报告区分单卡优化测试和多卡实际训练，不使用缓存后决策延迟替代训练 FPS。

## 训练时的打包

原生 mixer 将独立帧放在 batch 维，KDA 则把每条 fragment 的有效 token 按时间拼接。padding 不写入长期记忆；episode 边界通过 `cu_seqlens` 分段。默认 100 步 rollout，20 步反向 fragment，更新一次优化器前检查所有 fragment 的概率一致性。实际点前缀与教师点前缀分支依次前向、反向，累加梯度后才执行同一次参数更新，避免同时保留两套时序激活。

PPO 和导航 point CE 使用 rollout 实际生成的两个点与动作历史；动作 IL 另混入标注点前缀的 teacher forcing，不能替换 PPO 前缀，也不能覆盖行为概率。APOS 先按 NONE/POINT/LEFT/RIGHT/STOP 的概率总量选择类别，再在 POINT 内选位置；OPOS 先判定 NONE/POINT，再选择像素位置。两个阶段共同确定真正送入动作预测和缓存的 token。坐标未知的前进样本只约束 APOS 为 NONE 或 POINT 的兼容意图，不生成像素真值。grounding 的每张静态图像都从空缓存开始，不影响导航缓存。

四步计划的第一步是原动作头。追加实际动作 token 后，将其 hidden state 输入 LayerNorm → 128 维瓶颈 → 三个 1024 维残差特征，并共享动作输出表，预测后续三步。该分支没有再次调用语言模型，不追加未来 token，不修改 episode cache。它借鉴 Uni-NaVid 的四动作监督，但采用小型并行预测分支，保持当前帧与 KDA 缓存的接口。
