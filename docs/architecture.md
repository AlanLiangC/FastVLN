# LightNav-KDA 架构

策略以 LightNav-0 的 Qwen3-VL-4B 初始化：36 层语言主干、hidden size 2560，32 个 query heads、8 个 KV heads，head dimension 128。保留原视觉编码器、DeepStack 注入、MRoPE 和点位词表。

每个环境 step 接收当前 RGB、目标文本和该 episode 的 KDA 状态。当前图像按原处理器缩放到 448×256，为满足视频 tubelet 将同一帧重复两次；不会将前几帧作为策略输入。视觉特征在无梯度模式下计算。

全部 36 层保留当前决策内部的原生 attention。每四层中的前三层额外使用共享 Q/K/V 的 KDA 记忆残差，共 27 层；当前帧及生成 token 的局部 KV 在决策结束后释放，跨帧只保留 KDA 矩阵。零初始化残差系数使新记忆分支不会在初始化时破坏原生当前帧 attention；这不保证删除原历史输入后完整任务性能保持不变。

每层递归状态为 FP32 `(B,32,128,128)`，单环境约 54 MiB，不随 episode 长度增长。新 episode 清空缓存；BPTT 边界只断开梯度，不清空数值。

生成顺序是 `APOS → OPOS → action`。每次选择出的 token 都以对应 embedding 回送主干，OPOS 条件于实际 APOS，动作条件于两个实际点位。动作词表含六个 token，输入输出共用六个新增向量；不运行 RVQ 解码。STOP、前进、左右转的初始化参考原粗级轨迹码对应的运动含义，上下看初始化后仍需导航监督学习。

可训练参数包括各层 Q/K/V/O 的 rank-16 LoRA、rank-16 KDA 门控与残差系数、六动作向量及偏置、critic。原视觉、MLP、norm、embedding 和原 LM 输出行冻结；当前可训练约 1891 万参数，占独立参数总数的 0.42%。原 checkpoint 重复保存的相同输入/输出词表在校验完全相等后共享。视觉计算使用 BF16，语言主干和输出头使用 FP32，矩阵乘法允许 TF32 Tensor Core；可训练参数、优化器和 KDA 状态均为 FP32。底座权重数值来自同一原始检查点，仅改变存储和计算精度，不增加可训练参数。统计保存在活动训练目录的 `parameters.json`。

critic 从生成 APOS 前的状态读取。PPO 使用三 token 联合对数概率；重放保存实际点位、动作和片段初始缓存，所有片段累计梯度后统一更新一次参数。

回放按 20 帧批量执行：每帧的原生 attention 仅计算该帧的提示词、视觉和已采样 token，KDA 则按时间顺序读取有效 token。右侧 padding 不进入 KDA，episode 边界通过独立序列重置状态，梯度也在该处断开。推理仍自回归生成两个点与动作；单 token 无梯度解码使用 fused recurrent KDA，梯度回放使用 chunk KDA。

每个回放片段先检查实际采样概率；超出原 0.05 阈值时按原顺序重算整个片段，保留完整 BPTT。最终仍不满足校验或梯度非有限值时拒绝 update，日志同时记录批量原始误差与重算次数。语言 FP32 减少跨矩阵形状的舍入误差，使批量回放可以稳定工作；可以用 `float32_matmul_precision: highest` 禁用 TF32 做精度对照。
