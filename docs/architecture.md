# 架构

唯一活动模型族为 `qwen35_pointcot`，实现见 [policy.py](../src/streamnav/models/point_cot/policy.py)，配置见 [pointcot.yaml](../configs/pointcot.yaml)。它直接加载固定 revision 的官方多模态 [Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B)。

## 当前帧与历史

24 层语言网络保留原有 18 层 Gated DeltaNet、6 层 gated attention、MLP、norm 和多模态位置编码。视觉编码器冻结，沿用官方 processor。图像保持比例缩放、不裁切；点坐标定义在原始图像的归一化坐标中。

所有原生 token mixer 的状态只用于一次决策。当前帧和目标文本构成前缀，随后自回归生成 APOS、OPOS、action。原生 GDN 的卷积与矩阵状态、attention 的 KV 在这一块内延续，下一帧清空。

6 层 attention 另加 KDA 分支，使用共享 Q/K/V/O 和小门控参数。它使用未施加当前帧 RoPE 的 Q/K，按 token 顺序更新跨帧矩阵。残差系数从零初始化，保留基座的单帧计算。跨帧只保留六个矩阵，不保留历史 RGB、历史 KV 或逐帧特征列表。

每个 episode 的长期缓存为 6 × 8 × 256 × 256 个 FP32 元素，共 **12 MiB**，不会随轨迹长度增长。新 episode 必须通过 `start_episode` 重置缓存。最终 action token 也进入记忆，使后续帧可以利用执行过的动作。

## 可训练参数

默认 LoRA rank=8、alpha=16，覆盖 attention/GDN 投影与 MLP。新增低秩 KDA 门控、两组 tied point 输入/输出表、六动作表与 critic 均训练，原始视觉与语言参数冻结。

当前实测总参数 861,432,829，可训练参数 **8,446,909（约 0.98%）**。APOS/OPOS 是新的可训练 token，不是已具备 LightNav point 能力的冻结 token；需要 Stage I grounding 和导航几何监督。

语言与冻结视觉均采用 FP32，矩阵乘法使用 TF32 high；新增表与优化器参数 FP32。批量视觉编码与逐图编码校验保持同一图像、空间位置和 prompt，视觉 FP32 减少批次形状导致的数值差异。最大视觉像素预算 114,688，输入保留比例且不裁切。吞吐报告区分单卡优化测试和八卡实际训练，不使用缓存后决策延迟替代训练 FPS。

## 训练时的打包

原生 mixer 将独立帧放在 batch 维，KDA 则把每条 fragment 的有效 token 按时间拼接。padding 不写入长期记忆；episode 边界通过 `cu_seqlens` 分段。默认 100 步 rollout，20 步反向 fragment，更新一次优化器前检查所有 fragment 的概率一致性。

PPO 和导航 point CE 使用 rollout 实际生成的两个点与动作历史；动作 IL 另混入标注点前缀的 teacher forcing，不能替换 PPO 前缀，也不能覆盖行为概率。APOS 还以聚合像素概率的类型 CE 校准路点与方向/停止分类，不增加参数。grounding 的每张静态图像都从空缓存开始，不影响导航缓存。
