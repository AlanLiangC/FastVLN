# Point CoT 与监督

每步输出顺序为 **APOS → OPOS → action**。动作直接使用 STOP、MOVE_FORWARD、TURN_LEFT、TURN_RIGHT、LOOK_UP、LOOK_DOWN 六个 token。APOS 是可行局部路点或转向意图，OPOS 是指令指定的可见目标点。没有单独的并行感知头或 arrival 分类头。

## 词表与坐标

与 [LightNav-0 的 dual pointing](https://github.com/lightorigins/LightNav-0) 保持同类含义，使用 48 × 27 图像网格：1–1296 是像素格中心；0 表示显式无点。APOS 还有 1297=LEFT、1298=RIGHT、1299=STOP。词表定义见 [perception.py](../src/streamnav/contracts/perception.py)。

图像 resize 保留比例，不裁切。标注与视频都使用原图归一化坐标。APOS 和 OPOS 使用独立可训练表，输入 embedding 和输出投影在各通道内共享。动作由读过两个点 token 的 hidden state 预测。

## Grounding

当前使用 COCO 实例、RefCOCO/+/g、ScanNet 室内实例、ImageNet 弱区域和已有自然图像 PixMo，见 [数据文档](datasets.md)。grounding 与导航使用相同系统 prompt 和目标指令格式。外部图像仅监督 OPOS，没有 APOS 标签时使用当前模型的 APOS 作为条件并屏蔽 APOS loss；没有动作标签时屏蔽 action loss。

真实实例掩码用于离线选择内部标注点，并保存 48×27 网格中心的真实实例归属。COCO 同类别实例取并集，指代表达只允许其绑定实例，ScanNet 只使用对应帧的可见实例。OPOS loss 使用 20% 标注点概率损失与 80% 有效区域概率损失，保留点位锚定，同时允许预测物体内部的其他点。PixMo 百分比坐标映射到原图网格，图片内容匹配标注 SHA256；只有点标注时，80% 分支允许归一化半径 0.025 内的邻近网格。多个等价点使用 `-log(sum p(valid targets))`。明确缺失的 `[0]` 不做空间扩展，未知标注不等于缺失。

ImageNet 框是弱区域，使用框内候选格集合的概率质量，loss 权重 0.25，不把框中心当成精确点。区域指标单列，不计入可见目标主指标。实例内部命中与旧的标注点距离 5% 命中同时报告。实例掩码只在离线准备监督时读取，训练模型仍输出单个 OPOS token，没有分割分支。

## 导航

教师动作来自可执行的 OVSegDT explorer / geodesic oracle；成功区域内按实际 geodesic 到达条件返回 STOP。APOS 标签来自教师路线中的可执行未来路点投影，并进行深度遮挡检查；不可见时屏蔽，转向可使用左右特殊 token。成功区域的 STOP 才监督 APOS_STOP。

深度、路径、目标视点仅用于训练服务生成标注。策略始终只读 RGB、目标文本和自身缓存。当前场景没有实例语义资产，因此默认关闭弱 3D 目标中心的 OPOS 标签，避免把投影穿墙或遮挡误差教给模型；OPOS 的真实监督来自 grounding，并在导航训练中持续混入。

## Loss

`L = L_action_CE + point_coef × (L_APOS + L_OPOS + apos_type_coef × L_APOS_type) + ppo_coef × L_joint_PPO + value_coef × L_value - ppo_coef × entropy_coef × H + grounding_coef × L_grounding`。

导航 point CE 使用实际预测历史的回放 logits；action CE 默认各有 50% 来自实际预测点前缀和标注点前缀，均对齐教师动作。这让模型在推理点不准确时也接受 DAgger 动作监督，减少 teacher forcing 的分布偏差。APOS 类型 CE 系数为 2，将 NONE、所有像素路点、LEFT、RIGHT、STOP 分成五类；路点类别 logit 为 1296 个像素 logits 的 logsumexp。它不增加参数，用于约束路点与方向/停止的概率质量；推理仍按单个 token 的 flat argmax，质量改善需要后续评估确认。PPO 使用 `log p(APOS) + log p(OPOS|APOS) + log p(action|APOS,OPOS)`，记录与回放前缀完全相同。被教师覆盖或确定性 DAgger 的动作不参与 PPO。无可靠点标签的通道及其类型 loss 为零。

APOS 在导航几何数据中学习，OPOS 在开放词汇图像数据中学习。该设计不能保证目标距离和 STOP 已自然校准；近目标混合起点补充导航的到达/停止经验，最终判断仍需普通原始起点评估。
