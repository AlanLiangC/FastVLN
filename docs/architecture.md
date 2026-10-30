# 架构与缓存

固定版本为 update 15300。模型是官方 Qwen3.5-0.8B 加 rank-8 LoRA、六层跨帧 KDA、APOS / OPOS 点头、六动作头、四步动作计划、value / arrival / HINT 质量辅助头。总参数 862,234,753，可训练参数 9,248,833；官方基座冻结。

每步输入当前 RGB、目标文本、一个可选 HINT 和 episode 状态。Habitat RGB 为 480×270，网络图像设置为 256×448；坐标接口对应原始 RGB，转换到固定 48×27 网格。输出按 APOS → OPOS → 当前动作产生，另给出四步计划、辅助预测和下一状态。

当前帧内部保留 Qwen 原生局部计算，跨帧历史由 KDA 承接。KDA 启用层为 6，跨帧张量约 12 MiB，不累计历史整帧或无限增长的视觉 token。episode 切换或加载新权重时清空状态；训练中在 fragment 起点 detach，20 步内保留梯度。

## 单点条件与运动反馈

原 HINT embedding 与 OPOS 的空间 rows 共享，并有独立可学习 NULL。15300 另加二维坐标 MLP 和 12 维运动反馈 MLP，融合到同一个 token 位置，不增加输入 token 数。NULL 时空间残差为 0，运动残差继续生效。

反馈来自上一实际动作、相邻 RGB 8×8 缩略图差异、视觉静止与交替转向计数。状态只额外保存缩略统计与少量计数。碰撞、真实位移和真值距离可供训练奖励 / 诊断使用，不进入该反馈向量。

HINT 质量头是单输出线性层，仅辅助学习和诊断，不直接门控点或替换动作。模型没有语义分割分支，也不读取深度、语义图、地图或真值位姿。

## 执行与冻结边界

原生动作词表是 STOP、MOVE_FORWARD、TURN_LEFT、TURN_RIGHT、LOOK_UP、LOOK_DOWN。预测四步计划，每步只执行当前动作再观测；规划融合与 arrival STOP 融合权重均为 0。不将规划中的未来 STOP 排队执行。

只更新 LoRA、KDA、任务头和 adaptation 参数，不更新官方 Qwen 基础参数。加载时校验基座 revision、trainable 参数名 / 形状和 tensor-content 哈希。12200 的配置关闭 adaptation，用同一加载器可作为参考模型读取；不是将新参数强行加载进旧权重。

实现见 `src/streamnav/models/point_cot/`、`src/streamnav/contracts/state.py`；相对 12200 的逐项变化见 [版本差异](changes_from_12200.md)。
