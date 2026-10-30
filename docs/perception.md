# Point CoT、HINT 与监督

每步按 APOS → OPOS → 当前动作输出，另预测未来三步动作、value、三类 arrival 和 HINT 质量。未来计划不作为实际执行动作的替代。

## 点与单个 HINT

使用 48×27 网格：1–1296 是网格中心，OPOS_0 是显式不可见；APOS 另外有 1297=LEFT、1298=RIGHT、1299=STOP，0 表示无可用几何点。APOS 表达教师路线中的可行局部路点或转向意图，OPOS 表达指令所指当前可见目标的位置。

输入只接受一个候选 HINT。0 是独立可学习输入 NULL，与输出 OPOS_0 的语义和 embedding 不同；非空点由原空间 embedding 加坐标 / 运动残差构成。HINT 质量头不直接门控动作。所有坐标、标注和视频使用相同原始 RGB / 网格映射，橙色框只是点标记，不是目标边界框。

## 标签

导航动作来自可执行 OVSegDT explorer / geodesic oracle；成功区域内按实际到达条件返回 STOP。APOS 从真实可执行路线的未来路点投影并检查深度遮挡，无法可靠投影时屏蔽；转向可用特殊 token。成功区域才监督 APOS_STOP。

OPOS 从当前官方目标语义实例可见区域生成，最小可见区域为 9 像素；不可见或无可靠目标区域时显式监督 NONE，未知实例映射按有效性屏蔽。语义标签用于训练监督，没有分割 loss 或语义图 actor 输入。

实际输入的真实 / 模拟点送入教师绑定，episode / frame 不匹配拒绝执行。错误点和同类未列入官方目标的歧义实例区别处理：后者屏蔽质量负例监督。到达判据是官方目标视点测地距离 <0.25 m，不是到达任意 HINT 或路径中间点。

## Loss

动作 CE 对实际点 / 标注点前缀各 50%；point loss 使用实际预测历史，并结合 Grounding 的可见性与位置监督。OPOS 可见 / 缺失按完整 rollout 平方根平衡；位置结合可靠支持域与 anchor。APOS CE 已包含类型分类，额外类型系数为 0；未知几何且教师要求前进时，用 NONE+POINT 的概率监督前进意图，不伪造坐标。

arrival 为 FAR、NEAR（0.25–0.75 m）、READY（<0.25 m）。三类 CE 平方根逆频率平衡；动作 STOP 和 APOS_STOP 的 BCE 在阈值两侧近距离都加权，不单独提高正例先验。binary 难负例作用于实际执行或条件 greedy 误选 STOP、但教师要求继续的帧：`softplus(z_STOP - logsumexp(z_其他动作))`。全局密度归一化下限为 0.02，避免一个孤立误停主导梯度。

HINT 质量 BCE 系数为 0.1。PPO 只对随机学生的实际 APOS / OPOS / action 联合概率训练；未来三步按因果 executed_prefix CE 监督，合法性边界见 [训练文档](training.md)。所有有效帧权重在完整分布式 rollout 确定，再切 fragment，不能逐片重新平衡。

当前系数、起点课程和奖励见 [训练配方](training.md)，相对 12200 的新增部分见 [版本差异](changes_from_12200.md)。
