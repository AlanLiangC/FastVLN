# Point CoT 与监督

每步输出顺序为 **APOS → OPOS → action**。动作直接使用 STOP、MOVE_FORWARD、TURN_LEFT、TURN_RIGHT、LOOK_UP、LOOK_DOWN 六个 token。APOS 是可行局部路点或转向意图，OPOS 是指令指定的可见目标点。另有一个 3075 参数的到达辅助头，在读取 APOS 之前预测远、近、已到达，不增加自回归 token。

## 词表与坐标

与 [LightNav-0 的 dual pointing](https://github.com/lightorigins/LightNav-0) 保持同类含义，使用 48 × 27 图像网格：1–1296 是像素格中心；0 表示显式无点。APOS 还有 1297=LEFT、1298=RIGHT、1299=STOP。词表定义见 [perception.py](../src/streamnav/contracts/perception.py)。

图像 resize 保留比例，不裁切。标注与视频都使用原图归一化坐标。APOS 和 OPOS 使用独立可训练表，输入 embedding 和输出投影在各通道内共享。动作由读过两个点 token 的 hidden state 预测。

## Grounding

当前使用 COCO 实例、RefCOCO/+/g、ScanNet 室内实例、ImageNet 弱区域和已有自然图像 PixMo，见 [数据文档](datasets.md)。grounding 与导航使用相同系统 prompt 和目标指令格式。外部图像仅监督 OPOS，没有 APOS 标签时使用当前模型的 APOS 作为条件并屏蔽 APOS loss；没有动作标签时屏蔽 action loss。

真实实例掩码用于离线选择内部标注点，并保存 48×27 网格中心的真实实例归属。COCO 同类别实例取并集，指代表达只允许其绑定实例，ScanNet 只使用对应帧的可见实例。OPOS loss 使用 20% 标注点概率损失与 80% 有效区域概率损失，保留点位锚定，同时允许预测物体内部的其他点。PixMo 百分比坐标映射到原图网格，图片内容匹配标注 SHA256；只有点标注时，80% 分支允许归一化半径 0.025 内的邻近网格。多个等价点使用 `-log(sum p(valid targets))`。明确缺失的 `[0]` 不做空间扩展，未知标注不等于缺失。

ImageNet 框是弱区域，使用框内候选格集合的概率质量，loss 权重 0.25，不把框中心当成精确点。区域指标单列，不计入可见目标主指标。实例内部命中与旧的标注点距离 5% 命中同时报告。实例掩码只在离线准备监督时读取，训练模型仍输出单个 OPOS token，没有分割分支。

## 导航

教师动作来自可执行的 OVSegDT explorer / geodesic oracle；成功区域内按实际 geodesic 到达条件返回 STOP。APOS 标签来自教师路线中的可执行未来路点投影，并进行深度遮挡检查；不可见时屏蔽，转向可使用左右特殊 token。成功区域的 STOP 才监督 APOS_STOP。

深度、路径、目标视点和实例渲染仅用于训练服务生成标注。HM3D-Semantics v0.2 与当前 RGB 场景匹配后，按 episode 全部等价目标的实例 ID 取当前可见区域并集。网格中心确实在目标区域内部才生成有效点；没有任何目标像素且全部目标 ID 均已验证存在于场景注册表时，监督 OPOS_0。极小目标、缺失实例映射或无法落入网格的目标标为未知。中心投影不能代替可见实例标注。模型不输入语义图、不预测分割，策略只读 RGB、文本、一个可选目标点提示和自身缓存。

提示与监督分开保存。训练中约一半 episode 全程 NULL，其余使用实例标签模拟检测器点，包含 20% 帧级缺失、±1 格坐标扰动和 1% 帧级干扰点。坐标扰动必须留在可见目标实例的有效网格内，越界候选退回内部 canonical 点；随机误报属于独立噪声模式，不受此约束。静态 grounding 仍使用 NULL。

`no_hint` 只将外部输入设置为独立可学习的 NULL token，表示检测器未提供信息；它不同于输出端表示目标不可见的 OPOS_0。NULL 输入不能屏蔽 OPOS loss：目标可见仍监督内部像素点，确认不可见才监督 0，未知标签才屏蔽。视频没有洋红点时需查看 `OPOS=none (pred)` 与 `GT-O`，区分正确缺失与可见目标漏检。

常规 `simulated_hint` 评估使用干净实例点：目标可见且标注有效时提供一个内部点，否则为 NULL。评估噪声由 `eval.hint_noise` 独立控制，不能直接复用训练噪声。该设置反映理想点提示的收益，不代表真实检测器成绩；与带噪声评估比较时需核对日志中的 `hint_noise`。视频 trace 记录输入点来源、是否落在目标上和 OPOS 标注状态，标注仅供诊断。

当前 [训练配方](../configs/pointcot_temporal.yaml) 继续保留 HINT 条件监督：训练服务先生成实际输入点，再按该网格中心的实例归属选择候选的合法目标视点。教师动作与 APOS 都沿此候选的可执行路线生成；OPOS 有效区域收紧到当前绑定实例，不要求精确复制输入坐标。实例 ID、导航路径和深度不进入策略输入，也不增加模型参数。

只有实际交给策略的有效点能建立或切换绑定；NULL 帧不能根据未提供给策略的真值更新候选。已有候选在最后一次有效提示后的 40 步内可继续用于接近监督，暂时不可见时 OPOS 按该实例的可见性标注。超过期限、候选不可达或离散路线不可执行时回到原任务教师。全程 NULL 的 episode 保持原有类别目标监督；已有提示历史后的 NULL 不等于从未获得提示。

到达判据仍是任务合法目标视点的测地距离 <0.25 m，接近子路点不能产生 STOP 标签。四步计划在候选绑定变化时截断。输入 HINT 原样进入 rollout、回放与 value bootstrap；同一帧的重复标签查询不重复推进教师状态。当前继续保留无提示比例、起点分布、学习率、grounding 和 PPO 主任务奖励。

## Loss

`L = L_action_CE + 0.25 × (L_APOS + L_OPOS + L_forward_intent) + 0.25 × L_future_actions + 0.2 × L_arrival + 0.1 × L_stop_calibration + 0.05 × L_apos_stop_calibration + 0.2 × L_stop_hard_negative + ppo_coef × L_joint_PPO + value_coef × L_value - ppo_coef × entropy_coef × H + grounding_coef × L_grounding`。

到达标签为测地距离 <0.25 m 的 READY、[0.25,0.75) m 的 NEAR、其余 FAR；严格使用与成功判据一致的边界。到达 CE 使用全 rollout 平方根逆频率权重，归一化前最大为 4。STOP 和 APOS_STOP 的二分类校准作用于实际预测点前缀：成功范围内为正例，范围外为负例；两侧 <0.75 m 样本同样加权 2，避免只提高正例先验。原动作 CE 的 STOP 额外权重恢复为 1。

难负例指 rollout 实际执行或条件 greedy 预测 STOP、但教师要求继续移动的帧。其 STOP 概率额外受惩罚；按完整分布式 rollout 的密度归一化，密度下限 0.02，防止孤立帧主导梯度。所有权重在切分 fragment 前确定，不在每片内重新平衡；未知到达标签屏蔽。教师 STOP、READY 与真实成功距离不一致时拒绝训练。

导航 OPOS 使用 `L_OPOS = c_visible × mean_visible(L_visibility) + c_absent × mean_absent(L_visibility) + mean_visible(L_location|visible)`，其中 `c_class ∝ sqrt(global_class_fraction)`；某一类缺失时只对现有类归一化。可见性采用平方根频率修正，在保留负类先验的同时提高少数可见帧的有效权重；例如可见/不可见各占 20%/80% 时，两类可见性 loss 的总权重为 1/3、2/3。`visibility_balance_power=0` 对应原分布，1 对应正负各一半，当前为 0.5。

可见性约束 NONE 与全部像素概率总量，条件位置 loss 由 20% canonical 点与 80% 实例有效区域组成。未知标签全部屏蔽，位置按全局可见帧数归一化；分布式 rollout 统一计算权重，不能在 fragment 内重新归一化。导航 OPOS 系数仍为 0.25。比较实验 loss 时须保持类别修正与归一化口径一致。

OPOS 的确定性解码先比较 `P(NONE)` 与全部像素的 `P(POINT)` 总量，再在可见类别内选最大概率位置，和上述可见性/条件位置监督保持一致。实际选中的 token 会进入后续动作和缓存，并非只改变视频标记。随机 PPO 学生仍从完整 token 分布采样，并使用同一分布计算 log probability。

导航 point CE 使用实际预测历史的回放 logits；action CE 默认各有 50% 来自实际预测点前缀和标注点前缀，均对齐教师动作。完整 APOS CE 已包含类型分类，因此额外类型 CE 系数为 0；APOS 在推理时使用类别概率总量，再在像素类别内选位置。可靠几何坐标及左右/STOP 标注保持原义。坐标未知且教师要求前进时，`L_forward_intent = -log(P(NONE)+P(POINT))`，允许未知可见性但抑制错误转向/停止意图，不伪造像素或缺失标签。

未来动作 CE 使用 `executed_prefix`。头部已读实际 a_t，所以三路环境都可监督实际 s_(t+1) 上的教师动作；再往后要求中间执行动作与教师相同，不能将任意学生轨迹拼成专家计划。例如 a_(t+1) 与教师不同，第二步标签仍有效，第三、四步屏蔽。当前动作已有 IL/PPO，辅助项只训练后续三步；episode 结束、STOP、rollout 末尾或候选绑定变化均截断。未来 CE 按每帧有效 horizon 平均，再按全局有有效未来标签的帧比例归一化；实际/标注点前缀各占 50%，两者都以实际动作 token 为条件。旧 `expert_only` 模式继续供历史配方恢复使用。

PPO 使用 `log p(APOS) + log p(OPOS|APOS) + log p(action|APOS,OPOS)`，不包含未执行的未来动作。专家和确定性 DAgger 流不参与 PPO 或 critic loss。未来计划每步重新生成，执行第一步；视频中的计划在预测 STOP 处截断。

APOS 使用导航几何标签；OPOS 同时接受导航实例的可见性、位置监督和开放词汇图像监督。到达与 STOP 使用上述辅助监督，近目标及停止边界起点补充接近/停止经验，最终判断仍需普通原始起点评估。视频和 trace 额外记录预测的 `P(ready)`；它不读取真值距离，也不作为强制停止门控。
