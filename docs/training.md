# 15300 训练配方与恢复

固定权重为 `checkpoints/fastvln_15300`；配方为 [pointcot_15300.yaml](../configs/pointcot_15300.yaml)。冻结版本的原训练已停止；当前改进分支见下文。完整八卡 Adam、RNG、导航 sampler 和 Grounding sampler 状态保留，update=15300，累计 frames=32,758,920，PPO schedule origin=1403。

## 三路混合训练

每卡专家、greedy DAgger、随机 PPO 各一个环境，rollout=100、反向 fragment=20；八卡每 update 共 2400 条转移，两次同步 mini-batch 参数更新。100 步是采样窗口，episode 上限为 500 步，未结束的轨迹跨 update 继续。专家流严格执行教师动作，DAgger 在学生访问的状态接受 IL，随机学生流参与 PPO 和 critic。

PPO 系数 0.25、clip=0.2，联合概率包含实际采样的 APOS、OPOS 与当前动作，不包含未执行的未来动作。value=0.5，entropy=0.005，梯度同步后裁剪到 1。恢复时沿用原 Adam / 数据游标和已完成的 PPO 预热，不重新启动 warmup。

实际点前缀与标注点前缀的动作 IL 各占 50%。未来动作采用 `executed_prefix`：a_t 后监督真实下一状态的教师动作，更远 horizon 只在中间实际动作与教师一致时有效；episode、STOP、rollout 和 HINT 绑定变化均截断。缓存按真实执行历史更新。

## 配方常量

| 项目 | 值 |
|---|---|
| point / future / grounding | 0.25 / 0.25 / 0.5 |
| arrival / STOP BCE / APOS_STOP BCE / binary 难负例 | 0.2 / 0.1 / 0.05 / 0.2 |
| 碰撞恢复 / 停滞 / 错误点 IL 加权 | 3 / 2 / 1.5 |
| STOP IL 额外权重 | 1 |
| 原起点 / near / ready / boundary | 50% / 20% / 10% / 20% |
| near / ready / boundary 距离 | 0.75–3 / 0–0.25 / 0.25–0.75 m |
| Grounding | 每 rank 每 mini-batch 4 条；八卡每 update 64 条 |
| 保存 / 评估 | 每 5 / 100 update，训练输出保留最近 2 个及所选 best |

允许沿障碍滑动。成功为测地距离 <0.25 m 且执行 STOP；终止成功 +5，错误 STOP −5，进展系数 0.1，每步 −0.001，碰撞 −0.03。语义实例资产、距离与深度用于奖励 / 教师标签，不进入 actor。学习率与来源混合见 [版本差异](changes_from_12200.md)。

## 显式续训

```bash
source scripts/env.sh
# 当前没有训练在运行；需要继续时才执行
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 STREAMNAV_WORLD_SIZE=8 make train
# 输出到 runs/finetune_15300；默认从固定 15300 恢复，已有该输出的 latest 时恢复它
python tools/training_status.py --root runs/finetune_15300
bash scripts/stop_training.sh runs/finetune_15300
```

固定发布目录不会被后续训练覆盖。换卡数时显式传 `--allow-world-size-change`：存续 rank 保留各自状态，新增 rank 使用独立种子，并记录帧预算变化。精确八卡恢复不需要迁移参数。

预训练入口 `make grounding` 使用公共 Grounding 数据；`make pipeline` 是从 Grounding 完成后开始导航的独立新训练流程，不是加载当前发布权重的必要步骤。其输出在新的 run 目录，不能当作已发布的导航模型。

## 评估和视频

三个设置为 no_hint、simulated_hint、yoloe_point，日常每 split 8 例，完整评估配方为每 split 32 例。保存的 15300 训练期样例实际每设置 24 例；独立对比为每设置 300 个任务，分别保留，不能横混排名。

未来显式续训仍按 simulated_hint 选 best，另按 yoloe_point 选 best_real_hint。这些链接只属于新训练输出，默认部署始终固定到 `checkpoints/fastvln_15300`。视频每设置每 split 两个，含 APOS / OPOS / 实际 HINT、action 和辅助概率。当前只保留 15300 样例与 300 例对照。


## 从 15300 继续：Grounding 与 HINT 鲁棒性

新配方为 `configs/pointcot_grounded_hints.yaml`，输出为
`runs/finetune_15300_grounded_hints`。八卡从固定 15300 的 Adam、数据游标、
RNG 及 PPO schedule 继续，目标 update 22000；冻结权重和 12200 对照均不覆盖。
`make train` 启动此分支，已有 latest 时精确恢复。

这次修改依据五个 split、每 split 300 个相同任务的四臂对照：15300 的
YOLOE Unseen SR 为 37.33%，NULL 为 35.00%；12200 对应 39.33% / 39.00%。
各项多重比较校正后均未显著，不能凭几百分点断言版本全面退化。
错误 HINT 提前 STOP 的实例，以及公共 Grounding 未使用导航 NULL token 的
输入差异，则是可以直接验证的训练问题。Seen / Synonyms 和无提示能力同时监控。

改动范围：

- 公共 Grounding 的每张图使用与导航相同的 NULL HINT 槽。对有完整实例掩码或
  明确目标缺失标注的图，复用视觉特征增加一个 HINT 分支。完整掩码按 1:1 概率
  选正确点或掩码外错误点；缺失图使用随机错误点。错误点避开实例边缘一格。
  弱框和稀疏点没有错误点标签，直接跳过该辅助分支。
- 两个分支始终监督同一真实 OPOS；HINT 分支额外监督质量 BCE。辅助损失为
  NULL 点损失 + 0.5 × HINT 点损失 + 0.1 × HINT 质量损失，外乘 grounding_coef=0.5。
  每张图缓存重置，不监督导航动作、STOP 或到达状态，不把标注输入推理。
- 每个 mini-batch 两次 4 图梯度累积，共八卡每 update 128 张公共图；中间不更新
  参数，辅助梯度取平均。比原先 64 张覆盖更多样本，显存按 4 图控制。
- 公共来源 COCO / referring / ScanNet / ImageNet / PixMo 比例为
  20 / 35 / 20 / 15 / 10%，明确缺失样本占 10%。全部来自既有训练清单，
  导航仍仅用训练场景，未加入验证 episode 或验证语义资产。
- 导航 real / simulated / NULL episode 比例从 50 / 25 / 25% 改为
  40 / 20 / 40%；保留原 DAgger、PPO、点 CoT、未来动作和到达监督。
  LoRA 学习率从 5e-6 降至 2.5e-6，动作、记忆等主要导航组减半，点头仍为 5e-5。
- 导航网格 climb / cell_height 同时改为 0.20 / 0.20 m，与已完成的全量评估一致，
  相机、机身和成功距离不变。此项仅允许显式 recipe fork；精确 resume 不可修改。

模型参数结构和推理计算不变；质量头尚不作为硬门控。新辅助任务旨在改善目标确认，
并不能预先保证 SR 上升，也没有补足跨楼层探索教师的能力。
每 25 update 保存、每 200 update 在三个 split 各 32 例测 no_hint、simulated_hint、
YOLOE，每 1000 update 扩为各 100 例，继续保存明亮的 APOS / OPOS / HINT 视频。
主 best 仍按 simulated_hint，另保留 best_real_hint。应结合数百至上千 update 趋势及
同一 300 例对照判断，不因一次小样本评估波动停训或更换模型。

```bash
make train
python tools/training_status.py --root runs/finetune_15300_grounded_hints
bash scripts/stop_training.sh runs/finetune_15300_grounded_hints
```
