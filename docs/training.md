# 训练与恢复（revision 5）

## 2026-10-08 代码审查与诊断实验

revision 5 已在 update 76 保存退出，固定 update 50 节点继续保留。revision 6 八卡实验使用每步目标查询和较低主干 LR，第 5、10、25 次自主验证均为 0/144，随后在 update 30 保存退出。此前实验的日志和检查点留在原目录；下面的上游对照配方仍描述 revision 5。

当前候选使用独立配置 `model=qwen35_0p8b_kda_stable trainer=ovsegdt_kda_stable`（revision 8），从转换初始化重新训练。它保留 RGB＋文本、六动作、教师、奖励、beta=0、EALM/PPO/IL/value 公式、冻结视觉和 T=100，仅调整模型与预训练优化器的适配：

- 每步将目标 token embeddings 独立取均值，加到 NAV query；不增加递归状态或每帧 token 数。在 detach 的序列起点仍有当前目标 embedding 的动作损失梯度。
- 六个转换 KDA 层在 sigmoid 输出门之前补上每个 value head 的 RMS normalization（eps=1e-5）。这是安装版本 FLA KimiDeltaAttention 使用的归一化位置；本实现不增加 affine 参数，O 投影继续训练。
- 预训练主干 LR=1e-5，actor LR=2.5e-4；线性 critic 权重和 bias 从零开始，LR=2.5e-5。默认 revision 5 的初始化与学习率保持原值，便于旧 checkpoint 回放。

恢复校验新增目标路径、KDA 输出归一化、学习率、权重衰减、梯度裁剪和 IL class weights；拒绝静默更换配方，避免配置中的新 LR 被保存的 optimizer state 覆盖。推理按 checkpoint 保存的模型选项加载；修改运行中的 Python 源码后，需要重启服务才会应用新实现。

证据位于 `runtime/reports/policy_inputs_20261008/`，汇总为 `code_audit.json`。同一真实教师轨迹的 32 帧输入对照中，revision 5 update 75 对换目标和遮黑图像的最后一步概率 TV 均为 0；初始化仍有敏感性。revision 6 的短对照保留了部分图像与目标敏感性，但八卡自主 SR 未改善，不能把输入变化当作成功率。

局部同输入比较发现，旧 KDA 分支输出 RMS 仅为复制 Q/K/V/O 的 causal softmax 分支的约 0.8%–3.7%。比较没有 RoPE 或历史 KV，只用于检查分支尺度，不是完整 Qwen 等价测试。补齐输出归一化后的双卡试运行仍在第 3 次更新出现 value loss=67.38、裁剪前 grad norm=3455，因此又单独检查零初始化 / 较低 LR 的 critic。零初始化 / 较低 LR 版本第 3 次更新的 value loss=0.00019、裁剪前 grad norm=14.38，第 4 次分别为 0.00011 / 13.14。该初始化与学习率联合对照只能作为数值稳定性证据，不能拆解成单一因素的因果结论。曲线见 `value_stability.png`。

每个双卡短对照执行 4 次完整 T=100 更新，并逐次检查所有参数的跨 rank SHA 和更新前概率；四次更新的全部参数 SHA 均跨 rank 一致，更新前概率误差均为 0；它们不是导航 benchmark。稳定版本 update 4 的同轨迹最后一步换目标 TV≈0.0154 / 0.0093、遮黑 TV≈0.0509，仍不是 SR。随后八卡 revision 8 在 update 5、10、25、50 完成三个 split 各 48 条原始起点、最多 500 步的自主 argmax 验证，SR 均为 0/144；按原四轮全零规则在 update 50 保存停训。

revision 8 的首次 update 5 验证 oracle_success=0，动作几乎全部 TURN_RIGHT。八卡 value loss 保持约 1e-4，更新前概率误差为 0；同轨迹输入对照最后一步换目标 TV≈0.0049 / 0.0082、遮黑 TV≈0.0334。到 update 50，有 7/144 条验证轨迹进入成功距离（seen 3、synonyms 1、unseen 3），但没有主动 STOP，SR 仍为零。50 次更新的 STOP 标签约占 0.394%，最后 25 次约占 0.389%，近期 STOP recall=0；EALM alpha 仍全部为 1。数值与输入路径改善尚未转化为自主导航成功。

当前代码验证为单元/回归 63 项、GPU 6 项、真实 Habitat/训练集成 13 项通过（分批运行）；Ruff（包含 services）、format、mypy 与 diff whitespace 检查通过。首次合并集成检查仅暴露一张 GPU，两项旧测试固定访问 GPU 1 导致超时；暴露两张卡后这两项通过。课程回退新增步数耗尽、过早 STOP 两个真实仿真用例后，全部 7 项 Habitat 用例通过。网页实际重置和单步检查证据仍为 `runtime/reports/policy_inputs_20261008/viewer_check.json`。

## 监控后续与缓存优化

update 50 固定于 `checkpoints/revision8_stop_audit_20261008`，供本次试验只读使用。新并行评估入口初始化 DDP，重新检查同一 144 条轨迹：seen 的 508 次目标附近决策平均 STOP 概率约 0.362%，最大约 0.565%；synonyms 12 次平均约 0.337%；unseen 490 次平均约 0.490%。距离只用于记录，动作仍为策略 argmax。到达与停车两项能力均不足，不能仅靠修改成功判定解决。

从该锚点重新初始化优化器、sampler 与 EALM，运行以下独立双卡 T=100 诊断；不是八卡状态的连续恢复，也不是预算匹配的能力比较：

| 试验 | 更新 / transitions | STOP 标签 | 自主 SR | 到达过目标距离 | 处理 |
|---|---|---|---|---|---|
| 原配方＋视觉缓存 | 8 / 6,400 | 7 | 0/144 | 6/144 | 只采用等价缓存 |
| STOP IL 权重 8＋缓存 | 8 / 6,400 | 7 | 0/144 | 11/144 | 未切入主训练 |
| 50% 近目标训练 warmup＋缓存 | 4 / 3,200 | 16 | 0/144 | 15/144 | 未切入主训练 |

STOP 权重试验提高了目标附近 STOP 概率，但仍没有实际停车，碰撞率明显上升；课程试验同样无自主成功。课程使用的原有 warmup 在 250 步耗尽后会静默保留中间位置，因此不能把该试验解释为所有选中样本成功完成近目标 reset。现在步数耗尽、过早 STOP 或教师异常均恢复原始合法起点并记录 `curriculum_fallback`；主训练仍关闭课程。独立配置 `ovsegdt_kda_stop_balanced`（revision 9）、`ovsegdt_kda_stop_curriculum`（revision 10）仅保留供对照。

缓存优化只保存冻结视觉编码器的 BF16 输出；目标、图像边界和 NAV embeddings 每次重新读取当前可训练参数。解冻视觉时拒绝缓存。相同初始化 / 配方 / seed 的缓存与重算控制实验连续 8 次更新后，全模型参数 SHA 逐次完全一致。排除首次启动的 update 2–7 优化时间平均从 50.17 s 降至 47.06 s，观测下降约 6.20%；共享机器负载未控制，不能视为固定吞吐保证。每 rank 增加 110,592,000 bytes CPU 缓存。目标 token IDs 的缓存还减少了重复 CPU tokenizer 调用，但不缓存 embeddings，不改变梯度。

新增全局标签计数、动作混淆矩阵和条件 STOP 概率；常数动作先验熵改用全局标签分布，避免平均各 rank 的先验熵低估它。第一轮对照在 update 8 遇到新增诊断的空类均值聚合错误，导致日志阶段退出；已修复、增加回归用例并从同一锚点重新运行，未将失败轮的残缺指标当成成功实验。监控和停训脚本识别双卡 checked trainer，恢复启动阶段等候当前 launcher 注册 worker，避免旧监控配置误杀新进程；启动停滞仍有超时监督。失败记录、完整评估与哈希见 [本次审计](../runtime/reports/stop_audit_20261008/audit.json)。

活动八卡已从 update 50 恢复 Adam、scheduler、各 rank EALM / RNG / sampler；仿真 episode 重新开始。`ovsegdt_kda_cached` 仍为 revision 8，核心配方保持原值。活动目录仍为 `runs/streamnav_active/ealm` → `runs/streamnav_kda_stable_20261008/ealm`，上限为 200 updates（累计 640,000 transitions），每 5 次保存，在 55、75、100、150、200 次各运行三个 split 共 144 条自主验证。30 秒后台监控检查 worker、非有限值和停滞，update≥200 且连续五轮全零则请求保存退出；训练循环同时受 200 次上限约束，矩阵保活关闭。实际进程与新结果查看活动日志。200 次仅占上游 10 亿步预算的 0.064%，不是完整训练或预算匹配的能力结论。

续训 update 51–55 的更新前概率误差均为 0，value loss 约 9.7e-5–2.1e-4，EALM alpha 仍为 1，贪心 STOP 计数仍为 0。update 55 自主验证为 0/144，oracle_success 合计 13/144（seen 7、synonyms 4、unseen 2）；目标附近平均 STOP 概率分别约 0.315% / 0.306% / 0.155%，错误 STOP 为 0。与 update 50 的 7/144 相比，到达数量有所变化，但自主 SR 没有提升，不能将等价缓存当成能力改善的原因。网页已加载 update 55，真实重置与模型单步通过，记录为 `runtime/reports/stop_audit_20261008/viewer_check.json`。

有三项训练边界需要保留在效果解释中：

- 当前 HM3D-OVON 是短目标指令的 ObjectNav，不包含 R2R 一类长路径语言指令。
- 上游 `no_segm_loss` 只关闭 semantic loss，实际仍开启 GT mask 输入；当前 RGB＋文本限制省去了该输入，以及 GPS/compass、上一动作。
- 六层 softmax→KDA 转换复制 Q/K/V/O 并随机初始化新 gate，没有蒸馏，不能视为完整保留原 Qwen 多模态能力。

revision 5 的全部 76 次更新中 `ealm_alpha=1`：熵 EMA 最终约 1.054，超过上游 0.75 阈值，因此 PPO **策略**损失系数一直为 0（value 与 entropy 损失仍存在）。这是上游门控的实际行为，不是混合公式写反。策略仅拟合动作频率、IL 未优于动作先验时，EALM 也无法进入 RL 阶段；去掉 mask 后的条件动作熵是否适合原阈值仍待验证。当前诊断保留该公式，不声称已经解决此问题。

用户于 2026-10-08 确认：保留 RGB＋文本输入限制，恢复上游六动作，其他训练逻辑严格对照本地 OVSegDT。当前入口与使用方法见 [README](../README.md)，源码节点、差异和实测见 [当前版本](current_version.md)。此前 revision 3/4 的混合数据、教师接管、STOP 加权和课程均不属于当前配方。

导航网格同时使用上游 max_climb=0.10 m、cell_height=0.05 m，以及用户机体高 0.88 m / 半径 0.18 m；缓存签名和恢复校验包含全部四项，不复用旧网格。

## 数据与教师

训练只使用 HM3D-OVON train。按上游 VER worker 分配规则，将打乱的场景轮流分配给每卡 4 个环境，每个环境至少分配 16 场景（小数据集通过重复轮次覆盖，文件不重复加载）。episode 读取器采用惰性加载，但保持“全局打乱、按场景分组”相同的抽样分布：场景顺序按 episode 数量生成加权无放回排列，场景内部独立打乱、无放回读取。每场景连续 50,000 步 ±20%，在 episode 边界轮换；不是旧版每 16 个 episode 随机重选场景。随机流实现不同，不承诺与上游逐条相同顺序。

教师直接运行 pinned `frontier_exploration` 的 ObjNavExplorer。其地图、目标视点与导航信息仅用于生成监督标签；策略输入仍只有 RGB＋文本。不会用“离目标没有变近”判定探索教师失败。基础设施或教师异常单独记录，不伪造标签；若教师耗尽 frontier 或导航接口失败，当前有效前缀做显式截断并重置，连续三次失败则终止训练。这是异常处理路径，不能冒充正常终止或成功数据。

## 损失和采样

所有环境动作都从策略 π 采样，beta=0，教师只提供当前状态的 IL 标签。PPO 比率为 π_new(a)/π_old(a)，六动作 IL 等权，没有额外 STOP 权重。固定混合和教师接管代码仅供旧实验复现，默认不启用。

EALM 在每个 minibatch 使用**上一 minibatch 的平均熵 EMA**形成共同权重；初始 IL alpha=1，EMA decay=0.95，低/高阈值为 0.35/0.75，线性插值并截断到 [0,1]。一次优化完成后更新 EMA，供下一批使用。各 rank 的 EMA 随检查点保存和恢复。loss = alpha×IL + (1-alpha)×PPO + 0.5×value - 0.01×entropy。

gamma=0.99、GAE lambda=0.95，advantage 不归一化。value loss 与上游相同：先按 old value ±0.2 截断越界预测并屏蔽其梯度，再计算 0.5×MSE；不是常见 PPO 的 max(两个平方误差) 版本。正常 episode 结束和 500 步上限都切断 bootstrap，与上游 masks 处理一致。教师异常的有效前缀另行显式截断，不与普通 time limit 混为一谈。

奖励只有成功 +5、slack -0.001、碰撞 -0.003，没有进度奖励或 false STOP penalty。成功定义为主动 STOP 且距离 <0.25 m，额外记录严格 0.1 m 成功和轨迹到达过目标邻域的 oracle_success。

## 序列、优化器与八卡

每 rank 4 环境 ×100 步；两个 minibatch，每批 2 条完整 100 步序列，1 epoch。固定大小递归状态是 KDA backbone 的结构差异；训练序列内保留梯度，序列起点 detach，episode 边界重新预填目标。采样和重放都使用 chunk recurrence；BF16 的推理合批宽度也匹配训练批宽，避免形状相关的 kernel 舍入误差。

Adam eps=1e-5，无 weight decay；全局梯度范数裁剪 0.2。视觉编码器始终冻结。按实际 PIRLNav scheduler，第 1 次 update 只训练线性 critic，actor / state encoder 的 LR=0；第 2 次起解冻，LR=2.5e-4，保持常数。不能因为 YAML 名称含 `use_linear_lr_decay` 就套用线性衰减。解冻后重新建立 DDP reducer，使新参数参与跨 rank 同步。

8 卡为一个实验，每 update 3,200 新 transitions；312,500 updates 对应 10 亿步。较少 GPU 的诊断若要跑同样总步数，需要同步调整 num_updates；短 canary 会显式覆盖更新次数。BF16 autocast、FP32 master 参数、activation checkpointing 仅控制数值与显存，不改变 BPTT 长度或 objective。

本实现使用同步 rollout/replay；上游 VER 还支持可变经验长度和异步 worker 调度。这里每个环境贡献同样多的步数，不产生由 VER 异步经验分配导致的非均匀权重。它不是逐行复刻上游运行时，也不声称训练轨迹或随机数完全一致。

## 保存、恢复与进程监督

每 25 updates 保存，首次 update 也保存；第 1、25、50、100、200… updates 运行三个 OVON split 各 48 条诊断验证。检查点先写临时目录，完成后原子发布，再更新 latest。保留最近三份和最佳验证模型，best 按宏平均 SR、再按 SPL 选择。

恢复包括 Adam、LR scheduler、每 rank entropy EMA、RNG、场景/episode sampler。数据清单、机器人/teacher/reward、模型 head、训练 revision、DDP world size 和核心训练配方变化会拒绝恢复。仿真器物理状态与正在执行的 episode 不保存，恢复后从新 episode 开始；不承诺逐动作续接。

SIGTERM / SIGINT 请求在当前 update 后保存退出。后台监控每 30 秒检查真实 worker 身份、loss、更新和验证；非有限值、worker 丢失、3,600 秒不推进或 update≥500 后长期全零/坍缩会请求停训，不无限重启。具体触发原因写入 health_status.json / monitor.log。

关注这些指标：

- `preupdate_replay_log_prob_error_max`：首个 optimizer step 前概率误差，任一 rank >0.05 则拒绝更新；不等同于优化后的 PPO 变化。
- `vision_grad_norm=0`：当前冻结视觉是预期行为。首次 update 的 actor/kda 梯度也应为 0，之后应有有效梯度。
- `entropy_ema` / `ealm_alpha`：上游熵调度的实际状态。
- `expert_action_histogram` / `oracle_class_recall`：监督分布和各类拟合情况。
- `oracle_skipped_episodes`：教师异常，不能当作成功或静默删除。
- `eval_metrics.jsonl`：自主 argmax 导航；训练 `success` 来自随机采样策略，不能替代正式验证。
- `global_transitions`、`update_seconds`、GPU 显存：衡量实际训练效率。矩阵保活事件独立记录，不计作训练。

验证按固定全局 episode 顺序跨 rank 分片，按 episode 数汇总指标，保留成功和失败的逐条记录。视频位于 evaluation/update_*，网页入口见 README。历史 revision 3/4 和诊断实验的配置、指标、代码快照已合并进 [历史归档](../runtime/baselines/revision5_20261008/legacy_records.tar.gz)，旧权重和视频已清理；历史曲线不拼入新实验。

## 当前节点与日常保留

`runs/streamnav_active/ealm` 指向当前实验；训练、健康状态和最新检查点从这里查看。停止与恢复命令见 README。默认每次保留最近三份检查点及 best，活动日志和验证视频保留在当前实验中。

`checkpoints/revision5_20261008` 固定保留 update 50 的完整模型、优化器、各 rank 状态、配置与训练源码。它通过已发布检查点的硬链接保存，不受活动实验轮转影响。只读使用，禁止原地修改其中的文件；修改会影响同 inode 的文件。它不是额外复制的一份权重，也不是已达到导航效果目标的 best 模型。

日常继续训练使用活动目录的 latest，并保持日志、revision 与 world size 一致。固定节点用于检查和回放；若要从固定节点创建训练分支，应在新目录中使用匹配配置与八卡 world size，不能把现有较新实验的日志回退拼接。恢复总会重新开始仿真 episode。节点信息与源码快照见 [当前版本](current_version.md)。
