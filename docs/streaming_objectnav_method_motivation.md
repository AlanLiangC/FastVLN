# Streaming ObjectNav：Method Motivation 与方法设计

> 目标：构建一个用于 **快速 Object Navigation 的 Streaming VLN / VLM Policy**。  
> 模型不承担完整的长期任务规划，而作为外部 Agent 下方的高速导航策略模块，重点优化：
>
> - 当前观测到动作的低延迟；
> - 长 episode 下稳定的流式记忆；
> - 端到端训练；
> - HM3D-v1、HM3D-v2、HM3D-OVON 的统一训练与评测；
> - 简单、直接、可部署的离散动作输出。

---

# 1. Core Motivation

现有 VLM navigation 方法经常通过显式拼接历史帧或历史视觉 token 来获得时序信息：

```text
current observation
+ historical observations
+ instruction
        │
        ▼
      VLM
        │
        ▼
      action
```

这种方式的问题是：

1. history context 会随着 episode 增长；
2. 每一步都需要重新处理一定数量的历史 token；
3. inference latency 与 memory footprint 不够稳定；
4. 对一个由外部 Agent 控制的底层导航模块而言，模型承担了不必要的长期上下文管理。

本项目希望得到一个更接近机器人“System-1”的导航策略：

```text
Current RGB
+ Navigation Instruction
+ Persistent Internal State
        │
        ▼
   Fast Navigation Policy
        │
        ▼
Discrete ObjectNav Action
```

长期记忆、目标切换、任务分解、显式地图或 scene graph 可以由上层 Agent 负责；底层模型只负责高频、连续、带短中期记忆的导航决策。

---

# 2. Streaming Formulation

给定导航目标：

\[
g = \text{instruction}
\]

时刻 \(t\) 的输入只有当前 RGB observation：

\[
I_t
\]

模型内部维护 recurrent state：

\[
S_t
\]

策略形式为：

\[
a_t, V_t, S_t =
\pi_\theta(I_t, g, S_{t-1})
\]

其中：

- \(a_t\)：离散导航动作；
- \(V_t\)：critic value；
- \(S_t\)：更新后的 KDA recurrent state。

不再显式输入：

\[
I_0,I_1,\ldots,I_{t-1}
\]

历史信息全部通过 KDA state 传播。

---

# 3. Backbone

项目只保留一个 backbone：

```text
Qwen3.5-0.8B
        │
        ▼
KDA-converted Qwen3.5-0.8B
```

不再提供：

- LightNav-0 initialization；
- Qwen3.5-4B；
- 多 backbone registry；
- RVQ action decoder。

Qwen3.5-0.8B 本身已经包含 Gated DeltaNet 与 attention 的 hybrid sequence mixer，因此 KDA conversion 的默认实现只替换剩余 full-attention / gated-attention 部分，并保留原有 recurrent-friendly blocks。

最终模型统一记为：

```text
KDA-Qwen3.5-0.8B
```

---

# 4. KDA Conversion 的定位

KDA conversion 是 **backbone 构建步骤**，不是 ObjectNav 的单独训练阶段。

工程上提供：

```bash
python -m streamnav.tools.convert_kda \
    --model Qwen/Qwen3.5-0.8B \
    --output checkpoints/qwen35_0p8b_kda
```

得到：

```text
Qwen3.5-0.8B
    +
KDA replacement
    +
compatible recurrent cache
```

之后 ObjectNav 训练从这个 backbone 直接开始。

本项目的研究重点不是再设计一个复杂的 KDA pretraining pipeline，而是研究：

> KDA-based recurrent VLM 是否可以作为低延迟、长时运行的 Streaming ObjectNav policy。

---

# 5. Input Design

每个 episode 开始时：

```text
System Prompt
+
User Instruction
```

只做一次 prefill。

例如：

```text
System:
You are a fast object navigation policy.
Navigate to the requested object.

User:
Find a chair.
```

生成初始 recurrent state：

\[
S_0 = F(P_{system}, P_{user})
\]

之后每一步仅输入：

```text
Current RGB observation
+
previous KDA state
```

即：

\[
S_t =
F(I_t,S_{t-1})
\]

因此部署时不会重新拼接：

```text
instruction
history images
old action text
old model outputs
```

---

# 6. Visual Streaming

完整数据流：

```text
             Episode Start
                  │
       System + User Instruction
                  │
                  ▼
           Prompt Prefill
                  │
                  ▼
                 S0
                  │
──────────────────┼─────────────────────────
                  │
Current RGB It    │
      │           │
      ▼           │
Qwen Vision Encoder
      │
      ▼
Current Visual Tokens
      │
      ▼
KDA-Qwen3.5-0.8B  ◄──── St-1
      │
      ├───────────────► St
      │
      ├───────────────► Policy Hidden
      │
      └───────────────► Value Hidden
                           │
               ┌───────────┴───────────┐
               ▼                       ▼
          Action Head              Value Head
               │                       │
               ▼                       ▼
         Categorical π              V(s_t)
```

---

# 7. Action Space：直接采用 OVSegDT / Habitat ObjectNav 风格

删除之前的 Fast Action Grammar、RVQ code 和 autoregressive action token。

策略直接输出离散 action logits：

```text
STOP
MOVE_FORWARD
TURN_LEFT
TURN_RIGHT
```

记：

\[
\pi_\theta(a_t|I_t,g,S_{t-1})
=
Softmax(W_a h_t)
\]

其中：

\[
a_t \in
\{
STOP,
MOVE\_FORWARD,
TURN\_LEFT,
TURN\_RIGHT
\}
\]

训练：

```text
Categorical(logits)
```

评测：

```text
argmax(logits)
```

RL rollout：

```text
sample from Categorical(logits)
```

这种设计相比 autoregressive token generation 更适合本项目，因为：

- 只需要一次 policy head forward；
- 无 autoregressive decoding；
- 没有额外 tokenizer / grammar；
- PPO log-prob 直接定义；
- 与 Habitat ObjectNav action space 和 OVSegDT 的 actor-critic 训练方式兼容；
- 最小化 observation-to-action latency。

---

# 8. Policy Head

从 backbone 当前时刻的 policy representation：

\[
h_t
\]

直接接：

```text
Linear(hidden_dim, 4)
```

得到：

\[
z_t \in \mathbb{R}^{4}
\]

然后：

\[
\pi_t = Softmax(z_t)
\]

无需 LM vocabulary decoding。

因此 Qwen 的语言建模 head 在导航训练中可以保留用于 checkpoint compatibility，但 navigation inference 不经过完整词表 softmax。

---

# 9. Value Head

使用相同 backbone hidden：

```text
h_t
 │
 ▼
LayerNorm
 │
 ▼
MLP
 │
 ▼
scalar value
```

\[
V_\phi(s_t)
\]

critic 只增加很小的计算量，并允许整个模型像 OVSegDT 一样直接进行 PPO-style end-to-end optimization。

---

# 10. End-to-End Training

训练不再拆分成：

```text
BC
→ DAgger
→ RL fine-tuning
```

而是在同一个训练流程中进行：

```text
Environment Rollout
        │
        ├── Policy Action
        ├── Expert / Oracle Action
        ├── Reward
        └── Value
        │
        ▼
Unified Rollout Buffer
        │
        ▼
Imitation + PPO + Value + Entropy
        │
        ▼
KDA-Qwen3.5-0.8B
```

即：

> learner 一边与 Habitat 环境交互，一边获取 shortest-path/oracle supervision，同时利用环境 reward 进行 PPO 更新。

---

# 11. DAgger-style On-policy Supervision

对于每个 rollout state：

\[
s_t
\]

记录：

```text
policy action
expert action
action log probability
value
reward
done
entropy
```

expert action 来自 Habitat shortest-path follower / ObjectNav oracle。

模型执行的动作可以按 DAgger mixing：

\[
a_t^{env}
=
\begin{cases}
a_t^{expert}, & p < \beta\\
a_t^{policy}, & otherwise
\end{cases}
\]

其中 \(\beta\) 由配置控制并随训练降低。

即使环境最终执行 policy action，也仍可以记录 expert label：

\[
a_t^*
\]

用于 imitation objective。

因此不需要提前离线收集完整 BC dataset。

---

# 12. Imitation Objective

策略 action logits：

\[
z_t
\]

expert action：

\[
a_t^*
\]

直接使用 categorical cross entropy：

\[
L_{IL}
=
-\log
\pi_\theta(a_t^*|s_t)
\]

这里没有：

- segmentation loss；
- trajectory reconstruction loss；
- RVQ loss；
- language-generation loss。

ObjectNav 主监督就是：

```text
expert discrete action
```

---

# 13. PPO Objective

policy rollout 得到：

```text
reward
value
old_log_prob
done
```

使用 GAE：

\[
\hat A_t
\]

PPO ratio：

\[
r_t(\theta)
=
\exp[
\log\pi_\theta(a_t|s_t)
-
\log\pi_{\theta_{old}}(a_t|s_t)
]
\]

clipped PPO objective：

\[
L_{PPO}
=
-
\min
(
r_t\hat A_t,
clip(r_t,1-\epsilon,1+\epsilon)\hat A_t
)
\]

value loss：

\[
L_V =
(V_t-R_t)^2
\]

entropy：

\[
H_t =
-\sum_a \pi_t(a)\log\pi_t(a)
\]

---

# 14. OVSegDT-style Adaptive IL/RL Mixing

本项目采用 OVSegDT 的核心训练思想：

```text
DAgger-style expert supervision
+
PPO reinforcement learning
+
entropy-adaptive mixing
```

但完全移除 segmentation branch。

统一 policy objective：

\[
L_{policy}
=
\alpha_t L_{IL}
+
(1-\alpha_t)L_{PPO}
\]

其中 \(\alpha_t\) 由 policy uncertainty / entropy 驱动。

直觉：

```text
high uncertainty
    -> rely more on oracle imitation

low uncertainty
    -> allow stronger RL optimization
```

最终：

\[
L =
L_{policy}
+
\lambda_V L_V
-
\lambda_H H
\]

这套 loss 从训练开始到结束保持统一，不再做多个训练阶段切换。

---

# 15. Reward

保持 ObjectNav reward 简单，避免手工 reward engineering 盖过主方法。

推荐：

\[
R_t =
R_{progress}
+
R_{success}
-
R_{slack}
\]

其中：

### Progress

\[
R_{progress}
=
d_{t-1}^{geo} - d_t^{geo}
\]

### Success

成功 STOP：

```text
positive terminal reward
```

### Slack

每步小惩罚：

```text
-step_penalty
```

可选加入：

```text
collision penalty
false-stop penalty
```

但首版尽量保持和 OVSegDT/Habitat ObjectNav reward 体系一致。

---

# 16. Streaming State 在训练中的传播

这是本项目与普通 OVSegDT 最大的区别。

rollout 时：

```text
episode reset
    │
    ▼
prompt prefill -> S0
    │
    ▼
obs0 -> policy -> S1
    │
    ▼
obs1 -> policy -> S2
    │
    ▼
obs2 -> policy -> S3
    │
    ...
```

不能在每个 environment step：

```text
重新初始化 KDA state
```

否则模型等价于单帧 policy。

---

# 17. Training Unroll

为了控制显存，不需要整个 500-step episode 都做完整反向传播。

rollout 仍然可以完整保持 recurrent state，但 optimization 使用固定长度 sequence chunk：

```text
rollout:
    persistent KDA state for whole episode

update:
    B environments
    × T recurrent steps
```

例如：

```yaml
trainer:
  num_envs: 32
  rollout_steps: 32
```

每次收集：

\[
B \times T
\]

个 transitions，然后直接联合优化 IL + PPO。

这和 PPO 的 rollout/update pattern 本身一致，因此没有必要额外设计独立的 BC stage。

---

# 18. Cache Detach

在 rollout 收集期间：

```text
inference mode
state recurrently updates
```

优化时可以重新 replay 当前 rollout chunk，并在 chunk boundary：

```python
state = detach_state(state)
```

避免计算图跨越无限 episode。

但 deployment state 本身仍持续存在。

---

# 19. Dataset Coverage

统一支持：

```text
HM3D-v1
HM3D-v2
HM3D-OVON
```

训练器不区分三个独立 trainer，只通过 dataset sampler 混合 episode。

建议：

```yaml
dataset_mix:
  hm3d_v1: 0.25
  hm3d_v2: 0.25
  hm3d_ovon: 0.50
```

原因是 HM3D-OVON 对 open-vocabulary/generalization 更重要，因此给予更高采样权重。

比例全部由 config 控制。

---

# 20. Unified Episode Representation

```text
NavigationEpisode
├── dataset_id
├── split
├── scene_id
├── episode_id
├── goal_text
├── start_position
├── start_rotation
└── goal metadata
```

environment step 返回：

```text
RGB
reward info
done
oracle action
metrics
```

训练模型实际使用：

```text
RGB
goal instruction
KDA state
```

GT pose、geodesic distance、semantic annotations只允许用于：

```text
oracle
reward
evaluation
```

不能进入 policy inference input。

---

# 21. No Segmentation Task

明确删除：

```text
semantic segmentation branch
mask prediction
mask supervision
YOLO/semantic-mask policy input
segmentation auxiliary loss
```

模型必须直接从 RGB 与语言目标完成：

```text
visual grounding
+
navigation decision
```

因此本项目可以看作：

> 将 OVSegDT 的 DAgger + PPO + adaptive IL/RL training framework，迁移到一个 streaming KDA-VLM policy 上，并去掉 segmentation dependency。

---

# 22. Fast Response Design

本项目的低延迟来自四个地方：

### 22.1 无显式历史图像

只有：

```text
current RGB
```

### 22.2 KDA recurrent state

历史压缩在：

```text
fixed-size recurrent state
```

### 22.3 无语言解码

不调用：

```text
model.generate()
```

### 22.4 四分类 action head

只需：

```text
hidden -> Linear -> 4 logits
```

最终每个 control step 的主要开销就是：

```text
vision encode
+
one recurrent backbone update
+
tiny actor/critic heads
```

---

# 23. External Agent Interface

整个系统定位：

```text
High-level Agent
│
├── task reasoning
├── long-term memory
├── target switching
├── exploration strategy
└── recovery / replanning
        │
        ▼
Object instruction
        │
        ▼
Streaming ObjectNav Policy
│
├── Current RGB
├── KDA State
├── Actor Head
└── Value Head
        │
        ▼
STOP / FORWARD / LEFT / RIGHT
        │
        ▼
Robot / Habitat
```

Agent 不需要看到 KDA cache。

---

# 24. Evaluation

Navigation metrics：

```text
Success Rate
SPL
SoftSPL
Distance-to-goal
Collision Rate
Episode Length
```

Streaming/system metrics：

```text
P50 observation-to-action latency
P95 latency
P99 latency
decision frequency
peak VRAM
KDA state memory
memory vs episode length
```

核心实验应该证明：

\[
Latency_t
\]

不会因为 episode 历史越来越长而持续增长。

---

# 25. Main Ablations

只保留与论文主问题相关的 ablation。

### Memory

```text
no recurrent state
vs
KDA streaming state
```

### Training

```text
PPO only
DAgger only
fixed IL + PPO
entropy-adaptive IL + PPO
```

### Input history

```text
current frame only + KDA
vs
explicit history baseline
```

### Dataset

```text
HM3D
vs
HM3D + HM3D-OVON
```

---

# 26. Method Summary

最终方法可以压缩成一句话：

> We convert Qwen3.5-0.8B into a recurrent KDA-based multimodal policy, cache the navigation history entirely inside its streaming recurrent state, and train the resulting ObjectNav actor-critic end-to-end with OVSegDT-style DAgger/PPO adaptive supervision, while directly predicting Habitat discrete navigation actions from the current RGB observation.

结构：

```text
Instruction
     │
     └──────► one-time prefill
                   │
                   ▼
                  S0

RGB_t ─► Vision Encoder ─► KDA-Qwen3.5-0.8B ◄── S_{t-1}
                               │
                      ┌────────┴────────┐
                      ▼                 ▼
                  Actor Head        Value Head
                      │
                      ▼
          STOP / FORWARD / LEFT / RIGHT
                      │
                      ▼
                  Habitat
                      │
             reward + oracle action
                      │
                      ▼
        EALM-style IL + PPO objective
```

---

# 27. References

- LightNav-0 — architecture inspiration only  
  https://github.com/lightorigins/LightNav-0

- Flash Linear Attention / KDA  
  https://github.com/fla-org/flash-linear-attention

- qingyi-kda  
  https://github.com/Sisyphbaous-DT-Project/qingyi-kda

- OVSegDT  
  https://github.com/CognitiveAISystems/OVSegDT

- Qwen3.5-0.8B  
  https://huggingface.co/Qwen/Qwen3.5-0.8B

- Habitat-Lab  
  https://github.com/facebookresearch/habitat-lab

- HM3D-OVON  
  https://huggingface.co/datasets/nyokoyama/hm3d_ovon
