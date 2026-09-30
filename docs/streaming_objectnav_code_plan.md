# Streaming ObjectNav：Enterprise Code Plan

> 唯一模型配置：
>
> ```text
> KDA-converted Qwen3.5-0.8B
> ```
>
> 唯一训练范式：
>
> ```text
> online rollout
> + DAgger-style oracle supervision
> + PPO
> + entropy-adaptive IL/RL mixing
> ```
>
> 不再包含：
>
> - LightNav checkpoint initialization；
> - Qwen3.5-4B；
> - RVQ / trajectory tokenizer；
> - autoregressive Fast Action Grammar；
> - segmentation task；
> - BC → DAgger → PPO 多阶段训练。

---

# 1. Engineering Goals

必须满足：

```text
1. 单一 backbone，减少实验分支
2. Habitat 与模型解耦
3. Streaming state 显式管理
4. Actor/Critic 与 backbone 模块化
5. IL 与 PPO 在同一 trainer 中联合优化
6. HM3D-v1/v2/OVON 统一 dataset interface
7. 所有行为 config-driven
8. 能做并行 rollout
9. 能做 GPU latency regression
10. 能直接作为上层 Agent 的低延迟服务
```

---

# 2. Repository Layout

推荐：

```text
streaming-objectnav/
│
├── README.md
├── pyproject.toml
├── uv.lock
├── Makefile
├── .pre-commit-config.yaml
│
├── configs/
│   ├── model/
│   │   └── qwen35_0p8b_kda.yaml
│   ├── data/
│   │   ├── hm3d_v1.yaml
│   │   ├── hm3d_v2.yaml
│   │   ├── hm3d_ovon.yaml
│   │   └── mixed_hm3d.yaml
│   ├── trainer/
│   │   └── ovsegdt_e2e.yaml
│   ├── eval/
│   │   ├── hm3d_v1.yaml
│   │   ├── hm3d_v2.yaml
│   │   └── hm3d_ovon.yaml
│   └── serving/
│       └── policy_server.yaml
│
├── src/
│   └── streamnav/
│       ├── contracts/
│       │   ├── observation.py
│       │   ├── action.py
│       │   ├── episode.py
│       │   └── state.py
│       │
│       ├── models/
│       │   ├── qwen35_kda/
│       │   │   ├── backbone.py
│       │   │   ├── conversion.py
│       │   │   ├── kda_adapter.py
│       │   │   ├── cache.py
│       │   │   └── verification.py
│       │   │
│       │   ├── policy/
│       │   │   ├── actor_critic.py
│       │   │   ├── streaming_policy.py
│       │   │   └── action_distribution.py
│       │   │
│       │   └── vision/
│       │       └── preprocessing.py
│       │
│       ├── data/
│       │   ├── schema.py
│       │   ├── mixture.py
│       │   ├── manifest.py
│       │   └── habitat/
│       │       ├── hm3d_v1.py
│       │       ├── hm3d_v2.py
│       │       └── hm3d_ovon.py
│       │
│       ├── envs/
│       │   ├── protocol.py
│       │   ├── habitat_client.py
│       │   └── oracle.py
│       │
│       ├── training/
│       │   ├── trainer.py
│       │   ├── rollout.py
│       │   ├── rollout_buffer.py
│       │   ├── dagger.py
│       │   ├── ppo.py
│       │   ├── ealm.py
│       │   ├── gae.py
│       │   ├── rewards.py
│       │   ├── optimizer.py
│       │   └── checkpoint.py
│       │
│       ├── evaluation/
│       │   ├── runner.py
│       │   ├── navigation_metrics.py
│       │   └── latency_metrics.py
│       │
│       ├── serving/
│       │   ├── session.py
│       │   ├── server.py
│       │   └── batching.py
│       │
│       └── utils/
│           ├── logging.py
│           ├── profiling.py
│           ├── distributed.py
│           └── seed.py
│
├── services/
│   └── habitat_server/
│       ├── server.py
│       ├── env_factory.py
│       └── protocol.py
│
├── tools/
│   ├── convert_qwen35_to_kda.py
│   ├── validate_kda.py
│   └── benchmark_latency.py
│
├── tests/
│   ├── unit/
│   ├── gpu/
│   ├── integration/
│   └── regression/
│
└── docs/
    ├── architecture.md
    ├── training.md
    ├── datasets.md
    └── deployment.md
```

---

# 3. Model Configuration

唯一模型配置：

```yaml
model:
  base_model: Qwen/Qwen3.5-0.8B
  converted_checkpoint: checkpoints/qwen35_0p8b_kda

  dtype: bfloat16
  gradient_checkpointing: true

  action_dim: 4
  action_names:
    - STOP
    - MOVE_FORWARD
    - TURN_LEFT
    - TURN_RIGHT

  value_head:
    hidden_dim: 512
```

不需要：

```text
model registry
multiple backbone adapters
LightNav compatibility layer
action codec registry
```

---

# 4. KDA Conversion Module

转换工具独立于训练器：

```bash
python tools/convert_qwen35_to_kda.py \
    --source Qwen/Qwen3.5-0.8B \
    --output checkpoints/qwen35_0p8b_kda
```

职责：

```text
load Qwen3.5-0.8B
discover attention layers
replace target attention with KDAAdapter
initialize recurrent cache support
save converted model
save architecture manifest
```

训练器只接受已经转换好的 checkpoint。

---

# 5. Conversion 不进入主 Trainer

明确：

```text
KDA conversion != navigation training stage
```

主训练 CLI：

```bash
python -m streamnav.training.trainer \
    trainer=ovsegdt_e2e \
    data=mixed_hm3d \
    model=qwen35_0p8b_kda
```

它不会判断：

```text
是否需要转换
是否需要 BC pretrain
是否需要 RL finetune
```

训练从第一步就是统一 end-to-end objective。

---

# 6. Core Contracts

## Observation

```python
@dataclass(frozen=True)
class Observation:
    rgb: torch.Tensor
    frame_id: int
    timestamp_s: float
```

## NavigationAction

```python
class NavigationAction(IntEnum):
    STOP = 0
    MOVE_FORWARD = 1
    TURN_LEFT = 2
    TURN_RIGHT = 3
```

## StreamingState

```python
@dataclass
class StreamingState:
    kda_cache: Any
    episode_id: str
    instruction_hash: str
    step_index: int
```

## PolicyOutput

```python
@dataclass
class PolicyOutput:
    logits: torch.Tensor
    value: torch.Tensor
    state: StreamingState
```

---

# 7. Policy Architecture

```text
RGB
 │
 ▼
Qwen3.5 Vision Encoder
 │
 ▼
Visual Tokens
 │
 ▼
KDA-Qwen3.5-0.8B  ◄──────── recurrent state
 │
 ├──────────────► updated state
 │
 ▼
Navigation Representation
 │
 ├────────► Actor Head ─────► 4 logits
 │
 └────────► Critic Head ────► scalar value
```

---

# 8. Actor-Critic Module

```python
class NavigationActorCritic(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()

        self.actor = nn.Linear(hidden_dim, 4)

        self.critic = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 512),
            nn.GELU(),
            nn.Linear(512, 1),
        )

    def forward(
        self,
        hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.actor(hidden)
        value = self.critic(hidden).squeeze(-1)
        return logits, value
```

---

# 9. 不使用 LM Autoregressive Generation

禁止：

```python
model.generate(...)
```

导航动作直接来自：

```python
logits, value = actor_critic(hidden)
dist = Categorical(logits=logits)
```

训练 rollout：

```python
action = dist.sample()
log_prob = dist.log_prob(action)
entropy = dist.entropy()
```

评测：

```python
action = logits.argmax(dim=-1)
```

这就是项目中的“Fast Action”实现。

---

# 10. Streaming Policy API

```python
class StreamingObjectNavPolicy(nn.Module):

    def start_episode(
        self,
        episode_id: str,
        instruction: str,
    ) -> StreamingState:
        ...

    def forward_step(
        self,
        rgb: torch.Tensor,
        state: StreamingState,
    ) -> PolicyOutput:
        ...

    def act(
        self,
        rgb: torch.Tensor,
        state: StreamingState,
        deterministic: bool,
    ) -> tuple[NavigationAction, PolicyOutput]:
        ...

    def reset(
        self,
        episode_id: str,
        instruction: str,
    ) -> StreamingState:
        ...
```

---

# 11. Prompt Prefill

`start_episode()`：

```python
def start_episode(...):
    prompt = build_prompt(instruction)
    cache = self.backbone.prefill(prompt)

    return StreamingState(
        kda_cache=cache,
        episode_id=episode_id,
        instruction_hash=sha256(instruction),
        step_index=0,
    )
```

System prompt 和 user instruction：

```text
只 forward 一次
```

后续 observation step 不重复处理 instruction。

---

# 12. Observation Step

```python
def forward_step(rgb, state):
    visual_tokens = vision_encoder(rgb)

    hidden, next_cache = backbone.recurrent_forward(
        visual_tokens,
        cache=state.kda_cache,
    )

    nav_hidden = select_navigation_hidden(hidden)

    logits, value = actor_critic(nav_hidden)

    next_state = state.with_cache(next_cache)

    return PolicyOutput(
        logits=logits,
        value=value,
        state=next_state,
    )
```

---

# 13. Navigation Hidden Selection

必须是独立模块，不要散落 magic indexing。

例如：

```python
class NavigationPooling(nn.Module):
    def forward(self, hidden_states, visual_token_mask):
        ...
```

首版建议：

```text
last recurrent token
```

或：

```text
learnable [NAV] token
```

更推荐显式 `[NAV]` token，因为：

```text
actor
critic
```

都可以稳定读取同一个 representation。

---

# 14. Environment Action Space

严格统一：

```text
0 STOP
1 MOVE_FORWARD
2 TURN_LEFT
3 TURN_RIGHT
```

所有 dataset / trainer / evaluator / serving 只能通过：

```python
NavigationAction
```

传递动作，禁止使用裸 magic integer。

---

# 15. Habitat Environment Adapter

核心接口：

```python
class ObjectNavEnv(Protocol):
    def reset(self) -> EnvReset:
        ...

    def step(
        self,
        action: NavigationAction,
    ) -> EnvStep:
        ...

    def get_oracle_action(self) -> NavigationAction:
        ...
```

`EnvStep`：

```python
@dataclass
class EnvStep:
    observation: Observation
    reward: float
    done: bool
    success: bool
    geodesic_distance: float | None
    collision: bool | None
```

---

# 16. Dataset Interface

统一：

```python
class EpisodeSource(Protocol):
    def sample_episode(self) -> NavigationEpisode:
        ...
```

具体：

```text
HM3Dv1EpisodeSource
HM3Dv2EpisodeSource
HM3DOVONEpisodeSource
MixedEpisodeSource
```

---

# 17. Mixed Dataset Sampler

```python
class MixedEpisodeSource:
    def __init__(self, sources, weights):
        ...
```

config：

```yaml
data:
  sources:
    hm3d_v1: 0.25
    hm3d_v2: 0.25
    hm3d_ovon: 0.50
```

trainer 完全不知道 episode 来自哪一套数据。

---

# 18. Unified End-to-End Trainer

只有一个主 trainer：

```python
class EndToEndObjectNavTrainer:
    def collect_rollout(self):
        ...

    def compute_losses(self):
        ...

    def update(self):
        ...

    def evaluate(self):
        ...

    def save_checkpoint(self):
        ...
```

不再存在：

```text
BCTrainer
DAggerTrainer
PPOTrainer
```

三个独立 trainer。

---

# 19. Parallel Rollout Architecture

```text
                 Learner GPU
                     │
              current weights
                     │
         ┌───────────┼───────────┐
         ▼           ▼           ▼
     Env Worker   Env Worker   Env Worker
         │           │           │
      Habitat      Habitat      Habitat
         │           │           │
         └────── rollout ────────┘
                     │
                     ▼
               RolloutBuffer
                     │
                     ▼
           Unified IL + PPO Update
```

首版可以：

```text
single-node multi-process
```

后续再扩展分布式。

---

# 20. Rollout Transition

```python
@dataclass
class Transition:
    episode_id: str
    step_index: int

    action: torch.Tensor
    expert_action: torch.Tensor

    old_log_prob: torch.Tensor
    value: torch.Tensor
    reward: torch.Tensor
    done: torch.Tensor
    entropy: torch.Tensor

    cache_checkpoint_ref: str | None
```

RGB 不建议全部直接塞 Python object。

使用：

```text
shared tensor buffer
or
preallocated rollout tensor
```

---

# 21. DAgger Action Mixing

实现：

```python
def select_env_action(
    policy_action,
    expert_action,
    beta: float,
):
    use_expert = torch.rand_like(...) < beta
    return torch.where(
        use_expert,
        expert_action,
        policy_action,
    )
```

scheduler：

```python
class DaggerBetaScheduler:
    ...
```

config：

```yaml
dagger:
  beta_start: 0.8
  beta_end: 0.05
  decay_updates: 10000
```

注意：

```text
expert_action 始终记录
```

即使最终环境执行的是 policy action。

---

# 22. Rollout Collection

伪代码：

```python
for t in range(cfg.rollout_steps):

    output = policy.forward_step(
        rgb=obs.rgb,
        state=states,
    )

    dist = Categorical(logits=output.logits)

    policy_action = dist.sample()
    expert_action = envs.get_oracle_actions()

    env_action = dagger.select(
        policy_action,
        expert_action,
    )

    next_obs, reward, done, info = envs.step(env_action)

    buffer.add(
        rgb=obs.rgb,
        state_ref=...,
        action=policy_action,
        executed_action=env_action,
        expert_action=expert_action,
        old_log_prob=dist.log_prob(policy_action),
        entropy=dist.entropy(),
        value=output.value,
        reward=reward,
        done=done,
    )

    states = output.state
```

episode `done` 后：

```text
reset Habitat env
reset KDA cache
prefill new instruction
```

---

# 23. GAE

独立模块：

```python
advantages, returns = compute_gae(
    rewards,
    values,
    dones,
    gamma,
    gae_lambda,
)
```

config：

```yaml
ppo:
  gamma: 0.99
  gae_lambda: 0.95
```

---

# 24. Imitation Loss

```python
il_loss = F.cross_entropy(
    action_logits,
    expert_actions,
    reduction="none",
)
```

每一个 on-policy state 都有 expert supervision。

这就是 DAgger 的核心价值。

---

# 25. PPO Loss

```python
ratio = torch.exp(
    new_log_prob - old_log_prob
)

loss1 = ratio * advantage
loss2 = torch.clamp(
    ratio,
    1.0 - clip_eps,
    1.0 + clip_eps,
) * advantage

ppo_loss = -torch.minimum(
    loss1,
    loss2,
)
```

---

# 26. Value Loss

```python
value_loss = 0.5 * (
    value - returns
).pow(2)
```

可选：

```text
clipped value loss
```

与 PPO 实现保持一致。

---

# 27. Entropy-Adaptive Loss Mixing

模块：

```python
class EntropyAdaptiveLossMixer(nn.Module):
    def forward(
        self,
        il_loss,
        ppo_loss,
        entropy,
    ):
        ...
```

定义：

```python
alpha = normalize_entropy(entropy)
alpha = alpha.clamp(0.0, 1.0)

policy_loss = (
    alpha * il_loss
    + (1.0 - alpha) * ppo_loss
)
```

原则：

```text
policy uncertainty高
    -> IL 权重高

policy uncertainty低
    -> PPO 权重高
```

具体归一化和 scheduler 参数全部 config 化，便于严格复现 OVSegDT-style EALM。

---

# 28. Unified Loss

```python
total_loss = (
    policy_loss.mean()
    + cfg.value_coef * value_loss.mean()
    - cfg.entropy_coef * entropy.mean()
)
```

唯一 optimization step：

```python
optimizer.zero_grad()
accelerator.backward(total_loss)
clip_grad_norm_(...)
optimizer.step()
```

不存在：

```text
先训 IL
保存
重新加载
再训 RL
```

---

# 29. Trainer Main Loop

```python
for update in range(num_updates):

    rollout = collect_rollout(
        policy,
        envs,
        dagger_scheduler,
    )

    advantages, returns = compute_gae(rollout)

    for epoch in range(update_epochs):
        for minibatch in rollout.minibatches():

            outputs = replay_policy(minibatch)

            losses = compute_e2e_loss(
                outputs,
                minibatch,
                advantages,
                returns,
            )

            optimize(losses.total)

    scheduler.step()
    dagger_scheduler.step()

    log_metrics()
    maybe_evaluate()
    maybe_checkpoint()
```

这就是整个训练系统。

---

# 30. Recurrent State Replay

PPO 会对 rollout 重放多次，因此不能只保存一个不可恢复的 Python cache object。

推荐两种实现，首版优先 A。

### A. Sequence minibatch

Rollout buffer 按：

```text
env_id × contiguous sequence
```

组织。

每个 sequence 开头保存：

```text
detached initial KDA state
```

优化时：

```text
restore initial state
replay T frames
compute logits/value
```

这是推荐方案。

### B. State snapshot per step

每一步保存 KDA state。

简单但内存浪费较大。

---

# 31. Sequence Rollout Buffer

```python
class RecurrentRolloutBuffer:

    observations: Tensor
    actions: Tensor
    expert_actions: Tensor
    rewards: Tensor
    dones: Tensor

    old_log_probs: Tensor
    old_values: Tensor

    initial_states: list[StreamingState]
```

shape：

```text
[num_steps, num_envs, ...]
```

minibatch 必须按 contiguous sequence 采样，禁止随机打乱单步 transition。

---

# 32. KDA State Detach

rollout 开头：

```python
initial_state = detach_state(current_state)
```

optimization replay：

```python
state = clone(initial_state)

for t in sequence:
    output = policy.forward_step(obs[t], state)
    state = output.state
```

防止 graph 跨 rollout 无限扩张。

---

# 33. Episode Reset Safety

必须实现：

```python
if done:
    state = policy.start_episode(
        episode_id=new_episode.id,
        instruction=new_episode.goal_text,
    )
```

严禁：

```text
old episode KDA state
    ->
new episode
```

---

# 34. Action Distribution Module

```python
class ObjectNavActionDistribution:

    def build(
        self,
        logits: Tensor,
    ) -> Categorical:
        return Categorical(logits=logits)
```

不要耦合 Transformers generation API。

---

# 35. Oracle Interface

```python
class ObjectNavOracle(Protocol):
    def action(
        self,
        env_state,
    ) -> NavigationAction:
        ...
```

实现：

```text
HabitatShortestPathOracle
```

模型代码不允许直接读取：

```text
navmesh
GT goal position
geodesic path
```

---

# 36. Reward Module

```python
@dataclass
class RewardConfig:
    success_reward: float
    slack_penalty: float
    progress_scale: float
    collision_penalty: float = 0.0
```

```python
class ObjectNavReward:
    def compute(...):
        ...
```

默认尽量与 benchmark baseline 接近。

---

# 37. Dataset Leakage Tests

自动检查：

```text
train scene
validation scene
OVON seen category
OVON unseen category
```

防止误把：

```text
val_unseen
```

目标词或场景放进训练集。

---

# 38. HM3D Dataset Manifests

```text
data/manifests/
├── hm3d_v1_train.json
├── hm3d_v1_val.json
├── hm3d_v2_train.json
├── hm3d_v2_val.json
├── hm3d_ovon_train.json
├── hm3d_ovon_val_seen.json
├── hm3d_ovon_val_seen_synonyms.json
└── hm3d_ovon_val_unseen.json
```

每次实验保存 manifest hash。

---

# 39. Habitat 独立服务

仍建议保持：

```text
model environment
!=
Habitat environment
```

原因：

- Qwen/KDA 依赖更新快；
- Habitat-Sim 对 Python/CUDA 版本敏感；
- 两边绑在一个环境会导致工程维护成本很高。

使用：

```text
ZeroMQ
```

即可。

---

# 40. Habitat Protocol

最少：

```text
RESET
STEP
GET_ORACLE_ACTION
CLOSE
```

`RESET` 返回：

```text
episode_id
goal_text
RGB
```

`STEP` 返回：

```text
RGB
reward
done
success
metrics
```

---

# 41. Config

核心配置可以非常简单：

```yaml
model:
  checkpoint: checkpoints/qwen35_0p8b_kda
  dtype: bfloat16
  action_dim: 4

data:
  hm3d_v1: 0.25
  hm3d_v2: 0.25
  hm3d_ovon: 0.50

rollout:
  num_envs: 32
  num_steps: 32

dagger:
  beta_start: 0.8
  beta_end: 0.05
  decay_updates: 10000

ppo:
  gamma: 0.99
  gae_lambda: 0.95
  clip_eps: 0.2
  update_epochs: 2
  minibatches: 4

ealm:
  enabled: true
  entropy_low: 0.2
  entropy_high: 1.2

loss:
  value_coef: 0.5
  entropy_coef: 0.01

train:
  precision: bf16
  max_grad_norm: 1.0
```

---

# 42. Checkpoint

```text
checkpoint_x/
├── model.safetensors
├── actor_critic.safetensors
├── model_config.json
├── kda_layout.json
├── tokenizer/
├── optimizer.pt
├── scheduler.pt
├── dagger_scheduler.json
├── resolved_config.yaml
└── manifest.json
```

---

# 43. Checkpoint Manifest

必须保存：

```text
git SHA
Qwen revision
FLA revision
transformers version
torch version
dataset manifests
config hash
KDA layout hash
```

---

# 44. Unit Tests

最少：

```text
test_action_enum.py
test_action_distribution.py
test_actor_critic_shapes.py
test_cache_reset.py
test_cache_detach.py
test_ealm.py
test_gae.py
test_ppo_loss.py
test_dagger_scheduler.py
test_dataset_mixture.py
```

---

# 45. GPU Regression Tests

必须有：

### KDA chunk/recurrent equivalence

```text
chunk forward
vs
step-by-step recurrent forward
```

### Cache boundedness

连续：

```text
500 steps
```

state memory 不随 episode length 线性增长。

### Episode isolation

```text
A -> reset -> B
```

必须与 fresh B 一致。

---

# 46. End-to-End Integration Test

测试链：

```text
Habitat reset
    ->
instruction prefill
    ->
RGB
    ->
KDA policy
    ->
categorical action
    ->
Habitat step
    ->
oracle label
    ->
rollout buffer
    ->
IL + PPO loss
    ->
optimizer step
```

只跑 2–4 个 environment step 即可。

这条 test 是整个项目最重要的 smoke test。

---

# 47. Latency Benchmark

必须测：

```text
prompt prefill latency
vision latency
KDA recurrent update latency
actor head latency
total observation-to-action latency

P50
P95
P99
```

不再测：

```text
autoregressive decode latency
RVQ decode latency
```

因为都已经不存在。

---

# 48. Streaming Memory Benchmark

测试：

```text
10 steps
50 steps
100 steps
250 steps
500 steps
```

记录：

```text
KDA state bytes
GPU allocated memory
GPU reserved memory
per-step latency
```

论文最关键的 systems figure 建议直接来自这里。

---

# 49. Evaluation

```bash
python -m streamnav.evaluation.runner \
    eval=hm3d_ovon \
    checkpoint=...
```

输出：

```text
SR
SPL
SoftSPL
distance-to-goal
episode length
collision rate

latency P50
latency P95
latency P99
peak VRAM
state bytes
```

---

# 50. Serving API

上层 Agent 只需要：

```python
session = nav.start(
    instruction="Find a chair"
)

action = nav.step(
    session,
    rgb,
)

nav.close(session)
```

动作：

```python
NavigationAction.MOVE_FORWARD
```

而不是字符串 generation。

---

# 51. Session Manager

```python
class NavigationSession:
    session_id: str
    instruction: str
    state: StreamingState
```

```python
class SessionManager:
    def create(...)
    def step(...)
    def reset(...)
    def close(...)
```

---

# 52. Goal Change

Agent 改目标时首版直接：

```text
reset recurrent state
+
prefill new instruction
```

不要隐式把旧目标 state 带到新目标。

后续如果需要连续 goal switching，再单独研究 instruction state editing。

---

# 53. Enterprise Quality

CI：

```bash
ruff check .
ruff format --check .
mypy src
pytest tests/unit
```

GPU CI：

```bash
pytest tests/gpu
```

integration：

```bash
pytest tests/integration
```

---

# 54. Error Types

```python
class StreamNavError(Exception):
    pass

class CacheStateError(StreamNavError):
    pass

class EpisodeMismatchError(StreamNavError):
    pass

class DatasetIntegrityError(StreamNavError):
    pass

class KDACompatibilityError(StreamNavError):
    pass
```

禁止 silent fallback。

---

# 55. Logging

每次 update 至少记录：

```text
total_loss
IL_loss
PPO_loss
value_loss
entropy

EALM alpha
DAgger beta

reward
success
SPL proxy
collision

policy action histogram
expert action histogram

rollout FPS
training FPS
GPU memory
```

这对诊断 STOP collapse / forward collapse 非常重要。

---

# 56. Action Collapse Monitoring

添加：

```python
ActionHistogramMetric
```

如果连续窗口出现：

```text
MOVE_FORWARD > 95%
```

或：

```text
STOP > threshold
```

触发 warning。

ObjectNav RL 很容易出现 action collapse，这个应该是 production-level trainer 的标准诊断项。

---

# 57. Gradient Monitoring

记录：

```text
vision encoder grad norm
KDA layer grad norm
actor grad norm
critic grad norm
```

避免出现：

```text
actor在学
backbone没梯度
```

或者反过来。

---

# 58. Parameter Groups

建议：

```python
optimizer_groups = [
    {
        "params": backbone.parameters(),
        "lr": backbone_lr,
    },
    {
        "params": actor_critic.parameters(),
        "lr": head_lr,
    },
]
```

虽然是统一端到端训练，但可以：

```text
backbone_lr < head_lr
```

这不是训练 stage，只是同一个 optimizer 的 param group。

---

# 59. Vision Encoder

默认参与端到端训练。

如果显存不足，可以 config：

```yaml
model:
  freeze_vision_encoder: false
```

但默认：

```text
false
```

因为目标就是端到端 ObjectNav。

---

# 60. KDA Backbone

默认参与端到端训练：

```text
trainable = true
```

不要把 KDA backbone 当 frozen feature extractor。

---

# 61. Evaluation Determinism

eval：

```text
action = argmax(policy_logits)
```

禁止 sampling。

训练：

```text
Categorical.sample()
```

保持 PPO exploration。

---

# 62. Recommended Implementation Order

不按“训练 stage”拆，而只按工程依赖顺序实现：

```text
1. KDA-Qwen3.5-0.8B forward + cache
2. Actor/Critic head
3. Habitat client + oracle
4. Mixed HM3D episode loader
5. recurrent rollout buffer
6. unified DAgger/PPO/EALM loss
7. parallel rollout trainer
8. evaluation
9. serving
```

这只是代码开发顺序，不是模型训练阶段。

---

# 63. First Vertical Slice

第一条必须跑通：

```text
HM3D episode reset
        │
        ▼
instruction prefill
        │
        ▼
KDA state
        │
RGB ────┘
 │
 ▼
KDA-Qwen3.5-0.8B
 │
 ├── Actor -> action
 └── Critic -> value
        │
        ▼
Habitat step
        │
        ├── reward
        └── oracle action
        │
        ▼
IL + PPO + value loss
        │
        ▼
backward
```

只要这条链能跑，核心项目就成立。

---

# 64. MVP Definition

```text
[ ] Qwen3.5-0.8B successfully KDA-converted
[ ] current RGB only
[ ] one-time instruction prefill
[ ] persistent KDA state
[ ] STOP/FORWARD/LEFT/RIGHT categorical policy
[ ] actor + critic
[ ] HM3D-v1
[ ] HM3D-v2
[ ] HM3D-OVON
[ ] online oracle labels
[ ] DAgger action mixing
[ ] PPO
[ ] entropy-adaptive IL/RL loss
[ ] single unified trainer
[ ] no segmentation loss
[ ] no RVQ
[ ] no autoregressive action generation
[ ] recurrent rollout buffer
[ ] cache reset safety
[ ] 500-step bounded-state test
[ ] SR/SPL evaluation
[ ] P50/P95/P99 latency evaluation
[ ] Agent-facing session API
```

---

# 65. 最终代码主路径

最终项目的关键文件实际上只有这些：

```text
models/qwen35_kda/backbone.py
models/qwen35_kda/cache.py
models/policy/actor_critic.py
models/policy/streaming_policy.py

training/trainer.py
training/rollout.py
training/rollout_buffer.py
training/dagger.py
training/ppo.py
training/ealm.py

envs/habitat_client.py
envs/oracle.py

data/mixture.py
evaluation/runner.py
```

主训练调用链：

```text
Trainer
  │
  ├── Vector Habitat Envs
  │
  ├── Streaming Policy
  │      ├── Qwen Vision
  │      ├── KDA Backbone
  │      ├── Actor
  │      └── Critic
  │
  ├── Oracle
  │
  ├── Recurrent Rollout Buffer
  │
  └── Unified Loss
         ├── DAgger IL
         ├── PPO
         ├── EALM
         ├── Value
         └── Entropy
```

这就是建议最终实现的版本。

---

# 66. References

- OVSegDT  
  https://github.com/CognitiveAISystems/OVSegDT

- Flash Linear Attention / KDA  
  https://github.com/fla-org/flash-linear-attention

- qingyi-kda  
  https://github.com/Sisyphbaous-DT-Project/qingyi-kda

- Qwen3.5-0.8B  
  https://huggingface.co/Qwen/Qwen3.5-0.8B

- Habitat-Lab  
  https://github.com/facebookresearch/habitat-lab

- HM3D-OVON  
  https://huggingface.co/datasets/nyokoyama/hm3d_ovon

- LightNav-0 — only for initial high-level architecture inspiration  
  https://github.com/lightorigins/LightNav-0
