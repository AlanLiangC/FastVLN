# FastVLN / Streaming ObjectNav

使用 KDA-converted Qwen3.5-0.8B 的流式目标导航。策略只读取 **RGB＋目标文本**，输出六个动作；不读取语义 mask、GPS/compass、上一动作，也没有 semantic loss。每个 episode 预填目标，随后通过固定大小的递归状态处理图像。

当前基线为 **revision 5 · 2026-10-08**，训练配方对照本地 `third_party/OVSegDT`。已保留 update 50 的完整检查点、训练源码和配置作为版本节点。该节点自主验证仍为 **0/144**，尚未证明导航效果改善；它是可追溯的实现基线。实时进展通过下方状态命令查看。

2026-10-08 的代码审查发现旧策略忽略目标和画面、转换 KDA 缺少输出 RMS normalization，以及随机 critic 的早期数值问题。旧训练已在 update 76 保存退出；revision 6 验证仍为 0/144，已在 update 30 保存退出。**revision 8** 使用每步目标查询、KDA 输出归一化、主干 LR=1e-5、零初始化 critic / LR=2.5e-5；actor LR=2.5e-4。它在 update 50 保存停训时，SR 仍为 0/144，但有 7 条轨迹进入成功距离，没有主动 STOP；训练 STOP 标签仅占约 0.39%。STOP 加权和近目标课程的短试验均未提升自主 SR，未切入主配方。

活动入口仍指向 `runs/streamnav_kda_stable_20261008/ealm`，现已从 update 50 **恢复八卡训练**，使用 `ovsegdt_kda_cached`：缓存冻结的视觉输出，重放仍重新计算目标、图像边界与 NAV embeddings。连续 8 次对照的完整参数哈希逐次一致；优化耗时观测约下降 6%，每卡增加约 105 MiB CPU 缓存。OVSegDT 损失、教师、奖励和六动作保留，不使用 semantic。训练上限为 200 updates，在 55、75、100、150、200 次自主验证，每 5 次保存；30 秒监控检查进程、数值和停滞，update≥200 且连续五轮全零时请求保存退出。当前尚未证明 SR 改善，实测以 `eval_metrics.jsonl` 为准。详细对照见 [训练说明](docs/training.md) 和 [本次审计](runtime/reports/stop_audit_20261008/audit.json)。

续训 update 55 的三组自主验证仍为 **0/144**；13/144 条轨迹进入成功距离（seen 7、synonyms 4、unseen 2），没有贪心 STOP，目标附近平均 STOP 概率约为 0.315% / 0.306% / 0.155%。八卡更新前重放误差保持 0，value loss 约 1e-4；这些证据说明训练数值稳定，尚未解决实际停车和导航成功。

| 文档 | 内容 |
|---|---|
| [当前版本](docs/current_version.md) | 版本节点、上游差异、验证证据、空间整理与保留规则 |
| [架构](docs/architecture.md) | 输入、递归状态、六动作、仿真与策略接口 |
| [数据](docs/datasets.md) | 默认 OVON 数据、资产、清单和隔离规则 |
| [训练](docs/training.md) | IL/PPO/EALM、八卡、验证、检查点与恢复 |
| [部署与可视化](docs/deployment.md) | Habitat 网页、固定节点回放与会话 API |

## 机器人与输入

| 项目 | 当前设置 |
|---|---|
| RGB / HFOV | 480×270 / 120° |
| 相机高度 / 初始俯仰 | 0.88 m / 0° |
| 机体高度 / 半径 | 0.88 m / 0.18 m |
| 前进 / 转向 / 上下看 | 0.25 m / 30° / 30° |
| 动作 ID | STOP=0、MOVE_FORWARD=1、TURN_LEFT=2、TURN_RIGHT=3、LOOK_UP=4、LOOK_DOWN=5 |
| 成功判定 | 主动 STOP 且到目标视点的距离 <0.25 m |
| 导航网格爬升 / 垂直栅格 | 0.10 m / 0.05 m |

图像上下各补 9 行，以 **480×288（宽×高）** 进入视觉编码器，共 135 个视觉 token。上下看改变相机俯仰，重置 episode 恢复 0°。导航网格按实际机体尺寸重建并缓存。训练、评估和网页共用输入处理。

## 环境与目录

```bash
cd /inspire/qb-ilm2/project/spatiotemporal-intelligence-research/ky26298/Projects/Active_Navigation/FastVLN
bash
source scripts/env.sh
# 仅首次安装环境时执行：bash scripts/setup.sh
```

学习器使用 `.venv`（Python 3.12）；仿真器使用 `runtime/habitat-env`（Python 3.9 / Habitat-Sim 0.3.3）。新增下载、依赖和缓存放在共享项目目录。现有数据、转换权重和教师环境可直接使用。

| 路径 | 用途 |
|---|---|
| `src/streamnav/`、`services/habitat_server/` | 策略、训练、服务与独立仿真器 |
| `configs/`、`tests/`、`scripts/`、`tools/` | 配置、验证、运行入口与诊断工具 |
| `third_party/OVSegDT/`、`runtime/vendor/` | 上游源码、固定版本教师依赖；运行和差分测试需要 |
| `runtime/data/` | 场景、episode 与数据清单 |
| `checkpoints/qwen35_0p8b_kda/` | 转换后的初始化权重 |
| `checkpoints/revision5_20261008/` | 固定保留的 update 50 完整检查点 |
| `runs/streamnav_active/ealm/` | 当前八卡实验的稳定入口 |
| `runtime/baselines/revision5_20261008/` | 节点信息、源码快照、测试证据与历史压缩归档 |

`.config` 是本地凭据目录，不复制进实验、归档或分享材料。数据使用方法见 [数据说明](docs/datasets.md)。

## 八卡训练与监控

```bash
# 查看现有训练；已有八卡任务运行时不重复启动
python tools/training_status.py --root runs/streamnav_active
tail -f runs/streamnav_active/ealm/training.log
cat runs/streamnav_active/ealm/health_status.json

# 启动新实验时使用一个新的目录名
STREAMNAV_KEEPALIVE=0 STREAMNAV_RUN_ROOT=runs/my_kda_stable_experiment \
  bash scripts/start_eight_gpu_training.sh model=qwen35_0p8b_kda_stable trainer=ovsegdt_kda_cached \
  trainer.num_updates=200 trainer.eval_first_update=false 'trainer.early_eval_updates=[5,10,25,50,100,150,200]' \
  trainer.eval_interval=10000 trainer.checkpoint_interval=5 \
  trainer.supervision.min_updates=200 trainer.supervision.zero_sr_patience=5

# 当前 update 完成后保存并退出（仅在需要停训时执行）
bash scripts/stop_training.sh

# 恢复已经停止的当前实验，要求日志与 latest 对齐
STREAMNAV_KEEPALIVE=0 STREAMNAV_RUN_ROOT=runs/streamnav_active STREAMNAV_RESUME=1 \
  bash scripts/start_eight_gpu_training.sh model=qwen35_0p8b_kda_stable trainer=ovsegdt_kda_cached \
  trainer.num_updates=200 trainer.eval_first_update=false 'trainer.early_eval_updates=[55,75,100,150,200]' \
  trainer.eval_interval=10000 trainer.checkpoint_interval=5 \
  trainer.supervision.min_updates=200 trainer.supervision.zero_sr_patience=5
```

默认一个实验使用全部 8 卡；每卡 4 环境 ×100 步，每 minibatch 为 2 条完整序列，2 个 minibatch、1 epoch。每 update 3,200 transitions；312,500 updates 对应 10 亿步预算。策略自行采样动作，教师只提供 IL 标签。视觉编码器冻结，第 1 次 update 只训练线性 critic，之后解冻 actor / 递归主干。其余参数见 [训练说明](docs/training.md) 和 [默认配置](configs/trainer/ovsegdt_e2e.yaml)。

revision 5 默认在第 1、25、50、100、200… 次更新进行自主验证；活动实验已完成 5、10、25、50 次验证，续训将在 55、75、100、150、200 次验证。每轮使用三个 OVON split 各 48 条固定分层 episode，argmax、原始起点、最多 500 步。结果写入 `eval_metrics.jsonl`；视频位于 `evaluation/update_*/`。增加了目标附近 STOP 概率、错误停车和到达后未停车的诊断，距离不参与策略决策。这是诊断子集，不能替代全量 benchmark。教师成功不代表策略成功。

后台监控每 30 秒检查进程、非有限值、更新停滞和自主 SR；异常时请求安全保存并停训。矩阵保活只在 GPU 连续两小时低于 50% 时短时触发，独立记录、不计训练吞吐，可用 `STREAMNAV_KEEPALIVE=0` 关闭。Docker 或驱动退出后需手动恢复。

## 仿真器可视化

```bash
# 默认选择 best，没有 best 则选择 latest；八卡训练时先确认显存余量
STREAMNAV_VIEWER_GPU=7 bash scripts/start_viewer.sh
# 查看正在运行的服务，无需重启
curl -fsS http://127.0.0.1:8765/health
```

浏览器打开 http://127.0.0.1:8765；远程使用 `ssh -L 8765:127.0.0.1:8765 <服务器>`。选择验证集和 episode，点击“加载 / 重置”，再执行“模型单步”或“模型自动运行”。支持手动六动作、动作概率和状态查看。启动脚本默认 GPU 3，上述命令显式选择 GPU 7；已运行的服务不必再次启动。固定节点回放和 API 见 [部署说明](docs/deployment.md)。

## 开发验证

```bash
make check
make unit
# 下列检查需要 CUDA / 真实 Habitat，请在有空闲 GPU 时执行
make gpu
make integration
```

数学回归直接调用本地 OVSegDT 的 EALM、value clipping 和 PIRLNav 调度代码比较。源码快照、环境版本和既有验证结果见 [当前版本](docs/current_version.md)。修改训练配方需新建实验，不与历史 revision 的曲线拼接。
