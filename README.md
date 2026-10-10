# FastVLN / Streaming ObjectNav

基于 KDA-converted Qwen3.5-0.8B 的流式目标导航，训练方法参考本地 `third_party/OVSegDT`。策略输入为 **RGB＋目标文本**，输出六动作，不读取 semantic mask、GPS/compass 或上一动作，不使用 semantic loss。当前任务是 HM3D-OVON ObjectNav，目标文本是物体类别指令。

当前感知分支为 **revision 13**，增加可走点、目标位置和到达监督，预测参与六动作决策。八卡训练入口为 `runs/streamnav_active/ealm`，从无感知分支最佳 update 2500 初始化，目标新增 2500 updates，rollout 与 BPTT 均为 100 步。感知分支的 update 从零计，原有权重、Adam 状态和采样器保留；实时进度通过状态命令查看。

固定版本节点 **current_20261010** 保留 revision 12 的源码、配置、完整恢复检查点 update 2240 与节点最佳 update 2000。节点内最佳模型在固定 144 条诊断 episode 上 SR 为 18.75%，仍存在停车不足和持续转向问题。该节点不随当前实验更新；感知分支的收益需重新评估。

| 文档 | 内容 |
|---|---|
| [当前版本](docs/current_version.md) | 节点、依赖、验证结果与保留规则 |
| [架构](docs/architecture.md) | 流式消息、递归状态、策略与仿真接口 |
| [感知监督](docs/perception.md) | 双通道点位、到达状态、弱标签与动作融合 |
| [数据](docs/datasets.md) | 数据资产、划分、采样与清单校验 |
| [训练](docs/training.md) | 当前配方、加速、监控、评估与恢复 |
| [部署](docs/deployment.md) | 视频、Habitat 网页与会话 API |

## 环境

```bash
bash
source scripts/env.sh
# 已安装环境可直接使用；首次安装参考 scripts/setup.sh
```

学习器使用 `.venv`（Python 3.12），仿真器使用 `runtime/habitat-env`（Python 3.9）。数据、缓存和依赖放在项目内；`third_party/OVSegDT` 与 `runtime/vendor/frontier_exploration` 为保留的上游参照和教师依赖。

## 查看训练与视频

```bash
python tools/training_status.py --root runs/streamnav_active
tail -f runs/streamnav_active/ealm/training.log
cat runs/streamnav_active/ealm/health_status.json
```

每 100 updates 在三个 split 各评估 48 条固定 episode，并保存共 18 个代表 case 的 MP4、动作概率和碰撞 trace。播放页位于 `runs/streamnav_active/ealm/evaluation/update_*/index.html`。运行中的训练无需重新启动；保存停训和完整恢复命令见 [训练说明](docs/training.md#启动停止与恢复)。

```bash
# 可选交互回放：默认加载 best，没有 best 则加载 latest
STREAMNAV_VIEWER_GPU=7 bash scripts/start_viewer.sh
```

浏览器访问 `http://127.0.0.1:8765`，远程用 SSH 转发 8765 端口。八卡训练期间先检查显存余量，也可直接播放已保存的视频。

## 机器人与模型输入

| 项目 | 当前值 |
|---|---|
| RGB / HFOV | 480×270 / 120° |
| 相机高度 / 初始俯仰 | 0.88 m / 0° |
| 机体高度 / 半径 | 0.88 m / 0.18 m |
| 前进 / 转向 / 上下看 | 0.25 m / 30° / 30° |
| 动作 ID | STOP=0，MOVE_FORWARD=1，TURN_LEFT=2，TURN_RIGHT=3，LOOK_UP=4，LOOK_DOWN=5 |
| 成功条件 | 主动 STOP 且到目标视点距离 <0.25 m |
| navmesh max climb / cell height | 0.10 m / 0.05 m |

RGB 上下各补 9 行到 480×288，产生 135 个视觉 token。视觉编码器冻结；语言主干（KDA、GDN、MLP、embedding）、actor 和 critic 参与导航训练。

## 开发验证

```bash
make check
make unit
# GPU 与真实 Habitat 检查需要显存余量
make gpu
make integration
```

当前配方的完整配置保存在 [版本节点](runtime/baselines/current_20261010/resolved_config.yaml)。通用配置保留其他实验能力，直接运行默认 trainer 不等于当前主训练配方。历史更新流水已从文档移除；当前模型、完整训练记录和必要验证摘要的保留位置见 [当前版本](docs/current_version.md)。
