# FastVLN / LightNav-KDA

当前版本以官方 **LightNav-0 / Qwen3-VL-4B** 为基础，输入当前 RGB 和目标文本，跨帧历史由 KDA 递归缓存保存。保留 APOS → OPOS 的 point CoT，随后生成六动作之一；不使用原三阶段 RVQ 轨迹，不输入 semantic、GPS/compass 或历史图片列表。

训练采用低参数量微调：冻结预训练视觉、MLP、embedding 和原词表，训练注意力 LoRA、KDA 门控与残差、六个动作 token 和 critic。训练期使用原 LightNav 的历史帧点位分布做稀疏保留监督，并结合 DAgger 动作监督和 PPO。配置在 [lightnav_kda.yaml](configs/lightnav_kda.yaml)。新架构的效果以实际评估为准。

| 文档 | 内容 |
|---|---|
| [架构](docs/architecture.md) | 原生注意力、KDA 缓存、动作与参数冻结 |
| [点位](docs/perception.md) | APOS/OPOS、教师分布和视频标注 |
| [数据](docs/datasets.md) | HM3D-OVON 数据与任务边界 |
| [训练](docs/training.md) | 损失、训练阶段、评估和恢复 |
| [部署](docs/deployment.md) | 当前帧推理与视频访问 |

## 使用

```bash
source scripts/env.sh
make check
make unit
make gpu
bash scripts/start_lightnav_training.sh
python tools/training_status.py --root runs/lightnav_kda_active
tail -f runs/lightnav_kda_active/training.log
```

学习器使用 `.venv`，仿真器使用 `runtime/habitat-env`。八卡默认每卡两个环境：每个 update 采集 100 步，即 1600 条 transition；反向传播按 20 步截断，缓存跨片段保留，每轮仅更新一次参数。

每 100 updates 在三个验证划分各评估 48 条固定 episode，保存带鲜明 APOS/OPOS 标记的视频和动作 trace。视频页位于 `runs/lightnav_kda_active/evaluation/update_*/index.html`。

```bash
make viewer
# 浏览器访问 http://127.0.0.1:8765/evaluation/
```

官方底座固定为 `LightOriginsHQ/LightNav-0` revision `826dc5fbfa37afa8293d2e336d329b6ffc0bfb64`，文件 SHA256 随权重保存于 `checkpoints/lightnav0_base/source_revision.json`。训练检查点只保存可训练参数和优化器，加载时校验底座与参数集合。

旧版本的必要最终检查点、评估结论和清理清单在 `runtime/archive/legacy_streamnav_20261011`。旧实验目录和更新流水已清理；旧代码中保留的公共组件及兼容测试不作为当前训练入口。

## 仿真设置

RGB 480×270、HFOV 120°；模型按原 LightNav 预处理为 448×256。机体高度和相机高度 0.88 m、半径 0.18 m；前进 0.25 m，转向和俯仰各 30°。动作 ID 为 STOP=0、FORWARD=1、LEFT=2、RIGHT=3、UP=4、DOWN=5；主动 STOP 且距离目标视点小于 0.25 m 判定成功。
