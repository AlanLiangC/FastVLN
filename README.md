# FastVLN

当前节点：**`qwen35_pointcot_20261017`**。模型为 **Qwen3.5-0.8B + 跨帧 KDA + APOS → OPOS → 当前动作 + 四步动作计划**，使用 grounding 和专家 / DAgger / PPO 混合训练。基座冻结，可训练参数约 897.8 万，占总参数 1.04%。

每步输入目标文本、当前 RGB、一个可选目标点 token 和该 episode 的缓存；输出两个点、六动作 logits、四步计划、value 和下一状态。模型不输入历史 RGB、深度、语义图、地图或真值位姿，也没有分割分支。跨帧缓存为固定 12 MiB。

## 当前入口

| 内容 | 路径 |
|---|---|
| 配置 | [configs/pointcot.yaml](configs/pointcot.yaml) |
| 活动导航训练 | `runs/qwen35_pointcot_opos_hint` |
| 已完成 grounding | `runs/pointcot_grounding_refined`，best 为 update 800 |
| 固定节点 | [checkpoints/versions/qwen35_pointcot_20261017/manifest.json](checkpoints/versions/qwen35_pointcot_20261017/manifest.json) |
| 固定训练权重 | `checkpoints/versions/qwen35_pointcot_20261017/navigation`，update 5155 |
| 节点选中的 best | `checkpoints/versions/qwen35_pointcot_20261017/best`，update 4400 |
| 验证样本 | `runtime/fixtures/navigation/`，与实验输出分开保存 |

固定节点包含完整训练权重、Adam、八路 RNG / 采样状态、配置、源码及校验清单。活动训练继续更新 `checkpoints/latest`，节点不参与滚动清理。节点源代码入口为 [source.zip](checkpoints/versions/qwen35_pointcot_20261017/source.zip)；原训练使用的源码快照另存于节点目录。冻结基座、数据和运行环境继续使用本项目已有路径。

## 恢复训练与查看结果

```bash
source scripts/env.sh
# 当前八卡训练的恢复入口；运行中的任务无需重复启动
bash scripts/start_eight_gpu_training.sh \
  --resume runs/qwen35_pointcot_opos_hint/checkpoints/latest --skip-initial-eval
python tools/training_status.py --root runs/qwen35_pointcot_opos_hint
# 保存一个完整 update 后暂停
bash scripts/stop_training.sh runs/qwen35_pointcot_opos_hint
# 静态视频查看，不占用推理 GPU
make viewer
# http://127.0.0.1:8765/evaluation/
```

导航默认 GPU 0–7，每卡 3 个环境，rollout=100、反向 fragment=20，每 update 共 2400 条转移和两次同步 optimizer step。每 5 update 保存，每 100 update 双设置评估；日常每 split 12 条，每 500 update 扩至 48 条。每次评估保存 12 个视频。换卡数时显式使用 `--allow-world-size-change`，详见训练文档。

## 结果口径

以 0.25 m 内执行 STOP 为成功。最近较大评估 update 5000 共 144 条：无提示 SR **33.3%**、SPL **18.4%**；模拟干净实例点提示 SR **34.0%**、SPL **19.7%**。update 5100 的 36 条日常评估为 **27.8% / 30.6%**；5000 在同一组无提示 anchor 上也是 27.8%。这些成绩属于对应评估权重，固定节点 5155 尚无独立评估。

当前瓶颈是错误 STOP 和目标接近控制；5000 无提示中 72/144 条错误停止。best=4400 按固定 36 条 anchor 选择，不能与 144 条评估直接混比。详情见 `runtime/reports/success_rate_5100_analysis.json`。推理计时口径见 [部署文档](docs/deployment.md)。

## 文档与检查

- [架构与缓存](docs/architecture.md)
- [Point CoT、提示与 loss](docs/perception.md)
- [训练、保存、恢复和评估](docs/training.md)
- [数据、资产准备与防泄漏](docs/datasets.md)
- [推理接口、延迟与视频](docs/deployment.md)

```bash
make check
make unit
```

环境安装入口为 `scripts/setup.sh`，基座下载入口为 `tools/download_qwen_weights.py`。公开数据从 `/inspire/hdd/global_public/public_datas` 读取，当前 grounding 共 1,025,912 条训练查询；数据、安装环境、navmesh 和 Triton 缓存均保留。凭据位于 `.config/`，不进入版本快照。

参考实现保留在 `third_party/LightNav-0`、`third_party/OVSegDT` 和 `third_party/Uni-NaVid`。Qwen 基座固定为官方 revision，见数据文档。
