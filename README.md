# FastVLN

当前模型为 **Qwen3.5-0.8B + 跨帧 KDA + APOS → OPOS → 当前动作 + 四步动作计划**，使用 grounding 和专家 / DAgger / PPO 混合训练，并增加轻量到达辅助头。基座冻结，可训练参数约 898.2 万，占总参数 1.04%。

每步输入目标文本、当前 RGB、一个可选目标点 token 和该 episode 的缓存；输出两个点、六动作 logits、四步计划、value 和下一状态。模型不输入历史 RGB、深度、语义图、地图或真值位姿，也没有分割分支。跨帧缓存为固定 12 MiB。

## 当前入口

| 内容 | 路径 |
|---|---|
| 当前训练配置 | [configs/pointcot_temporal.yaml](configs/pointcot_temporal.yaml) |
| 本次分支父权重 | `runs/qwen35_pointcot_hint_conditioned/checkpoints/update_0010380`，保留完整训练状态 |
| 官方 Qwen 基础权重 | `checkpoints/qwen35_0p8b_base`，原始参数冻结 |
| 活动导航训练 | `runs/qwen35_pointcot_temporal` |
| 最新导航权重 / 续训入口 | `runs/qwen35_pointcot_temporal/checkpoints/latest` |
| 评估选中的导航权重 | `runs/qwen35_pointcot_temporal/checkpoints/best` |
| 动作融合对照与视频 | [288 条闭环对照](runs/temporal_policy_validation_20261020/index.html) |
| Grounding 初始化权重 | `runs/pointcot_grounding_refined/checkpoints/best`，update 800 |
| Grounding 最终恢复点 | `runs/pointcot_grounding_refined/checkpoints/latest`，update 2000 |
| 验证样本 | `runtime/fixtures/navigation/`，与实验输出分开保存 |

导航模型由官方基座和训练检查点内的 `adapters.safetensors` 共同组成；后者包含 LoRA、KDA 和任务头参数。`latest` 随训练保存更新，`best` 按 simulated_hint 的固定验证集成功率选择；对应 update 以各自 `manifest.json` 为准。检查点还包含 Adam、各 rank 的 RNG / 采样状态及配置，源码快照保存在运行目录的 `source/`。

## 恢复训练与查看结果

```bash
source scripts/env.sh
# 当前八卡训练的恢复入口；运行中的任务无需重复启动
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 STREAMNAV_WORLD_SIZE=8 \
  bash scripts/start_pointcot_training.sh \
  --resume runs/qwen35_pointcot_temporal/checkpoints/latest --skip-initial-eval
python tools/training_status.py --root runs/qwen35_pointcot_temporal
# 保存一个完整 update 后暂停
bash scripts/stop_training.sh runs/qwen35_pointcot_temporal
# 静态视频查看，不占用推理 GPU
make viewer
# http://127.0.0.1:8765/evaluation/
```

当前分支从 update 10380 延续 Adam、采样器和 PPO 计划，启用学生轨迹的因果未来动作监督，保留原有 HINT 条件监督。使用 GPU 0–7，每卡 3 个环境，rollout=100、反向 fragment=20，每 update 共 2400 条转移和两次同步 optimizer step。每 5 update 保存，每 100 update 双设置评估；日常每 split 12 条，每 500 update 扩至 48 条，每次保存 12 个视频。换卡数时显式使用 `--allow-world-size-change`，详见训练文档。

当前仍逐帧执行一步。冻结 update 10380 的四组闭环对照显示，融合旧规划虽减少摇摆，却未同时通过有／无 HINT 的成功率、SPL 和误停回归标准，因此动作融合与到达概率校准默认关闭。先补齐学生状态上的规划监督，再用同组案例验证是否适合启用，不能把少摇摆直接当作导航提升。

## 结果口径

以 0.25 m 内执行 STOP 为成功。当前最佳权重及其成绩见运行目录的 `best_evaluation.json`；各轮结果和视频见 `evaluation/update_XXXXXXX/`。best 按 simulated_hint 的固定 36 条 anchor 选择，同分保留原 best，不能与 144 条评估直接混比。

历史评估示例：update 5000 共 144 条，无提示 SR **33.3%**、SPL **18.4%**；模拟干净实例点提示 SR **34.0%**、SPL **19.7%**。这些数值仅属于当时的权重，分析见 `runtime/reports/success_rate_5100_analysis.json`。推理计时口径见 [部署文档](docs/deployment.md)。

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
