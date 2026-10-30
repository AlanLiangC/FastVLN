# FastVLN · update 15300

当前固定版本为 **Qwen3.5-0.8B + KDA + Point CoT + 单点 HINT**，以 Grounding 和专家 / DAgger / PPO 混合训练。默认加载 update **15300**。后续实验分支、权重及其代码改动已移除；训练处于停止状态。

| 内容 | 路径 |
|---|---|
| 当前模型 | `checkpoints/fastvln_15300`，含 adapter、完整八卡 Adam / RNG / sampler 状态 |
| 12200 参考基线 | `checkpoints/reference_12200`，仅推理对比，无优化器 |
| 官方冻结基座 | `checkpoints/qwen35_0p8b_base` |
| 当前训练配方 | [configs/pointcot_15300.yaml](configs/pointcot_15300.yaml) |
| 默认推理配置 | [configs/pointcot.yaml](configs/pointcot.yaml) |
| 相对 12200 的变化 | [版本差异](docs/changes_from_12200.md) |
| 300 例对照与视频 | [15300 / 12200 / OVSegDT](runs/comparison_300_15300/index.html) |

模型由官方 Qwen 基座与当前 checkpoint 的 `adapters.safetensors` 共同组成。基座冻结，训练 LoRA、KDA、点 / 动作头和适配层，共 **9,248,833** 个可训练参数。15300 的 tensor-content SHA256 为 `7daf959f45addcd10a2807495c7696df8f150f549861d93f59555755ecb01290`；校验方式是按参数名排序后累计名称与 tensor 原始字节，不是整个 safetensors 文件的哈希。

## 已测性能

同一批 300 个任务，各 split 100 例；FastVLN 使用 YOLOE 单点和 greedy，OVSegDT 使用各自原配置及 sampled 三 seed 均值。

| 模型 | Seen SR | Synonyms SR | Unseen SR | 总 SR | SPL |
|---|---:|---:|---:|---:|---:|
| FastVLN 15300 | 52.00% | 53.00% | 42.00% | 49.00% | 26.68% |
| FastVLN 12200 | 62.00% | 50.00% | 36.00% | 49.33% | 24.74% |
| OVSegDT | 54.33% | 37.33% | 41.33% | 44.33% | 19.74% |

15300 与 12200 总成功数相差 1/300；15300 的 SPL 与同义词表现更好，Seen 较低。这是固定样本的同任务对照，方法之间保留各自相机、机器人、滑动与解码设置，不等于论文全量基准，也不是单模块消融。

## 使用

```bash
source scripts/env.sh
# 静态结果与视频，不占推理 GPU
make viewer
# http://127.0.0.1:8765/
# 启动交互推理，默认固定到 15300
STREAMNAV_VIEWER_GPU=0 bash scripts/start_viewer.sh
# 独立评估，不覆盖保留结果
python -m streamnav.evaluation.runner checkpoint=checkpoints/fastvln_15300 run_dir=runs/manual_eval_15300
```

需要继续训练时才运行 `make train`：默认从完整 15300 状态恢复，输出至 `runs/finetune_15300`，不会修改固定发布权重。当前没有导航训练在运行。按此前资源保留要求，独立 GPU 守护留在 `runtime/operations/`，不属于模型版本；仅释放保留负载可执行 `touch runtime/operations/gpu_reservation/release.flag`。

## 文档

- [相对 12200 的准确变化](docs/changes_from_12200.md)
- [架构、输入与缓存](docs/architecture.md)
- [APOS / OPOS、HINT 与监督](docs/perception.md)
- [15300 训练配方与恢复](docs/training.md)
- [数据与运行环境](docs/datasets.md)
- [部署接口与视频](docs/deployment.md)
- [OVSegDT 与对照口径](docs/baselines.md)

`make check` 运行 lint、格式和类型检查，`make unit` 运行单元 / 回归测试。数据、安装环境、官方基座、YOLOE、OVSegDT 和必要缓存保留；`.config/` 是本地凭据目录，不进入版本源码快照。原始 15300 源码快照保存在当前 checkpoint 的 `source/`，模型与训练计算按该快照恢复，默认路径和文档已整理到固定版本。

当前改进训练分支：`runs/finetune_15300_grounded_hints`，八卡使用 `make train`。冻结 15300 的默认部署路径不变；改动和评估约束见 [训练文档](docs/training.md#从-15300-继续grounding-与-hint-鲁棒性)。
