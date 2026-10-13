# FastVLN

当前实现：**Qwen3.5-0.8B + 跨帧 KDA + APOS → OPOS → 六动作 point CoT**。策略输入为目标文本、当前 RGB 和 episode 独立缓存。基座冻结，约 0.98% 参数训练；先学习 grounding，再单独启动 HM3D-OVON 的 DAgger/PPO。

主配置：[configs/pointcot.yaml](configs/pointcot.yaml)。当前 grounding 数据为 **1,025,912 条训练查询**：COCO 实例、RefCOCO/+/g 指代表达、ScanNet 室内实例、ImageNet 弱区域和已有自然图像 PixMo。约 49.9 万条训练查询包含真实实例的有效点区域，明确缺失采样率 2.5%；详情见 [数据文档](docs/datasets.md)。

## 准备与启动

```bash
source scripts/env.sh
python tools/download_qwen_weights.py
python tools/download_referring_annotations.py
python tools/prepare_public_grounding.py --workers 24
python tools/prepare_grounding_support.py --workers 12
make unit
make check
# 六卡 grounding，GPU 6、7 留给其他任务
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 STREAMNAV_WORLD_SIZE=6 bash scripts/start_grounding_training.sh --initialize-from runs/pointcot_grounding/checkpoints/update_0004400
```

公共图像直接从 `/inspire/hdd/global_public/public_datas` 读取。ScanNet 仅导出筛选后的 RGB；实例掩码只用于离线生成点及有效网格区域。学生不接收语义、深度、地图或真值位姿，也没有分割预测分支。

当前运行目录为 `runs/pointcot_grounding_refined`。每卡 batch 32，全局 batch 192；当前配置是从保留的 4400 权重进行 2000 次小学习率改进更新，重新初始化优化器和采样器，update 从 0 计数。原训练的最佳和暂停权重保留在 `runs/pointcot_grounding`。从基座开始训练时需另设完整训练预算和学习率，不能把这 2000 次改进更新当作完整预训练。使用批量视觉编码、并行图像读取、FP32/TF32 计算和 fused AdamW。先保存、评估并保留 update 0 的定位基线，然后每 200 update 验证同一组 1,536 条查询并保存 32 个案例及全部验证预测。

```bash
make viewer VIEWER_RUN_DIR=runs/pointcot_grounding_refined
# http://127.0.0.1:8765/grounding_visualizations/
bash scripts/stop_training.sh runs/pointcot_grounding_refined
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 STREAMNAV_WORLD_SIZE=6 bash scripts/start_grounding_training.sh --resume runs/pointcot_grounding_refined/checkpoints/latest
```

可视化包含原图、预测、真值、有效实例轮廓、总览和逐例 JSON。洋红菱形为预测 OPOS，青绿圆圈为标注点，绿色轮廓为真实实例有效区域，橙框为弱区域。实例内部命中、标注点距离命中、误判缺失率和验证 loss 分开报告；弱框不计入可见目标主指标。日志和分来源评估分别为 `training.log`、`train_metrics.jsonl`、`grounding_eval.jsonl`。

导航需要经过定位验证的 grounding 权重，单独启动：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 STREAMNAV_WORLD_SIZE=6 bash scripts/start_pointcot_training.sh --initialize-from runs/pointcot_grounding_refined/checkpoints/best
```

## 文档

- [架构与缓存](docs/architecture.md)
- [point CoT 与监督](docs/perception.md)
- [训练、暂停、评估](docs/training.md)
- [数据与下载](docs/datasets.md)
- [推理与视频](docs/deployment.md)

基座来自 [官方 Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B)。point CoT 参考 [LightNav-0](https://github.com/lightorigins/LightNav-0)，动作模仿与 PPO 参考本地 [OVSegDT](third_party/OVSegDT)。
