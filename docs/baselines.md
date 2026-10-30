# OVSegDT 与固定 300 例对照

结果入口为 `runs/comparison_300_15300/index.html`；`summary.json` 包含逐例结果、总指标和分 split 指标，`protocol.json` 记录 case 哈希、模型哈希、各方法机器人配置、解码和 seed。只保留 15300、12200 与 OVSegDT 的同任务证据。

## 配置与结果

Seen / Synonyms / Unseen 各 100 个固定任务。FastVLN 使用 YOLOE-v8l-seg 的一个像素点、greedy 解码；OVSegDT 使用原方法 sampled 设置，seed=100/101/102，共 900 条轨迹，按三 seed 均值汇总。各方法保留自身相机、机器人和滑动设置，最大 500 步，成功为 <0.25 m 且 STOP。

| 模型 | Seen SR / SPL | Synonyms SR / SPL | Unseen SR / SPL | 总 SR / SPL |
|---|---:|---:|---:|---:|
| FastVLN 15300 | 52.00 / 26.80 | 53.00 / 30.33 | 42.00 / 22.92 | 49.00 / 26.68 |
| FastVLN 12200 | 62.00 / 29.77 | 50.00 / 26.84 | 36.00 / 17.63 | 49.33 / 24.74 |
| OVSegDT | 54.33 / 23.34 | 37.33 / 16.54 | 41.33 / 19.33 | 44.33 / 19.74 |

数值单位为百分比。15300 不是所有指标都优于 12200：Seen 更低，总 SR 只差 1 例；保留它作为当前版本是基于更高 SPL 和同义词 / 陌生类别表现。与 OVSegDT 的 Unseen SR 接近，不应夸大 0.67 个百分点的差异。

这是固定任务子集、各自原设置的方法对照，不是统一机器人 / 相机的受控模块比较，也不是论文全量 split。独立测试集没有加入训练；不能把模拟提示或训练期小验证集的成绩与这张表混比。

## 可复用资源

- OVSegDT 导航权重：`checkpoints/ovsegdt/ckpt.5.pth`。
- YOLOE-v8l-seg：`checkpoints/ovsegdt/yoloe-v8l-seg.pt`，同时用于 FastVLN 的真实点来源。
- 环境：`runtime/ovsegdt-env`、`runtime/ovsegdt-sim-env` 与 `runtime/habitat-env`。
- 资产和依赖记录：`configs/baselines/`。
- 工具：`tools/compare_navigation_baselines.py` 与 `tools/ovsegdt_runtime.py`，参考实现保留于 `third_party/OVSegDT`。

页面仅对已录制的案例显示视频；未录视频的任务仍保留 trace 和指标，避免把视频子集当作全部样本。重测时使用新的输出目录，不覆盖保留结果。

## 全量 HM3D ObjectNav V2 验证

入口为 `runs/evaluation_v2_full_15300/index.html`。本轮只测 FastVLN 15300 + YOLOE 单点，不运行 OVSegDT 导航模型。完整 V2 val 为 36 场景、1,000 例：bed 165、chair 195、plant 152、sofa 187、toilet 166、tv_monitor 135；它与上面的 OVON 300 例不是同一个测试集。

原 0.10 m 攀爬设置已完成 1,000/1,000：635 例成功，SR **63.50%**、SPL **33.52%**，平均 177.83 步。完整逐例结果及分类指标见该目录 `summary.json`。

保持 FastVLN 自身相机、机器人、允许滑动和逐步 greedy 推理，最大 500 动作，STOP 且距离 <0.25 m 才算成功。YOLOE 权重、224×224 预处理、单点提取及阈值沿用现有工具；固定词表采用完整 `val_seen`，再加入 `plant` 及 `val_unseen` 中的 plant 阈值 0.1，不按当前目标裁减词表。检测类别 `sofa→couch`、`tv_monitor→tv`；导航指令只把内部类别代码 `tv_monitor` 改写为自然词 `tv`。不使用真值点或 oracle 动作。

V2 的原始 episode ID 在各场景的各类别中重新计数，所以评测 UID 加上类别，并在 metadata 保存原始 ID。清单包含全部 1,000 个不同的场景／类别／ID 组合，不按原始 ID 去重。每卡 125 例，可断点续测；预选每类 3 例，共 18 例录制视频，所有案例保存逐步 trace。根目录 `summary.json` 只汇总已完成的案例；`progress.json` 为进度，完成前的 SR 不是最终成绩。

### 并发评测

每卡默认运行 4 个独立进程，共 32 路。每路保留 batch=1、独立模型／检测器／模拟器／KDA 状态，通过独立的 CUDA MPS 服务重叠计算。没有更改动作解码、数值精度、500 步上限或成功判据。283 条已完成的串行结果保留于原目录，其余 717 条按 `execution.json` 固定分配给 `shards/gpu_*/lane_*`；汇总递归读取两层目录并拒绝重复案例。

2026-10-29 的 64.4 秒稳定窗口：八卡合计约 270 动作/秒，平均 GPU 利用率 91.5%；串行版本在排除启动和 GPU 0 并发测试干扰后，七卡测得每卡 11.25–13.39 动作/秒，折合八卡约 102，窗口吞吐提升约 2.65 倍。并发版本显存约每卡 29–30 GB。该窗口包含 GPU 0 上额外的轨迹一致性复跑，属于实际评测流水线吞吐，不是单机器人延迟。不同案例长度和最后少量任务的负载不均会影响全程耗时，不能把窗口倍率直接视作整轮加速比。

测量和一致性记录在 `runtime/reports/v2_eval_parallel_benchmark/`：`throughput.json`、`mps_samples.json`、`parity.json`、`mps_parity.json`。非 MPS 四路并发的 18 条复跑、2,349 步，以及 MPS 下六类目标各一例的 6 条复跑、783 步，均已逐步核对动作、HINT、APOS、OPOS 与串行结果完全一致；MPS 复跑的初始 RGB 哈希和终局指标也完全一致。

启动或续测：

```bash
source scripts/env.sh
# 仅在本项目私有 MPS 服务尚未启动时执行；不要启动第二份。
mkdir -p "$STREAMNAV_RUNTIME/logs/mps-v2"
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  CUDA_MPS_PIPE_DIRECTORY=/tmp/fastvln-v2-mps \
  CUDA_MPS_LOG_DIRECTORY="$STREAMNAV_RUNTIME/logs/mps-v2" nvidia-cuda-mps-control -d
python tools/launch_job.py --name evaluation --run-dir runs/evaluation_v2_full_15300 -- \
  tools/evaluate_hm3d_v2.py --gpus 0 1 2 3 4 5 6 7 --workers-per-gpu 4 \
  --mps-pipe-directory /tmp/fastvln-v2-mps
```

先检查该目录 `evaluation.pid` 对应的进程已退出，避免同时启动两份；控制器额外使用文件锁防止重复调度。固定清单、类别映射、权重哈希及分卡分配保存在 `protocol.json`，改变科学评测协议时必须换输出目录；并发方式单独记入 `execution.json`。续测保持原有 lane 分配。

## 统一通行参数的全量 V1、V2 与 OVON 队列

入口为 `runs/evaluation_full_15300_hm3d/index.html`，只测同一份 FastVLN 15300 + YOLOE 单点：

| Benchmark | Episodes | Scenes |
|---|---:|---:|
| HM3D V1 val | 2,000 | 20 |
| HM3D V2 val（统一参数重测） | 1,000 | 36 |
| OVON val_seen | 3,000 | 36 |
| OVON val_seen_synonyms | 3,000 | 36 |
| OVON val_unseen | 3,000 | 36 |
| 合计 | 12,000 | 场景存在重叠，不相加 |

相机、机器人尺寸、滑动、500 步上限、0.25 m STOP 成功判据、精度和逐步 greedy 推理均沿用 V2。评测专用的最大攀爬高度从 0.10 m 改为 0.20 m，navmesh 垂直栅格高度从 0.05 m 改为 0.20 m，所有五个 benchmark 使用同一设置。原始 0.10 m navmesh 在 V1 中切断楼梯并使部分起点无效：实测案例原地转向导致 1.45 m 位移；两项均改为 0.20 m 后约为 0.9 mm，且恢复有限路径。五个问题案例的起始测距与原数据集 geodesic_distance 在浮点误差范围内一致；仅修改攀爬高度而保留 0.05 m 垂直栅格仍有任务不连通。测距回退不能解决起点移动，因此未采用；严格测距与机器人运动共用同一 navmesh。模型权重、训练配置和推理逻辑未改。此前 0.10 m 的 V2 和 OVON 300 例结果保留为历史口径，不能把与新结果的差异全部归因于模型能力。V1 使用与 V2 相同的固定六类目标检测映射；OVON 保留各 split 自己的校准词表和阈值，以及原实现的 SigLIP 最近类别映射。OVON 的目标文本和 episode UID 不变，便于与保留的 300 例对照。

V1 的原始 ID 在同一场景、同一类别内也可能重复，因此额外加入完整清单中的索引来区分缓存状态；完整 2,000 条起点／目标记录均保留，不按原始 ID 去重。`cases_index.json` 记录独立 UID、原始 ID、场景、类别和原始起始位姿。

工具 `tools/evaluate_navigation_full.py` 将 12,000 例拆成 377 个至多 32 例的固定块。每卡最多 4 个独立 batch=1 进程，通过 MPS 并行；某个块完成后，该卡从全局队列领取下一块，减少固定分卡的尾段空闲。块首次领取时记录 GPU，异常退出在同一 GPU 断点重试，避免改变该块的协议。导航权重只固定复制一份，所有块共享读取，不重复保存 377 份权重。结果逐例原子写入，协议和每块案例哈希在运行和恢复时校验。

每个 benchmark 预选 18 个案例保存视频，共 90 个，其余同样保存完整 trace。`summary.json` 的 `by_split` 分开报告五组 SR/SPL，`by_category` 提供组内分类指标；完成前的指标只对应已经结束的案例，不代表最终成绩。

在本项目私有 MPS 服务已经运行、该输出目录没有控制器占用时启动／续测：

```bash
source scripts/env.sh
python tools/launch_job.py --name evaluation --run-dir runs/evaluation_full_15300_hm3d -- \
  tools/evaluate_navigation_full.py --gpus 0 1 2 3 4 5 6 7 --workers-per-gpu 4 \
  --chunk-size 32 --navmesh-max-climb 0.2 --navmesh-cell-height 0.2 --mps-pipe-directory /tmp/fastvln-v2-mps
```

0.10 m 的 V1 初始尝试因任务不可达而中止，原始异常和部分结果仅保留在 `runtime/reports/full_v1_ovon_preflight/robot_climb01_attempt/` 作为诊断，不混入新成绩。climb=0.20/cell_height=0.05 的中间验证也仅作为诊断，存于同级 `robot_climb02_cell005_attempt/`。最终闭环记录为同级 `raster_02_validation.json`。

## 12200 / 15300 与 YOLOE 开关的配对复核

本次为五个 benchmark 各随机固定 300 例（seed=20261030），同一案例比较四组，共 6,000 条轨迹：1,500 条复用完整 15300+YOLOE 结果，其余 4,500 条新增。所有起点、朝向、目标、相机、机器人、climb/cell_height=0.20 m、500 步和 STOP 距离 <0.25 m 一致，greedy、batch=1。每个四元组的初始 RGB 哈希和 episode UID 已核对；no_hint 的 HINT 帧数全部为零。

| Split | 12200 无提示 SR / SPL | 12200 YOLOE SR / SPL | 15300 无提示 SR / SPL | 15300 YOLOE SR / SPL |
|---|---:|---:|---:|---:|
| hm3d_v1_val | 58.00 / 25.35 | 56.33 / 29.19 | 59.00 / 31.12 | 55.33 / 31.49 |
| hm3d_v2_val | 65.67 / 28.61 | 67.33 / 34.26 | 69.33 / 32.94 | 65.00 / 33.03 |
| hm3d_ovon_val_seen | 51.33 / 23.52 | 48.67 / 26.80 | 48.00 / 25.30 | 46.33 / 26.40 |
| hm3d_ovon_val_seen_synonyms | 48.67 / 24.62 | 48.00 / 25.27 | 47.00 / 27.02 | 48.67 / 29.90 |
| hm3d_ovon_val_unseen | 39.00 / 14.94 | 39.33 / 18.73 | 35.00 / 15.47 | 37.33 / 17.37 |

表中是同一固定子集的百分比，不能当作旧权重／无提示模式的全量指标。OVON 三组共 900 例：12200 无提示 SR 46.33%、SPL 21.02%；15300+YOLOE SR 44.11%、SPL 24.56%。15300 在 V1/V2 的无提示模式表现更高，在 OVON 三组的无提示模式则低于 12200；不能称为全面升级。对 15300，YOLOE 在 V1/V2/Seen 的 SR 点估计为负收益，在 Synonyms/Unseen 为正收益。

20 个既定 SR 对比经 Holm 校正后均未达到 p<0.05（最小校正 p=0.8657）；多数差异较小，点估计排名不代表稳定胜出。配对 bootstrap 区间、精确 McNemar 检验、完整逐例结果见 `runs/hint_ablation_12200_15300/analysis.json` 与 `summary.json`。

结论与案例入口：`runs/hint_ablation_12200_15300/analysis.html`；四组保存动作重放视频：`diagnostics/index.html`。视频案例有意选择用于机制说明，不作为全量误检率。模型、权重和默认配置没有因本次分析改写。

可复用入口：`tools/compare_hint_ablations.py`。无提示仍预测 APOS/OPOS，只将外部 HINT 设为 NULL；15300 的运动适配分支仍生效，因此不能把“15300 无提示”当成“恢复 12200”。
