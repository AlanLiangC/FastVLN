# OVSegDT 安装与方法对照

**各方法使用自身原本设置，只对齐任务**：相同目标、episode、起始位置 / 朝向、500 步上限、0.25 m 内 STOP 的成功判定，以及 SR / SPL。相机、机器人、滑动、记忆和动作选择各自保留，不把 OVSegDT 改成 FastVLN 的设置。

| 项目 | OVSegDT | FastVLN |
|---|---|---|
| 相机宽×高、HFOV | 360×640、42° | 480×270、120° |
| 相机高 | 1.31 m | 0.88 m |
| 机器人高 / 半径 | 1.41 / 0.17 m | 0.88 / 0.18 m |
| 沿障碍滑动 | 禁用 | 启用 |
| 场景仿真 | 原仓库指定 Habitat-Sim 0.2.3 | 当前训练的 0.3.3 |
| 动作选择 | 原 `policy.act(deterministic=False)` 采样 | 当前默认 greedy |
| 历史状态 | 原 100 步 KV cache、上一动作 / mask 状态 | 当前 KDA 状态 |
| 感知输入 | YOLOE 目标 mask | 无提示；另列外接 YOLOE 单点 |

OVSegDT 参数取自原仓库 [评估配置](../third_party/OVSegDT/config/experiments/transformer_rl_segm_loss-validation.yaml)，缺省动作尺度取检查点保存值：前进 0.25 m，转向 / 抬头 / 低头 30°。其独立模型进程按原 trainer 开启 TF32，保留原 SigLIP、ResNet9、Transformer、动作头与采样。GT 点只保留为 FastVLN 可选诊断，不进入默认主对照。

## 权重与独立环境

| 内容 | 位置 |
|---|---|
| 官方导航权重 | `checkpoints/ovsegdt/ckpt.5.pth` |
| YOLOE 与文本编码器 | `checkpoints/ovsegdt/yoloe-v8l-seg.pt`、`mobileclip_blt.ts` |
| SigLIP 目标编码器 / tokenizer | `checkpoints/ovsegdt/siglip_text/` |
| 类别文本缓存 | `checkpoints/ovsegdt/siglip.pkl`，379 类、768 维 |
| 固定来源、revision、SHA-256 | [资产清单](../configs/baselines/ovsegdt-assets.json) |
| 模型解释器 | `runtime/ovsegdt-env/bin/python` |
| 原版场景仿真解释器 | `runtime/ovsegdt-sim-env/bin/python` |

导航权重由原仓库 README 指向的 [wingrune/OVSegDT](https://huggingface.co/wingrune/OVSegDT) 下载，revision 为 `c67bff3caa7a2b5b751d7545dbedeffc5b4efe1a`，516,824,617 字节，SHA-256 为 `9615be13e1a8a873e4997faa7b5dcf38eb5f5e4080fb8e99c12e0a4b0ed631cb`。含 278 个状态张量、129,151,438 个元素，已有完整 SigLIP 视觉编码器，与 FastVLN 的 Qwen / adapter 无关。`extra_state.step=15,011,232` 是该阶段计数，原配置另存为 `checkpoints/ovsegdt/resolved_config.json`。

模型环境使用 Python 3.9、PyTorch 2.2.1、Transformers 4.31.0、timm 1.0.15、Ultralytics 8.3.161，见 [模型依赖](../configs/baselines/ovsegdt-requirements.txt)。场景仿真单独使用 Python 3.8.20 / Habitat-Sim 0.2.3，见 [仿真环境锁定](../configs/baselines/ovsegdt-simulator-explicit.txt)。进程通过本地 IPC 通信，不修改 FastVLN 环境或训练进程。原仓库未锁定所有模型依赖，因此记录本机版本，不声称逐比特复现作者机器。

原作者的 `modeling_llama.py` 仅复制到独立模型环境。[导入补丁](../scripts/patches/ovsegdt_inference.patch) 只延后训练分割 loss 的 Detectron2 导入；导航前向使用原代码。通过 `torch.nn.Module.load_state_dict(..., strict=True)` 校验完整参数，避免原策略内部 `strict=False` 漏报。

```bash
CUDA_VISIBLE_DEVICES=1 bash scripts/setup_ovsegdt.sh
bash scripts/setup_ovsegdt_simulator.sh
source scripts/ovsegdt_env.sh
CUDA_VISIBLE_DEVICES=1 "$OVSEGDT_PYTHON" tools/validate_ovsegdt_install.py --detector
```

安装检查覆盖严格加载、原 `policy.act()` 一致性、105 步缓存窗口、episode 重置、空 / 非空 mask 和外接 / 内部分割通路等价性。闭环短测另外验证了原版 0.2.3 与 FastVLN 自身仿真能分别运行。

## 原策略与检测逻辑

SigLIP 文本缓存按原脚本重建：直接编码类别字符串，64 token，不加导航 prompt，不额外归一化。原仓库 `data/text_embeddings/siglip.pkl` 链接到统一资产目录。

YOLOE 使用作者提供的各 split 完整校准词表、逐类别阈值、原 224×224 Resize 和 `conf=0.01`。词表会合并冗余词，故查询词按原实现与 SigLIP 类别向量做 **L1 最近邻匹配**，再选对应类别和阈值；例如 `flower vase` 匹配 `vase`，不能要求字符串完全相同。seen 使用作者的 `cats_val_seen.json` 补齐原源码缺失的 `yolo_val_seen` 初始化分支。

同一帧的 YOLOE mask 送入原策略张量通路。内部 `segmentation_source="gt"` 只表示直接接收 mask，实际来源是 YOLOE，无 GT 标签进入 OVSegDT。每个 episode 重置 KV cache、上一动作及 mask 状态；每步调用原 `act()`，实际动作写入下一状态，不加动作覆盖或防抖后处理。

[原版仿真后端](../services/habitat_server/ovsegdt_eval_env.py) 按 OVONSim 重建 navmesh，使用 0.2.3 默认几何，不复用 0.3.3 缓存。旧 API 的 `include_static_objects=False` 通过重建函数参数传入；旧 API 的路径终点用于兼容环境辅助字段。测距始终是原生多目标 geodesic 到官方 viewpoints，目标集合包含 episode 子类别，成功及 SPL 公式与原指标口径一致。

## 当前对照入口

[当前播放页](../runs/native_method_comparison_20261021/index.html)。默认从三个当前验证 manifest 按 seed 2025 分层选取每 split 12 条，共 36 个固定 episode；完整内容写入 `cases.json`，不依赖历史实验目录。

| 设置 | 运行方式 | 次数 |
|---|---|---|
| ovsegdt_native_sampled | 原相机 / 机器人 / 仿真、YOLOE mask、采样 | 100 / 101 / 102 三个 seed |
| fastvln_no_hint | FastVLN 自身配置、NULL、greedy | 一次 |
| fastvln_yoloe_point | FastVLN 自身配置、外接 YOLOE 单点、greedy | 一次 |

默认共 180 条轨迹。每个采样 seed 与 episode UID 确定该例 RNG seed，写入结果以便断点重跑；不保证与作者多环境运行的随机数序列相同。SR / SPL 按 seed 汇总，另报告 seed 间标准差。

FastVLN 的单点优先取 mask 最大连通区域，选择离边界较深、且网格中心仍在 mask 内的 48×27 token；没有可支持网格点则输入 NULL，不用 GT 修正检测器。默认仅保存每 split 第一个案例的视频，其余保存 trace，以控制空间。

```bash
source scripts/env.sh
CUDA_VISIBLE_DEVICES=1 python tools/compare_navigation_baselines.py \
  --output runs/native_method_comparison_20261021
```

脚本固定启动时的 FastVLN best adapter、manifest 和配置，后续训练切换 best 不影响对照。只对齐任务，不要求不同方法 RGB 相同；同一方法重复实验核验起点一致。已完成项可断点续测，协议不匹配则拒绝混合结果。

`index.html` 是播放页，`summary.json` 包含方法汇总、逐 seed 汇总和逐例指标，`progress.json` 记录进度或异常。结果包含实际机器人、仿真版本、初始距离、动作、碰撞、误停及最长连续左右交替。青色是 YOLOE mask，橙色是输入点，绿 / 洋红是 FastVLN APOS / OPOS。与训练共享 GPU，时间不作为独占延迟结果。

共享相机或将 OVSegDT 改为 greedy 的旧诊断不进入本次主对照。当前是固定验证子集，完整论文基准结论仍需全量验证。

## 论文指标与复现状态

当前属于使用原策略和原设置的适配测试，**尚未验证复现论文指标**。[论文表 1](https://arxiv.org/html/2508.11479v3) 使用经过 predicted-mask 微调和置信度校准的 YOLOE，报告 seen / synonyms / unseen 的 SR 为 43.6% / 40.1% / 44.7%，SPL 为 20.1% / 17.9% / 20.6%。不能用“论文使用 GT 分割”解释与当前测试的差距。

原先每 split 12 例由与 FastVLN 定期验证相同的选择器和 seed 产生，与 checkpoint 选择使用的验证集重合；三个 OVSegDT seed 只是重复这些案例，不能视为三倍的独立样本。这个集合适合回放诊断，不能独立支撑总体方法优劣结论。

2026-10-21 已启动[独立扩样验证](../runs/ovsegdt_replication_20261021/index.html)：每 split 从 episode 中均匀无放回抽取 100 例，排除定期验证固定的前 48 例，覆盖各 split 的全部 36 个场景。OVSegDT 仍使用三个 seed，共 900 条轨迹；`selection.json` 记录采样范围和 seed。这仍是小规模验证，完整基准每 split 有 3000 个 episode。

已补充两个真实案例各 150 帧的回放核查，含非空检测 mask 及超过 100 步缓存窗口：外接 mask 通路与作者 inline YOLOE 通路的 mask、动作分布、价值和 KV cache 全部相同。核查记录为 `runtime/reports/ovsegdt_adapter_replay_audit_20261021.json`。已完成案例的初始 geodesic 另与原数据元信息核对。这些检查未覆盖完整原生 Habitat-Lab evaluator，不能代替论文复现结果。
