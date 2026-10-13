# 数据与下载

导航使用 HM3D-OVON train；评估使用 val_seen、val_seen_synonyms、val_unseen。清单位于 `runtime/data/manifests/`，实际配置见 [pointcot.yaml](../configs/pointcot.yaml)。train 共 6,911,470 episodes，三个验证 split 各 3,000。常规评估每 split 固定分层选择 48 条。

场景解析实际 `.basis.glb`，目标读取 `goals_by_category` 及 children categories 的视点。教师、奖励和成功判断可以读目标元数据，学生不能读。场景、episode、已安装环境和授权凭据均保留。凭据不进入日志或源码快照。

## Grounding 数据

当前清单位于 `runtime/data/grounding/public_v2/{train,val}.jsonl`，准备报告为同目录 `preparation.json`，记录来源数量和 manifest SHA256。以下是实际转换后的查询数，图像数在不同来源间存在重叠，不能相加。

| 来源 | 训练查询 | 验证查询 | 监督 |
|---|---:|---:|---|
| COCO 实例 | 186,731 | 123,333 | 内部标注点及真实实例有效网格，同类别实例为等价目标 |
| RefCOCO / RefCOCO+ / RefCOCOg | 253,703 | 47,438 | 完整指代表达绑定对应实例，使用 COCO 原始实例掩码 |
| ScanNet | 58,305 | 4,185 | 过滤后的真实 RGB、可见实例点及有效网格 |
| ImageNet LOC | 517,370 | 26,988 | 1,000 类的弱区域监督，不伪造框中心真值 |
| PixMo 已下载自然图像 | 9,803 | 499 | 人工点和明确缺失，训练包含 5,699 条缺失查询 |
| 合计 | **1,025,912** | **202,443** | 约 50.9 万条点查询、51.7 万条弱区域训练查询 |

COCO/RefCOCO 共享本地 `LaSco/train2014`、`LaSco/val2014` 原图，未复制完整图像集。ImageNet 直接读 `ILSVRC/Data/CLS-LOC/train` 和顶层 `LOC_*` 标注。公共目录从不写入或清理。

```bash
source scripts/env.sh
python tools/download_referring_annotations.py
python tools/prepare_public_grounding.py --workers 24
python tools/prepare_grounding_support.py --workers 12
# 已分来源准备完毕时，仅合并清单
python tools/prepare_public_grounding.py --merge-only
```

RefCOCO 下载镜像只包含标注，来源与固定 revision 见 `runtime/data/grounding/referring/sources.json`。实例 ID 与官方 COCO 标注关联后制点。数据组织参考 [官方 REFER](https://github.com/lichengunc/refer)。三个 referring 数据集的所有非训练图像联合保留，所有测试图像也从本次验证侧排除；额外的图像级内部 holdout 同时作用于 COCO 和 referring 查询。这是联合内部验证，不能冒充单个官方 benchmark 成绩。

ScanNet 本地 `scans/` 含 729 次扫描、347 个物理空间。按同帧 `.sens` RGB 和 `*_2d-instance-filt.zip` 的可见实例制点，名称来自 `.aggregation.json`；[官方说明](https://github.com/ScanNet/ScanNet) 给出了格式。每 80 帧检查候选，结合位姿变化、图像清晰度和实例大小筛选，每次扫描最多 64 帧。实际训练保留 12,646 张 RGB，验证 836 张；按 `sceneXXXX` 物理空间分组，同空间多次扫描不得跨集合。

PixMo 完整原始标注有 2,376,222 条查询、227,963 个图像哈希，仍保留于 `runtime/data/grounding/pixmo-points/data/`；本轮只复用 **1,948 张已下载并校验的自然图像**，没有声称所有 URL 图片已下载。现在保留这些图上的所有合格查询，再通过训练采样控制缺失比例，不再按每图前 8 条截断。百分比坐标采用 [Ai2 loader 的 point_scale=100](https://github.com/allenai/molmo/blob/main/olmo/data/pixmo_datasets.py)。CoSyn 文档/界面数据与旧小样本清单已移除。

## Prompt、监督与采样

Grounding 与导航调用相同的 `frame_features_batch` 聊天模板和系统提示。类别指令使用 `Find a chair.`，指代表达使用 `Find the chair on the left.`；完整目标描述保留在用户指令中。输入仅当前 RGB 与目标文本，每张静态图从空缓存开始。

JSONL 字段包括 `image_id`、`image`、`instruction`、`source`、`apos_targets`、`opos_targets`、`supervision`。`opos_support` 是 48×27 实例归属 bitmap，按 little-endian bit order 打包后 base64 保存，每个 bitmap 为 162 字节；在采样时解码，避免常驻完整候选 token 列表。约 498,739 条训练和 174,956 条验证查询包含此字段。原始 canonical 点、来源顺序、图像和 split 与 public_v1 完全一致。空列表表示未知、不计 loss；`[0]` 只表示明确无可见目标。多点使用等价目标的联合概率。区域记录使用归一化 `opos_boxes`，采样时才展开框内网格集合，避免把几十万张图的全部候选网格常驻内存。区域 loss 权重为 0.25，区域命中与真实点命中分开评估。

每个样本先以 2.5% 概率从明确缺失记录采样；其余正样本按 COCO:referring:ScanNet:ImageNet:PixMo = 38:48:11:2:1 选择来源，减少小数据源和重复缺失查询的过采样。训练随机状态、各来源顺序和游标随检查点保存。未标注某类别不能推出该对象不存在；外部图片不提供 APOS 或导航动作标签。掩码只用于离线生成点和实例有效网格，模型不增加分割预测分支。

准备清单应生成在新目录或停训后重新生成，不能覆写正在运行的 manifest。恢复会校验数据哈希及采样配方。公共原始数据审计与 GT 样例保存在 `runtime/reports/public_grounding_audit.json` 和 `public_grounding_samples/`；当前转换数量以 `public_v2/preparation.json` 为准。

ScanNet++ 本地有 96 场景、34,376 张 DSLR 图像，尚未完成可靠的可见实例投影，本轮未接入。LaSco/CIRR/FashionIQ 的检索配对没有空间坐标，不能直接当点真值；EPIC 自动检测结果不能与人工实例标注同等处理。

## 场景监督与防泄漏

当前未准备实例语义资产，所以导航服务默认只生成可靠几何 APOS，屏蔽弱目标中心 OPOS。深度用于训练遮挡检查，不作为学生输入。近目标采样只改训练 reset，不修改原始清单，也不会写入验证 episode。

导航与 grounding 的 manifest 哈希进入配置/检查点。恢复检查数据和配方，防止同一路径被替换后静默继续训练。机器人 navmesh 以高 0.88 m、半径 0.18 m、max climb 0.10 m、cell height 0.05 m 重建和缓存。

Qwen 权重下载入口为 [download_qwen_weights.py](../tools/download_qwen_weights.py)，固定官方 revision `2fc06364715b967f1860aea9cf38778875588b17`，存放 `checkpoints/qwen35_0p8b_base`。模型不需要离线注意力转换。
