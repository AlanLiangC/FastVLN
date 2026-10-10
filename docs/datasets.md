# 数据与可复现性

当前训练只使用 **HM3D-OVON train**，验证使用 `val_seen`、`val_seen_synonyms`、`val_unseen`。通用入口为 [训练数据配置](../configs/data/hm3d_ovon.yaml) 和 [验证配置](../configs/eval/hm3d_ovon.yaml)，当前有效配置见 [版本节点](current_version.md)。实际清单位于 `runtime/data/manifests/`。

| Split | 原始 episode 数 | 用途 |
|---|---:|---|
| train | 6,911,470 | 训练 |
| val_seen | 3,000 | 已见类别验证 |
| val_seen_synonyms | 3,000 | 同义词验证 |
| val_unseen | 3,000 | 未见类别验证 |

定期评估每 split 固定分层选择 48 条，原始起点、最多 500 步。诊断子集与完整 split 成绩分别报告。Manifest 记录场景和类别，不通过文件名猜测划分。

## 资产、episode 与校验

场景来自已准备的 `hm3d-{train,val,minival}-habitat-v0.2.tar`，解析器按实际 `.basis.glb` 文件建立映射。Episode 的 `hm3d_v0.2/` 前缀不要求额外下载场景。缺 mesh、目标视点或必需资产时直接报错，不构造替代任务。

统一 episode 包含 dataset、split、scene、episode ID、目标文本、起始位姿和目标元数据。`goals=[]` 时读取 `goals_by_category`，OVON children categories 也解析为合法目标视点。地图、位姿和目标视点供教师／奖励／评估使用，策略仍只接收 RGB＋文本。

可选感知分支使用现有三维目标 anchor 和训练时深度生成弱点位标签，仿真深度不进入策略，评估关闭标注传感器。当前场景未准备 semantic mesh；逐帧实例可见性并非现成真值。标签的置信度、遮挡屏蔽和验证流程见 [轻量感知监督](perception.md)。

Manifest 保存源文件 SHA256、episode 数、场景映射和类别集合。启动检查训练／验证场景及 unseen 词表泄漏；恢复核对训练清单本身的哈希。当前 `verify_data_hashes=false` 避免每次启动重算所有数据源文件，可显式开启完整源文件检查。训练惰性按场景加载、场景内无放回，分配规则见 [训练说明](training.md#数据教师与监督)。

Navmesh 使用机体高 0.88 m、半径 0.18 m、max climb 0.10 m、cell height 0.05 m 重建。`runtime/cache/navmesh` 的缓存签名包含上述参数，不复用不同机器人尺寸的网格。

## 可选数据与依赖

HM3D-v1/v2 场景和 episode 被部分真实 Habitat 集成测试使用，因此保留，不参与当前 OVON 训练。独立实验可选 `data=hm3d_v1 eval=hm3d_v1` 或 `data=hm3d_v2 eval=hm3d_v2`。混合数据能力保留但未启用；与 OVON unseen 相交的类别需过滤，否则泄漏检查拒绝启动。

已有资产可直接使用。下载与清单准备工具位于 [tools](../tools/)，episode 来源记录为 [HM3D-v1](https://dl.fbaipublicfiles.com/habitat/data/datasets/objectnav/hm3d/v1/objectnav_hm3d_v1.zip)、[HM3D-v2](https://dl.fbaipublicfiles.com/habitat/data/datasets/objectnav/hm3d/v2/objectnav_hm3d_v2.zip)、[HM3D-OVON](https://huggingface.co/datasets/nyokoyama/hm3d_ovon)。场景使用已授权资产。

`.config` 凭据仅供本地下载，不进入日志、快照或分享材料。数据、已安装环境、教师源码及活动缓存完整保留。
