# 数据与可复现性

当前 revision 5 **只使用 HM3D-OVON train 训练**，验证使用 `val_seen`、`val_seen_synonyms`、`val_unseen`。入口为 [训练数据配置](../configs/data/hm3d_ovon.yaml) 和 [验证配置](../configs/eval/hm3d_ovon.yaml)，实际清单在 `runtime/data/manifests/`。

| 默认 split | 原始 episode 数 | 用途 |
|---|---:|---|
| train | 6,911,470 | 训练 |
| val_seen | 3,000 | 已见类别验证 |
| val_seen_synonyms | 3,000 | 同义词验证 |
| val_unseen | 3,000 | 未见类别验证 |

定期训练验证从每个 split 固定分层选择 48 条 episode，原始起点、最多 500 步。完整 split 与诊断子集成绩须分别报告。场景与类别清单由 manifest 记录，不通过文件名猜测划分。

## 资产和 episode

场景来自用户已准备的 `hm3d-{train,val,minival}-habitat-v0.2.tar`，解析器按实际 `.basis.glb` 文件名建立显式映射。episode 的 `hm3d_v0.2/` 前缀不代表需要另下载一份场景。缺少 mesh、目标视点或必需资产时直接报错，不构造替代任务。

统一 episode 包含 dataset id、split、scene、episode id、goal text、起始位姿和目标元数据。`goals=[]` 时查 `goals_by_category`；OVON children categories 一并解析为合法目标视点。地图、位姿和目标视点用于教师和评估，策略仍只接收 RGB＋目标文本。

Manifest 保存源文件 SHA256、episode 数、场景映射和类别集合。启动训练校验数据哈希，并检查训练/验证场景及 unseen 词表泄漏。训练读取器按场景惰性加载、场景内无放回；分配和采样规则见 [训练说明](training.md)。

导航网格使用当前机体高 0.88 m、半径 0.18 m、max_climb 0.10 m、cell_height 0.05 m 重建。缓存位于 `runtime/cache/navmesh/`，签名包括这些参数，不能用旧机器人尺寸的网格替代。

## 保留的可选数据

HM3D-v1/v2 的已安装场景和 episode 仍被部分真实仿真集成测试使用，因此本次整理保留。它们不参与默认 OVON 训练。独立实验可选 `data=hm3d_v1 eval=hm3d_v1` 或 `data=hm3d_v2 eval=hm3d_v2`，并使用新的实验目录。

历史混合配置与读取能力也保留，但没有默认启用。混合数据额外排除与 OVON unseen 相交的 `plant`；过滤计数写入 manifest，不改写上游压缩包。含该类别的未过滤数据与 OVON unseen 组合会被泄漏检查拒绝。

已有数据无需重新下载。相关工具位于 [tools](../tools/)，公开 episode 的既有来源记录为 [HM3D-v1](https://dl.fbaipublicfiles.com/habitat/data/datasets/objectnav/hm3d/v1/objectnav_hm3d_v1.zip)、[HM3D-v2](https://dl.fbaipublicfiles.com/habitat/data/datasets/objectnav/hm3d/v2/objectnav_hm3d_v2.zip) 和 [HM3D-OVON](https://huggingface.co/datasets/nyokoyama/hm3d_ovon)。公开 episode 不需要 Matterport 凭据；场景使用现有已授权资产。

`.config` 中的凭据只供本地下载使用，不写入日志、源码快照或分享材料。数据资产、已安装环境、上游教师源码和活动编译缓存属于当前运行依赖；本次空间整理的删除边界见 [当前版本](current_version.md)。
