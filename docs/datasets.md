# 数据集与可复现性

官方 episode 来源：

- [Habitat ObjectNav HM3D-v1](https://dl.fbaipublicfiles.com/habitat/data/datasets/objectnav/hm3d/v1/objectnav_hm3d_v1.zip)
- [Habitat ObjectNav HM3D-v2](https://dl.fbaipublicfiles.com/habitat/data/datasets/objectnav/hm3d/v2/objectnav_hm3d_v2.zip)
- [HM3D-OVON](https://huggingface.co/datasets/nyokoyama/hm3d_ovon)

场景使用用户已有 `hm3d-{train,val,minival}-habitat-v0.2.tar`。路径解析按实际 `.basis.glb` 文件名建立显式映射，因此 v2 episode 中的 `hm3d_v0.2/` 前缀不会被误当成另一个必须下载的目录。每个 mesh 必须有对应 navmesh；不存在的资产会立即报错。

原始数据量：v1 train 3,971,566、val 2,000；v2 train 7,196,434、val 1,000；OVON train 6,911,470，val_seen/val_seen_synonyms/val_unseen 各 3,000。三种训练源分别涉及 80/145/145 个场景，验证分别涉及 20/36/36 个场景。

统一 episode 包含 dataset id、split、scene、episode id、goal text、起始位姿、目标元数据。`goals=[]` 的 episode 从 `goals_by_category` 查找；OVON 的 children categories 一并解析成合法目标视点。没有目标视点时不构造替代任务。

Manifest 保存源文件 SHA256、episode 数量、场景映射和类别集合。混合数据额外排除与 OVON unseen 相交的 `plant`，不改写上游压缩包。`excluded_categories` 和 `excluded_episodes` 显式记录过滤规则；sampler 使用过滤后的计数加权选场景。`check_leakage()` 对所有训练源的并集检查验证场景和 unseen 词表。

独立 HM3D 实验可选未过滤的 `data=hm3d_v1`/`hm3d_v2`，但应搭配各自的 `eval=hm3d_v1`/`hm3d_v2`。将含 plant 的未过滤训练集与 OVON unseen 组合会被拒绝。

本项目不需要 Matterport 凭据下载公开 episode。现有已授权场景直接使用。`.config/matterport.json` 不被下载工具或日志打印，也不复制进任何产物；重新获取未准备的受许可场景应遵循 [HM3D 官方下载说明](https://github.com/facebookresearch/habitat-sim/blob/main/DATASETS.md)。

相机和动作参数写在 `configs/config.yaml`。当前快速配置为方形 RGB、无语义 sensor、无深度输入、四动作。它与官方标准协议存在分辨率/动作数量差异，报告成绩时必须连同 resolved config 一起发布。
