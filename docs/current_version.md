# 当前版本：revision 5 · 2026-10-08

本节点固定当前实现、上游参照和 update 50 检查点，用于后续训练对照与回放。当前自主导航效果仍未达标：该检查点在三个 OVON 诊断 split 各 48 条 episode 上均为 0 成功、SPL=0。代码验证通过不能替代导航效果验证。

## 当前诊断实验

revision 5 在 update 76 保存停训。随后 revision 6 在 update 5、10、25 自主验证均为 0/144，已保存至 update 30 并停止。活动目录为 `runs/streamnav_kda_stable_20261008/ealm`（revision 8）：每步目标查询、KDA 输出 RMS normalization、较低主干 LR、零初始化 / 较低 LR 的 critic。该实验在 update 50 按全零验证规则停止，SR=0/144，7 条轨迹进入成功距离但没有主动 STOP。update 50 另固定于 `checkpoints/revision8_stop_audit_20261008`，作为短对照的只读初始化锚点，禁止原地修改。

STOP 加权和训练起点课程的短对照均为 0/144，未提升自主 SR。主实验已从 update 50 恢复八卡优化器 / sampler / EALM 状态，只启用等价视觉缓存（`ovsegdt_kda_cached`），上限为 200 updates，续训验证为 55、75、100、150、200。缓存与重算连续 8 次参数 SHA 完全一致，优化耗时观测约下降 6%，不是 SR 改善证据。当前检查通过单元/回归 63 项、GPU 6 项、真实 Habitat/训练集成 13 项（分批运行），Ruff、format、mypy 与 diff 检查通过；对照、失败记录和缓存开销见 [审计 JSON](../runtime/reports/stop_audit_20261008/audit.json)。固定 revision 5 节点与其历史证据保持原值；具体差异和实时验证见 [训练说明](training.md)。

续训 update 55 的自主 SR 仍为 0/144，13 条轨迹进入成功距离，没有贪心 STOP。数值与更新前重放检查正常；网页已加载该检查点。训练继续在上述预算和后台监控范围内运行，状态工具从实际进程读取当前更新。

## 节点定位

| 项目 | 固定值或路径 |
|---|---|
| 节点 ID | `revision5_20261008` |
| 训练 revision / update | `5` / `50`，160,000 transitions |
| 基线实验（已停止） | `runs/streamnav_ovsegdt_aligned_20261008/ealm`，最终保存 update 76 |
| 完整检查点 | [checkpoints/revision5_20261008](../checkpoints/revision5_20261008/) |
| 节点记录 | [baseline.json](../runtime/baselines/revision5_20261008/baseline.json) |
| 本次整理后的源码与文档 | [source.zip](../runtime/baselines/revision5_20261008/source.zip)，SHA256 记录在节点 JSON |
| 当时训练的实际源码 | [检查点内 source.zip](../checkpoints/revision5_20261008/source.zip) |
| OVSegDT commit | `646a3d53e7eae5879a4ce28d3a4dbaacca52d2b7` |
| frontier_exploration commit | `a8890d68cfa0d10254238abe9266a76856cb1f17` |

训练源码树 SHA256 为 `943edfd8c5c5edd3ec9dc49d3ea6e7411b3ac6297c9a14c529b534acf6eeb381`。节点不是新的 Git commit；现有工作区改动通过源码快照固定，基础 commit 和 dirty 状态记录在 manifest 中。固定节点形成时的整理只修改文档和辅助工具默认路径。此后的 2026-10-08 训练审查已修改诊断分支并重启独立实验，见上节；不能把固定节点源码 SHA 当作新实验的源码 SHA。

检查点完整保留模型、actor/critic、Adam、调度器、各 rank RNG / sampler / entropy EMA、tokenizer、配置与训练源码。通过已发布文件的硬链接保存，不额外复制大权重，并能独立于活动实验的自动轮转保留。禁止原地改写其中的文件。日常续训使用当前实验 latest；固定节点回放见 [部署说明](deployment.md)。

## 当前约束和上游差异

以用户最终确认的 RGB＋文本限制、六动作和机器人参数为准：480×270，HFOV 120°，相机高度 0.88 m / 初始俯仰 0°，机体高度 0.88 m / 半径 0.18 m。图像上下补齐到 480×288；导航网格按当前机体和上游爬升参数重建。

| 项目 | 当前实现与上游关系 |
|---|---|
| backbone | 用 KDA-converted Qwen3.5-0.8B；目标文本预填递归模型，替代上游视觉/目标编码路径 |
| 策略输入 | 保留 RGB＋文本限制；不采用上游 no_segm_loss 配置中的 mask、GPS/compass、上一动作输入 |
| semantic loss | 关闭；无语义预测监督 |
| 教师 | 直接执行 pinned ObjNavExplorer，适配 Habitat / episode 接口；特权信息不进入策略 |
| 核心训练配方 | 六动作、on-policy 标签监督、EALM、PPO/value、奖励、Adam 与 PIRLNav 解冻调度对照上游实际代码 |
| 运行时 | 同步八卡 DDP rollout/replay；没有复刻 VER 异步经验调度 |
| 数据读取 | 惰性按场景加载，保留对应抽样分布；随机流不保证逐 episode 相同 |
| 评估 | 用户机器人参数下的固定分层诊断子集，不能直接当成官方全量 benchmark |

旧版的教师接管、STOP 加权、距离 shaping、近目标课程和混合训练数据已退出默认配方。当前完整参数见 [训练说明](training.md)。KDA 转换没有额外蒸馏，不是无损替换；输入和 backbone 差异仍可能影响导航能力，不保证复现 OVSegDT 的成绩。

## 验证证据与当前限制

清理前完成的 revision 5 验证结果已保留在 [evidence](../runtime/baselines/revision5_20261008/evidence/)：

- 单元与回归 49 项、GPU 4 项、真实 Habitat / 训练集成 6 项通过；Ruff 与 mypy 通过。
- 差分测试直接执行本地上游 EALM、value clipping、PIRLNav 调度代码。
- 完整 T=100 双卡检查覆盖 update 1–3 及恢复后的 update 4；各次更新全部参数 SHA 跨 rank 一致，更新前重放概率误差为 0。
- 网页完成六动作控制和模型回放验收；相机与机体参数、重建网格也有检查记录。
- [节点验证指标](../runtime/baselines/revision5_20261008/evidence/checkpoint50_evaluation.json) 为三个 split 各 0/48；当前进程状态由状态工具实时读取，不能用旧截图代替。

本次整理后的链接、脚本、环境导入与服务检查单独记录在 [workspace_checks.json](../runtime/baselines/revision5_20261008/workspace_checks.json)，不将历史训练测试写成此次重新执行。后续修改训练代码需重新运行相应验证。

## 空间整理与保留规则

2026-10-08 清理 20 个停用实验、9 份过时文档及旧截图、历史诊断目录、关闭的仿真日志、无用临时文件、pip/uv/conda 下载缓存和可选浏览器资源。已删除旧实验大权重与视频；当前实验、初始化权重、数据、已安装环境和必要上游源码完整保留。

清理瞬间项目分配空间从约 **577.55 GiB 降至 59.41 GiB，释放 518.15 GiB**。训练仍在写入，后续占用会变化；精确时间、字节数和逐项删除记录见 [cleanup_report.json](../runtime/baselines/revision5_20261008/cleanup_report.json)。

历史文档、配置、指标、源码快照和诊断日志共 4,620 个文件，合并为约 31 MiB 的 [legacy_records.tar.gz](../runtime/baselines/revision5_20261008/legacy_records.tar.gz)，已检查压缩流并记录 [SHA256](../runtime/baselines/revision5_20261008/legacy_records.json)。该归档不含旧模型权重、视频或 `.config` 凭据；归档内路径和结论是历史记录，不作为当前运行指南。

日常使用以下边界：

- `docs/` 只保存当前说明；新诊断图片和临时报告写到 `runtime/reports/`。
- 当前实验保留最近三份检查点及 best；固定版本节点单独保留，不混入轮转规则。
- 数据、`.venv`、`runtime/habitat-env`、`runtime/vendor`、`third_party/OVSegDT` 和初始化权重是运行依赖。
- Triton、TorchInductor、NVIDIA、navmesh 等正在使用的缓存保留；删除进程仍引用的缓存或日志会破坏可追溯性或运行稳定性。
- 可再生成的下载缓存按需清理，先检查环境是否依赖其中的符号链接；新实验使用新目录，不覆盖当前基线。
