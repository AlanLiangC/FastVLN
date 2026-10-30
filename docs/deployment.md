# 加载、单步接口与视频

默认配置 `configs/pointcot.yaml` 固定加载 `checkpoints/fastvln_15300`；需要官方冻结基座 `checkpoints/qwen35_0p8b_base`。checkpoint 内保存 trainable adapters，不是另一份完整 Qwen 基座。加载器校验官方 revision、参数名 / 形状和 tensor-content 哈希。

```bash
source scripts/env.sh
# 交互服务，默认 15300
STREAMNAV_VIEWER_GPU=0 bash scripts/start_viewer.sh
# 独立评估输出
python -m streamnav.evaluation.runner checkpoint=checkpoints/fastvln_15300 run_dir=runs/manual_eval_15300
# 静态结果与视频
make viewer
# http://127.0.0.1:8765/
```

## 输入输出

每个 episode 调用 `start_episode(uid, instruction)`，随后输入当前 uint8 RGB、可选目标点及缓存。输出 APOS、OPOS、六动作 logits、四步动作计划、value 和下一状态；推理执行当前一步再观测。切换 episode 或权重时重置缓存。

HTTP `/sessions/{session_id}/step` 与 `/batch_step` 的 `object_hint` 可省略或为 null。一个点可写成 `{"frame_id":0,"xy_px":[300,120]}` 或 `{"frame_id":0,"token":649}`，二者不能同时给；坐标对应输入原始 RGB，token 为 48×27 网格。帧号从 0 开始，帧号 / episode 不匹配或坐标越界退化为 NULL。

`previous_action`（0–5）回传上一实际执行动作，外部 agent 覆盖动作时应提供；省略则按上一模型建议更新。RGB 缩略统计和动作反馈进入单个 HINT 位置的适配层。碰撞、真实位移和目标距离不进入该反馈输入。输出的质量概率为诊断量，不直接作为门控。

不需要历史图片、深度、语义图、地图或教师。独立 Python 调用可使用 `load_policy(config).eval()`，checkpoint 字段指定固定目录；旧 12200 使用 `checkpoints/reference_12200`，其保存配置会关闭新增适配层。

## 视频与口径

`runs/comparison_300_15300` 保存同 300 个任务的 15300 / 12200 / OVSegDT 指标、trace 和已录制视频。`runs/evaluation_15300/evaluation/update_0015300` 保存训练期三设置样例，每设置 24 例，不用于替代 300 例排名。

APOS 为绿色、OPOS 为品红色、实际 HINT 为橙色。NONE / TURN / STOP 等特殊 token 不伪造像素位置；没有品红点时查看 OPOS 是否预测 0。模拟提示来自干净目标实例，只代表理想提示能力。

策略延迟含 RGB 预处理、视觉编码、Qwen / KDA、点与动作生成及 GPU 同步，不含外部检测器、仿真 STEP、场景加载或视频。每卡独立推理，不需要八卡合做一个动作。工程计时可用 `tools/benchmark_hint_eval.py --checkpoint checkpoints/fastvln_15300 --batch 1 --warmup 8 --iterations 40`，使用保留真实 RGB 重复帧；该计时不等于导航性能或完整机器人端到端延迟。
