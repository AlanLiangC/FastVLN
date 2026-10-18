# 推理、延迟与视频

模型配置为 [pointcot.yaml](../configs/pointcot.yaml)，检查点格式为 `qwen35-pointcot-adapters-v1`。部署需要官方基座和 adapter；加载时校验基座 revision、参数名称、形状和权重校验和。当前固定节点位于 `checkpoints/versions/qwen35_pointcot_20261017/navigation`，活动模型位于 `runs/qwen35_pointcot_opos_hint/checkpoints/{latest,best}`。

## 单步接口

每个 episode 调用 `start_episode(uid, instruction)`，随后每步输入当前 uint8 RGB 和该 episode 的状态，输出 APOS、OPOS、六动作 logits、value、四步计划和下一状态。跨帧只保留 KDA，加载新权重或切换 episode 后重置缓存。规划执行当前一步，下一帧重新规划。

`forward_batch(..., hint_tokens=[k,...])` 每帧只接受一个可选提示：1–1296 表示图像 48×27 网格的坐标，0 表示独立可学习输入 NULL；NULL 不等于输出端表示不可见的 OPOS_0。模型不依赖历史图片、深度、语义图、地图、真值距离或教师。

HTTP `/sessions/{session_id}/step` 和 `/batch_step` 的 `object_hint` 可省略或为 null。一个点可写为 `{"frame_id":0,"xy_px":[300,120]}`，坐标对应原始 RGB；或写为 `{"frame_id":0,"token":649}`，两者只能选一个。帧号从 session 的 0 开始，帧号、可选 episode_id 不匹配或坐标越界时退化为 NULL。响应包含实际使用的 `object_hint_token` 与预测感知信息。

## 计时口径

RTX 4090、当前参数 / 计算配置下，已记录的单卡推理延迟如下。每张卡独立运行模型，不需要八卡共同完成一次动作。

| 评估 | 单卡活跃轨迹 | 按动作数加权的平均延迟 |
|---|---:|---:|
| update 5100，无提示 | 1–2 | 63.8 ms，约 15.7 Hz |
| update 5100，有提示 | 1–2 | 58.0 ms，约 17.3 Hz |
| update 5000，无提示 | 1–8 | 119.1 ms |
| update 5000，有提示 | 1–8 | 119.7 ms |

计时从当前 RGB 的策略调用开始，到 GPU 同步后的预测输出及少量诊断结束，包括图像预处理、视觉编码、文本 / KDA、APOS → OPOS → action 和规划分支；不含外部检测器、仿真 STEP、视频和场景加载。提示与无提示访问轨迹不同，实际 batch 和缓存热身不同，耗时差不能解释为提示本身提速。

`decision_hz = 1 / mean(step_latency)`；平均毫秒数可用 `1000 / decision_hz` 还原。`latency_p50_ms` 是中位数，`policy_decisions_per_second` 是所有 rank 的总吞吐；批次中的每条轨迹都要等待整轮决策完成。较大评估包含冷图捕获等长尾，P99 约 1.1 秒。上述结果不是严格独占 GPU 的 batch=1 部署基准。

单卡基准工具读取当前权重和保留的 RGB 样本，预热后记录均值和 P50/P95/P99，并连续更新 KDA。使用空闲 GPU 运行，避免混入训练争用：

```bash
source scripts/env.sh
CUDA_VISIBLE_DEVICES=0 python tools/benchmark_hint_eval.py \
  --checkpoint checkpoints/versions/qwen35_pointcot_20261017/navigation \
  --batch 1 --warmup 8 --iterations 40
```

这是重复真实帧的工程计时，不是导航性能评估。输入样本位于 `runtime/fixtures/navigation/rgb_inputs.pt`，输出位于 `runtime/reports/`。

## 查看视频与独立评估

```bash
make viewer
# http://127.0.0.1:8765/evaluation/
# 独立评估写入新目录，保留训练原始记录
python -m streamnav.evaluation.runner \
  checkpoint=checkpoints/versions/qwen35_pointcot_20261017/navigation \
  run_dir=runs/manual_node_eval
```

视频目录为 `RUN_DIR/evaluation/update_XXXXXXX/{no_hint,simulated_hint}/`，包含 MP4、逐步 trace、播放页和 split 汇总。APOS 荧光绿、预测 OPOS 洋红、外部提示橙色，均加白边并与当前帧对齐。HINT 方框是单点标记，尺寸不表示目标边界。NONE、LEFT/RIGHT、STOP 等特殊 token 不伪造像素点；四步计划在预测 STOP 处截断。

画面与 trace 的 GT-O、hint 来源用于诊断，不进入无提示输入。没有洋红点时先看 `opos`：0 表示实际预测 NONE，导航成功不要求输出位置点。`simulated_hint` 使用干净的实例内部点，只反映理想点提示，不代表真实检测器。可用 `tools/audit_eval_pointing.py --run-dir RUN_DIR --update UPDATE` 重放保存动作核验标记对齐。

`make viewer` 优先显示活动导航评估；可用 `VIEWER_RUN_DIR=runs/pointcot_grounding_refined` 查看 grounding。远程访问可转发 8765，静态播放不占用推理 GPU。交互推理使用 `scripts/start_viewer.sh CHECKPOINT`，GPU 由 `STREAMNAV_VIEWER_GPU` 设置，机器人参数与训练一致。
