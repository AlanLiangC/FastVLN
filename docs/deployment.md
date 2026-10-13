# 推理与视频

当前配置为 [pointcot.yaml](../configs/pointcot.yaml)。`load_policy` 加载官方 0.8B 基座与当前格式的 adapter；保存格式为 `qwen35-pointcot-adapters-v1`，不同基座 revision 或参数配方会拒绝加载。

每个 episode 调用 `start_episode(uid, instruction)`。每步输入当前 uint8 RGB 和该 episode 的状态，输出 APOS、OPOS、六动作 logits、value 与下一状态。`forward_batch` 在评估中生成确定性点和动作；`step` 可用于采样或实际 token 的 teacher forcing。部署不需要深度、教师或历史图片。

```bash
source scripts/env.sh
python -m streamnav.evaluation.runner checkpoint=runs/qwen35_pointcot_active/checkpoints/best
make viewer
# http://127.0.0.1:8765/evaluation/
```

视频位于 `RUN_DIR/evaluation/update_XXXXXXX/`，含播放页、MP4、逐步 JSON trace 和 split 汇总。点位绘制在模型决策的当前帧上，APOS 荧光绿、OPOS 洋红，均加白边。特殊 token 使用文字显示，不伪造像素点。图像不整体调暗。

`make viewer` 优先显示正式导航评估；尚无导航评估时显示 grounding 可视化，再无结果时显示 grounding 运行目录。可用 `make viewer VIEWER_RUN_DIR=runs/自己的实验目录` 指定目录；grounding 的浏览入口为 `/grounding_visualizations/`，导航为 `/evaluation/`。

远程访问可通过 SSH 转发 8765。静态视频查看不占用推理 GPU。交互推理可用 `scripts/start_viewer.sh CHECKPOINT`，其 Habitat 机器人参数与训练一致，默认 GPU 可用 `STREAMNAV_VIEWER_GPU` 设置。

加载新权重或切换 episode 后必须重置缓存。评估从原始起点开始，使用真实仿真动作和允许沿障碍滑动的配置，不能把近目标训练成功率当作普通搜索成功率。
