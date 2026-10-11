# 推理与视频

新策略从 `configs/lightnav_kda.yaml` 加载官方底座，并由训练检查点加载 LoRA、KDA、动作 token 和 critic。`streamnav.training.checkpoint.load_policy(config, checkpoint)` 支持新模型。

每个 episode 调用 `start_episode(uid, instruction)` 创建独立状态；每一步 `forward_batch` 输入当前 RGB、instruction 和状态，取得六动作 logits、value、更新后的状态以及 APOS/OPOS 结果。选择 action 后向环境执行，再把返回的新 RGB 和保存的状态用于下一步。不同 episode 不能共享缓存；策略部署不需要教师、深度或历史图片列表。

评估视频可以直接播放，无需占用推理 GPU：

```bash
make viewer
# 浏览器访问 http://127.0.0.1:8765/evaluation/
```

远程服务器可通过 SSH 转发 8765 端口。每次评估的视频目录含 HTML 播放页、MP4 与动作 trace；点位使用醒目的颜色。通用旧版交互服务保留为兼容代码，当前默认展示已保存视频。
