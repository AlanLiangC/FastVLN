# 部署与可视化

先在 Bash 中 `source scripts/env.sh`。服务默认本机 `127.0.0.1:8765`，读取一个已经完成训练保存的 checkpoint。Qwen/KDA inference 与 Habitat backend 仍分离；单独使用 policy session API 时不启动仿真器。

```bash
bash scripts/start_viewer.sh runs/streamnav_active/ealm/checkpoints/best
```

画面使用 480×270 / HFOV 120°，相机 0.88 m / 0°，机体 0.88 m 高、0.18 m 半径；页面显示仿真器实际生效的参数。样本列表与定期验证一致，覆盖 36 个场景，并标明已加载 checkpoint 的成功 / 失败；不使用过去只覆盖开头场景的顺序索引。浏览器首页提供真实 Habitat episode 切换、模型单步/自动运行、暂停、手动动作和策略概率。`/docs` 提供 OpenAPI。`/demo/*` 使用固定 episode 的真实 object category，不允许随意把未存在的目标文本拼成伪任务。

Agent 调用示例：

```python
import base64
import requests

base = "http://127.0.0.1:8765"
session = requests.post(base + "/sessions/start", json={"instruction": "Find a chair."}).json()["session_id"]
frame = base64.b64encode(open("current_rgb.jpg", "rb").read()).decode()
result = requests.post(base + f"/sessions/{session}/step", json={"rgb_base64": frame}).json()
print(result["action_name"], result["probabilities"])
requests.post(base + f"/sessions/{session}/reset", json={"instruction": "Find a bed."})
requests.delete(base + f"/sessions/{session}")
```

每个 frame 只包含 RGB，模型不接受 GPS、目标坐标或 oracle action。新目标必须 reset。默认最多 16 个 session，闲置 900 秒会过期；过期后返回 404。`/batch_step` 接收同分辨率的多个 `{session_id, rgb_base64}` 项，最多 8 个且 session id 不得重复。并发模型访问串行化，batch 是显式 API 批处理。

默认只对可信本地客户端开放，不提供公网认证/多租户服务。远程查看使用 SSH 端口转发。`/videos/update_.../*.mp4` 可访问该 run 的验证视频。`/training` 返回最近 train/eval 指标及 `health_status` / `gpu_status`。`/health` 显示当前加载 checkpoint、latest 路径和是否有新版本。点击“加载 / 重置”时自动读取完整发布的 best（也可显式指定 latest）；为避免不同权重共用递归状态，更换模型会清空所有 session，API 客户端需重新 start。模型自动运行中不切换权重。需要固定版本时传入具体 update 目录，或设置 `serving.reload_on_demo_reset=false`。

后端不需要 X11，使用 Habitat headless EGL。GPU 3 同时承担仿真渲染和可视化推理；如果它还运行主实验训练，交互延迟可能增加。可通过 `STREAMNAV_VIEWER_GPU` 指定其他卡；可停止 viewer 释放模型显存，训练与已保存视频仍保留。

不要对 `runtime/` 启用全目录 HTTP 文件服务：它包含环境和实验产物。内置服务仅挂载本次实验的 `evaluation/` 视频目录。
