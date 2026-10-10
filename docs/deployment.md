# 部署与仿真器可视化

在 Bash 中 `source scripts/env.sh` 后使用。网页运行真实 Habitat；独立策略会话 API 不需要启动仿真环境。机器人、输入和动作定义见 [README](../README.md)。

## 选模型与启动

```bash
# 默认 best，不存在时使用 latest
STREAMNAV_VIEWER_GPU=7 bash scripts/start_viewer.sh

# 跟随活动 latest；与上面选择其一
STREAMNAV_VIEWER_GPU=7 bash scripts/start_viewer.sh runs/streamnav_active/ealm/checkpoints/latest

# 固定节点的最佳模型 update 2000；指标和视频来自活动实验
STREAMNAV_VIEWER_GPU=7 bash scripts/start_viewer.sh checkpoints/current_20261010/best \
  run_dir=runs/streamnav_active/ealm serving.reload_on_demo_reset=false
```

脚本默认 GPU 3，示例显式选择 GPU 7。八卡训练期间先检查额外推理显存；已有服务运行时不重复启动。服务监听 `http://127.0.0.1:8765`，远程用 `ssh -L 8765:127.0.0.1:8765 <服务器>` 转发。Headless EGL 不需要 X11。

选择 split 与 episode，点击“加载／重置”，再执行“模型单步”或“模型自动运行”。支持手动六动作、暂停、动作概率、距离、SPL 和状态内存查看。样本目录是评估使用的固定分层子集，成功／失败来自该 checkpoint 的已保存验证，不保证交互回放逐动作一致。

## 状态、权重与视频

`/health` 返回已加载 checkpoint、所跟随路径、更新可用状态和传感器参数；`/training` 返回活动 train/eval 指标、进程健康与 GPU 状态；`/docs` 提供 OpenAPI。

跟随 best/latest 时，在“加载／重置”读取最新完整发布的权重，自动导航过程中不切换模型。更换权重清空会话，API 客户端需重新 start。固定 checkpoint 或 `serving.reload_on_demo_reset=false` 用于固定模型。目标或 episode 改变也清空递归状态。

评估录像在 `runs/streamnav_active/ealm/evaluation/update_*/index.html`，也可经网页服务 `/videos/update_.../*.mp4` 访问。服务只挂载活动实验的 `evaluation`。没有运行 viewer 时也可直接打开播放页和 MP4。逐 case `.jsonl` 保留每步动作概率、距离、碰撞等诊断信息。

网页与正式评估使用 FP32 参数、BF16 autocast。单流与合批的浮点差异可能使长轨迹分叉，模型进度以固定协议的批量评估为准。

## 策略会话 API

```python
import base64
import requests

base = "http://127.0.0.1:8765"
session = requests.post(base + "/sessions/start", json={"instruction": "Find a chair."}).json()["session_id"]
with open("current_rgb.jpg", "rb") as image:
    frame = base64.b64encode(image.read()).decode()
result = requests.post(base + f"/sessions/{session}/step", json={"rgb_base64": frame}).json()
print(result["action_name"], result["probabilities"])
requests.post(base + f"/sessions/{session}/reset", json={"instruction": "Find a bed."})
requests.delete(base + f"/sessions/{session}")
```

输入 frame 仅为 RGB，新目标必须 reset。默认最多 16 个 session，闲置 900 秒后过期，访问过期会话返回 404。`/batch_step` 接收最多八个同分辨率 `{session_id, rgb_base64}`，同一批不能重复 session；模型访问串行化。`/demo/*` 使用真实 episode 的 object category。

可选自动网页验收使用 [verify_viewer.py](../tools/verify_viewer.py)，它会重置 episode 并执行一步模型动作。需要时安装项目内 Playwright 浏览器：

```bash
export PLAYWRIGHT_BROWSERS_PATH="$STREAMNAV_RUNTIME/browsers"
python -m playwright install chromium
python tools/verify_viewer.py
```

生成截图写入 `runtime/reports`。正常查看网页和训练录像不需要该验收依赖。
