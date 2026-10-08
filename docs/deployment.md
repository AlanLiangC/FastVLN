# 部署与仿真器可视化

使用前在 Bash 中 `source scripts/env.sh`。策略推理与 Habitat 后端分离；网页包含真实 Habitat 仿真，单独使用策略会话 API 时不启动仿真器。当前机器人、RGB＋文本限制和六动作定义见 [README](../README.md)。

## 启动和选模型

```bash
# 默认 best，不存在时使用 latest；已有服务运行时无需重复启动
STREAMNAV_VIEWER_GPU=7 bash scripts/start_viewer.sh
# 显式跟随当前实验的 latest
STREAMNAV_VIEWER_GPU=7 bash scripts/start_viewer.sh runs/streamnav_active/ealm/checkpoints/latest
# 固定回放版本节点；run_dir 指向指标和视频所在实验
STREAMNAV_VIEWER_GPU=7 bash scripts/start_viewer.sh checkpoints/revision5_20261008 run_dir=runs/streamnav_active/ealm serving.reload_on_demo_reset=false
```

上述三条是不同启动方式，每次选择一条。脚本默认 GPU 3，示例显式选择 GPU 7；八卡训练时需留出额外推理模型的显存。停止 viewer 可释放其显存，训练与已保存视频保留。

浏览器打开 http://127.0.0.1:8765。远程用 `ssh -L 8765:127.0.0.1:8765 <服务器>` 转发。后端为 headless EGL，无需 X11；界面显示实际传感器和机体参数。服务默认只监听本机，不提供公网认证或多租户隔离。

选择验证 split 与 episode，点击“加载 / 重置”，再选择“模型单步”或“模型自动运行”。可暂停或手动执行 STOP、前进、左右转、上下看；手动轨迹有标记。图像是 480×270 / HFOV 120° 的真实仿真画面，同时显示目标、动作概率、距离、SPL 和状态内存。

样本列表使用与定期验证相同的固定分层子集。成功 / 失败标签来自所加载检查点的已保存验证记录，不保证交互回放得到相同结果。固定节点 update 50 的自主验证为 0/144，不属于成功模型展示。

## 更新、状态与视频

`/health` 显示实际已加载 checkpoint、所跟随路径、是否出现新模型和传感器参数。`/training` 返回当前实验的最近 train/eval 指标、进程健康与 GPU 状态；`/docs` 提供 OpenAPI。

跟随 best/latest 时，“加载 / 重置”会读取该路径最新完整发布的模型；模型自动运行中不切换权重。更换模型会清空所有会话，API 客户端需重新 start。具体 update 目录或 `serving.reload_on_demo_reset=false` 用于固定模型。目标或 episode 改变也会清空递归状态。

验证视频保存在 `runs/streamnav_active/ealm/evaluation/update_*/`，可由 `/videos/update_.../*.mp4` 访问。服务仅挂载本次实验的 `evaluation/`，不把整个 runtime 作为文件服务目录。

网页与独立评估使用 FP32 参数、BF16 autocast。合批与单流浮点差异可能使长轨迹分叉，训练进展应按固定协议的批量验证判断。

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

每个 frame 只含 RGB，不接受 GPS、目标坐标或教师动作。新目标必须 reset。默认最多 16 个 session，闲置 900 秒后过期，访问过期会话返回 404。`/batch_step` 接收同分辨率的多个 `{session_id, rgb_base64}` 项，最多 8 个，同一批 session id 不得重复；模型访问串行化。

`/demo/*` 使用真实 episode 自带的 object category，不能将不存在于场景中的任意目标文字伪装成有效导航任务。

## 可选网页验收

[verify_viewer.py](../tools/verify_viewer.py) 会重置当前网页 episode 并执行模型单步，仅在需要验收时运行。截图默认存放 `runtime/reports/viewer.png`，不再把生成物写进 docs。Playwright 浏览器下载缓存已在空间整理中清除；需要此可选检查时先安装项目内浏览器：

```bash
export PLAYWRIGHT_BROWSERS_PATH="$STREAMNAV_RUNTIME/browsers"
python -m playwright install chromium
python tools/verify_viewer.py
```

此依赖只用于自动网页验收，日常打开浏览器查看模型不需要安装。
