import base64
import io
import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, cast

import hydra
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from omegaconf import OmegaConf
from PIL import Image
from pydantic import BaseModel, Field

from streamnav.contracts.action import NavigationAction
from streamnav.data.schema import HabitatEpisodeSource
from streamnav.envs.habitat_client import HabitatClient
from streamnav.serving.batching import batch_steps
from streamnav.serving.session import SessionManager
from streamnav.training.checkpoint import load_policy
from streamnav.utils.process_health import current_health


class StartRequest(BaseModel):
    instruction: str = Field(min_length=1, max_length=2048)


class FrameRequest(BaseModel):
    rgb_base64: str = Field(max_length=6_000_000)


class BatchItem(FrameRequest):
    session_id: str


class DemoReset(BaseModel):
    episode_index: int = Field(default=0, ge=0, le=1000)
    split_index: int = Field(default=0, ge=0)


class DemoStep(BaseModel):
    action: int | None = Field(default=None, ge=0, le=5)


def decode_rgb(encoded):
    try:
        raw = base64.b64decode(encoded, validate=True)
        with Image.open(io.BytesIO(raw)) as image:
            if image.width > 2048 or image.height > 2048:
                raise ValueError("Image exceeds 2048 pixels per side")
            return torch.from_numpy(np.asarray(image.convert("RGB")).copy())
    except Exception as exc:
        raise HTTPException(400, "Invalid base64 RGB image") from exc


def encode_rgb(rgb):
    stream = io.BytesIO()
    Image.fromarray(rgb.numpy()).save(stream, format="JPEG", quality=90)
    return base64.b64encode(stream.getvalue()).decode()


def create_app(policy, config):
    serving = config["serving"]
    manager = SessionManager(policy, serving["max_sessions"], serving["session_ttl_s"])
    demo: dict[str, Any] = {"env": None, "session": None, "observation": None}
    catalogs = {}

    def catalog(split_index):
        if not 0 <= split_index < len(config["eval"]["manifests"]):
            raise ValueError("Unknown validation split")
        if split_index not in catalogs:
            source = HabitatEpisodeSource(config["eval"]["manifests"][split_index])
            catalogs[split_index] = list(
                source.evaluation_episodes(
                    serving.get("demo_episodes", 48), stratified=True, seed=config.get("seed", 17)
                )
            )
        return catalogs[split_index]

    @asynccontextmanager
    async def lifespan(app):
        yield
        if demo["env"] is not None:
            demo["env"].close()

    app = FastAPI(title="StreamNav", lifespan=lifespan)

    @app.exception_handler(KeyError)
    async def missing_session(request, exc):
        from fastapi.responses import JSONResponse

        return JSONResponse({"detail": "Unknown or expired session"}, status_code=404)

    @app.exception_handler(ValueError)
    async def invalid_request(request, exc):
        from fastapi.responses import JSONResponse

        return JSONResponse({"detail": str(exc)}, status_code=400)

    @app.get("/", response_class=HTMLResponse)
    def home():
        return (Path(__file__).with_name("viewer.html")).read_text()

    @app.get("/health")
    def health():
        latest = Path(config.get("checkpoint") or config["model"]["checkpoint"]).resolve()
        return {
            "status": "ready",
            "checkpoint": manager.policy.loaded_checkpoint,
            "latest_checkpoint": str(latest),
            "new_checkpoint_available": str(latest) != manager.policy.loaded_checkpoint,
            "splits": [Path(p).stem for p in config["eval"]["manifests"]],
            "demo_episode_count": serving.get("demo_episodes", 48),
            "demo_selection": "scene_stratified",
            "inference_mode": config.get("model", {}).get("inference_mode", "auto"),
            "parameter_dtype": str(
                getattr(
                    getattr(getattr(manager.policy, "backbone", None), "nav_token", None),
                    "dtype",
                    "unknown",
                )
            ),
            "compute_dtype": str(getattr(manager.policy, "inference_compute_dtype", "unknown")),
            "sensor": {
                k: config["habitat"][k]
                for k in (
                    "width",
                    "height",
                    "hfov",
                    "sensor_height",
                    "sensor_pitch_deg",
                    "agent_height",
                    "agent_radius",
                )
            },
        }

    @app.post("/sessions/start")
    def start(request: StartRequest):
        return {"session_id": manager.create(request.instruction)}

    @app.post("/sessions/{session_id}/step")
    def step(session_id: str, request: FrameRequest):
        return manager.step(session_id, decode_rgb(request.rgb_base64))

    @app.post("/sessions/{session_id}/reset")
    def reset(session_id: str, request: StartRequest):
        manager.reset(session_id, request.instruction)
        return {"reset": True}

    @app.delete("/sessions/{session_id}")
    def close(session_id: str):
        manager.close(session_id)
        return {"closed": True}

    @app.post("/batch_step")
    def batch(requests: list[BatchItem]):
        return batch_steps(
            manager,
            [(r.session_id, decode_rgb(r.rgb_base64)) for r in requests],
            serving["max_batch_size"],
        )

    @app.post("/demo/reset")
    def demo_reset(request: DemoReset):
        with manager.lock:
            episodes = catalog(request.split_index)
            if request.episode_index >= len(episodes):
                raise ValueError("Episode index out of range")
            episode = episodes[request.episode_index]
            latest = Path(config.get("checkpoint") or config["model"]["checkpoint"]).resolve()
            if (
                serving.get("reload_on_demo_reset", True)
                and str(latest) != manager.policy.loaded_checkpoint
            ):
                if not (latest / "actor_critic.safetensors").is_file():
                    raise ValueError("Latest checkpoint is not complete")
                # Stage the old weights on CPU to avoid two FP32 GPU copies
                # alongside the learner. In-flight requests share this lock.
                previous = manager.policy
                previous_device = getattr(getattr(previous, "backbone", None), "device", None)
                manager.sessions.clear()
                demo["session"] = None
                if previous_device is not None:
                    previous.to("cpu")
                    torch.cuda.empty_cache()
                try:
                    replacement = load_policy(
                        {**config, "checkpoint": str(latest)}, preserve_master_weights=True
                    )
                except Exception:
                    if previous_device is not None:
                        previous.to(previous_device)
                    raise
                manager.replace_policy(replacement)
                demo["session"] = None
                demo["observation"] = None
            if demo["env"] is None:
                demo["env"] = HabitatClient(config["habitat"], "viewer")
            if demo["session"] is not None:
                manager.close(demo["session"])
            observation = demo["env"].reset(episode)
            demo["session"] = manager.create(episode.goal_text)
            demo["observation"] = observation
            return {
                "rgb_base64": encode_rgb(observation["rgb"]),
                "instruction": episode.goal_text,
                "episode_id": episode.uid,
                "distance": observation["distance"],
                "done": False,
                "robot": observation["robot"],
                "checkpoint": manager.policy.loaded_checkpoint,
                "evaluation_index": request.episode_index,
            }

    @app.get("/demo/episodes")
    def demo_episodes(split_index: int = 0):
        with manager.lock:
            episodes = catalog(split_index)
            records = {}
            manifest = Path(manager.policy.loaded_checkpoint) / "manifest.json"
            if manifest.exists():
                update = json.loads(manifest.read_text())["update"]
                split = Path(config["eval"]["manifests"][split_index]).stem
                path = (
                    Path(config["run_dir"])
                    / "evaluation"
                    / f"update_{update:07d}"
                    / f"{split}_episodes.jsonl"
                )
                if path.exists():
                    records = {
                        r["episode_id"]: r for r in map(json.loads, path.read_text().splitlines())
                    }
            return {
                "selection": "same fixed scene-stratified subset as periodic validation",
                "checkpoint": manager.policy.loaded_checkpoint,
                "episodes": [
                    {
                        "index": i,
                        "episode_id": e.uid,
                        "goal": e.goal_text,
                        "success": records.get(e.uid, {}).get("success"),
                    }
                    for i, e in enumerate(episodes)
                ],
            }

    @app.post("/demo/step")
    def demo_step(request: DemoStep):
        with manager.lock:
            if demo["observation"] is None or demo["observation"].get("done"):
                raise ValueError("Reset an episode first")
            prediction = manager.step(demo["session"], demo["observation"]["rgb"])
            action = NavigationAction(
                prediction["action"] if request.action is None else request.action
            )
            observation = demo["env"].step(action)
            demo["observation"] = observation
            return {
                **prediction,
                "executed_action": action.name,
                "manual": request.action is not None,
                "rgb_base64": encode_rgb(observation["rgb"]),
                "done": observation["done"],
                "metrics": observation["metrics"],
            }

    @app.get("/training")
    def training():
        result = {}
        for name in ("train_metrics", "eval_metrics"):
            path = Path(config["run_dir"]) / f"{name}.jsonl"
            if path.exists():
                with open(path, "rb") as f:
                    f.seek(max(0, path.stat().st_size - 32768))
                    lines = f.read().splitlines()
                    if lines:
                        result[name] = json.loads(lines[-1])
        result["health_status"] = current_health(config["run_dir"])
        for name in ("gpu_status", "best_evaluation"):
            path = Path(config["run_dir"]) / f"{name}.json"
            if path.exists():
                result[name] = json.loads(path.read_text())
        return result

    evaluation_dir = Path(config["run_dir"]) / "evaluation"
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    app.mount("/videos", StaticFiles(directory=evaluation_dir, follow_symlink=True), name="videos")
    return app


@hydra.main(version_base=None, config_path="../../../configs", config_name="config")
def main(cfg):
    config = cast(dict[str, Any], OmegaConf.to_container(cfg, resolve=True))
    app = create_app(load_policy(config, preserve_master_weights=True), config)
    uvicorn.run(app, host=config["serving"]["host"], port=config["serving"]["port"])


if __name__ == "__main__":
    main()
