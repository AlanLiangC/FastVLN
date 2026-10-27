"""Download pinned OVSegDT navigation, goal encoder and segmentation assets."""

import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import requests
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[1]


def verified(path, asset):
    if not path.exists() or path.stat().st_size != asset["bytes"]:
        return False
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest() == asset["sha256"]


def download_url(path, asset):
    """Resolve the release once, then resume verified HTTP ranges from its CDN."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(
        asset["download_url"],
        headers={"Accept": "application/octet-stream"},
        stream=True,
        timeout=30,
    ) as response:
        response.raise_for_status()
        url = response.url
    chunk = 4 * 1024 * 1024
    parts = ROOT / "runtime/downloads" / (path.name + ".parts")
    parts.mkdir(parents=True, exist_ok=True)
    count = (asset["bytes"] + chunk - 1) // chunk

    def fetch(index):
        first, last = index * chunk, min(asset["bytes"], (index + 1) * chunk) - 1
        part = parts / f"{index:04d}"
        if part.exists() and part.stat().st_size == last - first + 1:
            return
        for attempt in range(3):
            try:
                response = requests.get(
                    url, headers={"Range": f"bytes={first}-{last}"}, timeout=(15, 60)
                )
                response.raise_for_status()
                expected = f"bytes {first}-{last}/{asset['bytes']}"
                if response.status_code != 206 or response.headers.get("Content-Range") != expected:
                    raise RuntimeError("Server did not return the requested file range")
                if len(response.content) != last - first + 1:
                    raise RuntimeError("Incomplete download range")
                temporary = part.with_suffix(".tmp")
                temporary.write_bytes(response.content)
                temporary.replace(part)
                return
            except (requests.RequestException, RuntimeError):
                if attempt == 2:
                    raise
                time.sleep(10 * (attempt + 1))

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(fetch, range(count)))
    temporary = path.with_suffix(path.suffix + ".download")
    with temporary.open("wb") as handle:
        for index in range(count):
            handle.write((parts / f"{index:04d}").read_bytes())
    if not verified(temporary, asset):
        raise RuntimeError("SHA-256 mismatch: " + str(temporary))
    temporary.replace(path)
    for index in range(count):
        (parts / f"{index:04d}").unlink()
    parts.rmdir()


def main():
    assets = ROOT / "checkpoints/ovsegdt"
    assets.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((ROOT / "configs/baselines/ovsegdt-assets.json").read_text())
    for asset in manifest["assets"]:
        path = assets / asset["path"]
        if not verified(path, asset):
            if "repo_id" in asset:
                hf_hub_download(
                    repo_id=asset["repo_id"],
                    filename=asset["filename"],
                    revision=asset["revision"],
                    local_dir=path.parent,
                    force_download=path.exists(),
                )
            else:
                download_url(path, asset)
        if not verified(path, asset):
            raise RuntimeError("SHA-256 mismatch: " + str(path))
        print("Verified " + str(path.relative_to(ROOT)), flush=True)
    manifest["verified_time_utc"] = datetime.now(timezone.utc).isoformat()
    (assets / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
