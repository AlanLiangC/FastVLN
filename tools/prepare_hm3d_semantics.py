"""Download official instance labels; credentials never enter logs or artifacts."""

import argparse
import hashlib
import io
import json
import shutil
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlparse

import requests


class RemoteTar(io.RawIOBase):
    """Read tar headers by byte range; skip meshes outside this project."""

    def __init__(self, url, credentials):
        self.session = requests.Session()
        response = self.session.get(
            url, auth=credentials, headers={"Range": "bytes=0-1023"}, timeout=60
        )
        if response.status_code != 206:
            raise RuntimeError("Official archive does not support authorized byte ranges")
        self.url = response.url
        self.auth = credentials if urlparse(self.url).hostname == "api.matterport.com" else None
        self.size = int(response.headers["Content-Range"].split("/")[-1])
        self.position = 0
        self.cache_offset, self.cache = 0, response.content

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        self.position = (
            offset if whence == 0 else self.position + offset if whence == 1 else self.size + offset
        )
        return self.position

    def read(self, size=-1):
        size = min(size if size >= 0 else self.size - self.position, self.size - self.position)
        if size <= 0:
            return b""
        start = self.position
        if not self.cache_offset <= start or start + size > self.cache_offset + len(self.cache):
            end = min(self.size - 1, start + max(size, 65536) - 1)
            response = self.session.get(
                self.url, auth=self.auth, headers={"Range": f"bytes={start}-{end}"}, timeout=90
            )
            if response.status_code != 206:
                raise RuntimeError("Semantic archive header range failed")
            self.cache_offset, self.cache = start, response.content
        self.position += size
        return self.cache[start - self.cache_offset : start - self.cache_offset + size]

    def fetch(self, member, destination):
        if destination.exists() and destination.stat().st_size == member.size:
            return str(destination)
        temporary = destination.with_suffix(destination.suffix + ".partial")
        session = requests.Session()
        with temporary.open("wb") as output:
            for offset in range(0, member.size, 4 << 20):
                start = member.offset_data + offset
                length = min(4 << 20, member.size - offset)
                for attempt in range(5):
                    try:
                        response = session.get(
                            self.url,
                            auth=self.auth,
                            headers={"Range": f"bytes={start}-{start + length - 1}"},
                            timeout=90,
                        )
                        if response.status_code != 206 or len(response.content) != length:
                            raise RuntimeError("Semantic mesh range size mismatch")
                        output.write(response.content)
                        break
                    except Exception:
                        if attempt == 4:
                            raise RuntimeError(
                                f"Unable to download semantic instance asset: {destination.name}"
                            ) from None
        temporary.replace(destination)
        return str(destination)


def prepare(split, credentials, root):
    archive_root = root.parent / "semantic_archives"
    archive_root.mkdir(exist_ok=True)
    targets = {p.name: p for p in (root / split).iterdir() if p.is_dir()}
    extracted = []
    for kind in ("configs", "annots"):
        if kind == "annots":
            remote = RemoteTar(
                f"https://api.matterport.com/resources/habitat/hm3d-{split}-semantic-annots-v0.2.tar",
                credentials,
            )
            with ThreadPoolExecutor(32) as workers:
                futures = []
                with tarfile.open(fileobj=remote, mode="r:") as tar:
                    for member in tar:
                        scene = next(
                            (part for part in Path(member.name).parts if part in targets), None
                        )
                        name = Path(member.name).name
                        if (
                            member.isfile()
                            and scene is not None
                            and name.endswith((".semantic.glb", ".semantic.txt"))
                        ):
                            futures.append(
                                workers.submit(remote.fetch, member, targets[scene] / name)
                            )
                for i, future in enumerate(futures):
                    extracted.append(future.result())
                    if i % 10 == 0:
                        print(
                            json.dumps(
                                {
                                    "split": split,
                                    "instance_assets_ready": i + 1,
                                    "total": len(futures),
                                }
                            ),
                            flush=True,
                        )
            continue
        archive = archive_root / f"{split}_{kind}.tar"
        if not archive.exists() or not tarfile.is_tarfile(archive):
            temporary = archive.with_suffix(".partial")
            url = f"https://api.matterport.com/resources/habitat/hm3d-{split}-semantic-{kind}-v0.2.tar"
            with requests.get(url, auth=credentials, stream=True, timeout=(30, 120)) as response:
                response.raise_for_status()
                size, started, reported = 0, time.monotonic(), 0.0
                with temporary.open("wb") as f:
                    for chunk in response.iter_content(4 << 20):
                        f.write(chunk)
                        size += len(chunk)
                        if time.monotonic() - reported > 30:
                            print(
                                json.dumps(
                                    {
                                        "split": split,
                                        "component": kind,
                                        "bytes": size,
                                        "elapsed": time.monotonic() - started,
                                    }
                                ),
                                flush=True,
                            )
                            reported = time.monotonic()
            if not tarfile.is_tarfile(temporary):
                raise RuntimeError(
                    "Semantic asset response is not a tar archive; check authorization"
                )
            temporary.replace(archive)
        with tarfile.open(archive) as tar:
            for member in tar:
                parts = Path(member.name).parts
                scene = next((part for part in parts if part in targets), None)
                if not member.isfile():
                    continue
                name = Path(member.name).name
                if scene is None and name.endswith(".scene_dataset_config.json"):
                    with tar.extractfile(member) as src, (root / name).open("wb") as dst:
                        shutil.copyfileobj(src, dst)
                    extracted.append(str(root / name))
                    continue
                if scene is None:
                    continue
                if not name.endswith((".semantic.glb", ".semantic.txt", ".json")):
                    continue
                destination = targets[scene] / name
                if not destination.exists() or destination.stat().st_size != member.size:
                    with tar.extractfile(member) as src, destination.open("wb") as dst:
                        shutil.copyfileobj(src, dst)
                extracted.append(str(destination))
    result = {
        "split": split,
        "files": extracted,
        "semantic_meshes": sum(p.endswith(".semantic.glb") for p in extracted),
    }
    print(json.dumps({k: v for k, v in result.items() if k != "files"}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--credentials", default=".config/matterport.json")
    parser.add_argument("--root", default="runtime/data/scene_datasets/hm3d")
    args = parser.parse_args()
    secrets = json.loads(Path(args.credentials).read_text())
    credentials = (secrets["token_id"], secrets["token_secret"])
    root = Path(args.root)
    with ThreadPoolExecutor(2) as pool:
        results = list(pool.map(lambda split: prepare(split, credentials, root), ("train", "val")))
    manifest = {"source": "official HM3D-Semantics v0.2", "splits": results}
    manifest["file_hashes"] = {
        p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
        for result in results
        for p in result["files"]
    }
    (root / "semantic_assets.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
