"""Download public episode archives and unpack supplied licensed HM3D scene archives.

No credential is required for these public episode URLs. Matterport meshes come
from the user's existing licensed archives; credentials are never logged or copied.
"""

import argparse
import json
import tarfile
import urllib.request
import zipfile
from pathlib import Path

from streamnav.data.manifest import file_hash


def download(url, path):
    if path.exists():
        return
    temp = path.with_suffix(path.suffix + ".partial")
    urllib.request.urlretrieve(url, temp)
    temp.rename(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-archives", default="/inspire/dataset/hm3d/v1")
    parser.add_argument("--output", default="runtime/data")
    args = parser.parse_args()
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    report = []
    for version in ("v1", "v2"):
        archive = root / f"objectnav_hm3d_{version}.zip"
        url = f"https://dl.fbaipublicfiles.com/habitat/data/datasets/objectnav/hm3d/{version}/objectnav_hm3d_{version}.zip"
        download(url, archive)
        destination = root / "datasets/objectnav/hm3d" / version
        with zipfile.ZipFile(archive) as f:
            bad = f.testzip()
            if bad:
                raise ValueError(f"Corrupt archive member: {bad}")
            f.extractall(destination)
        report.append({"url": url, "sha256": file_hash(archive)})
        print(f"Prepared HM3D {version} episodes", flush=True)
    archive = root / "hm3d_ovon.tar.gz"
    url = "https://huggingface.co/datasets/nyokoyama/hm3d_ovon/resolve/main/hm3d.tar.gz"
    download(url, archive)
    with tarfile.open(archive) as f:
        f.extractall(root / "datasets/ovon", filter="data")
    report.append({"url": url, "sha256": file_hash(archive)})
    for split in ("train", "val", "minival"):
        archive = Path(args.scene_archives) / f"hm3d-{split}-habitat-v0.2.tar"
        if not archive.exists():
            raise FileNotFoundError(f"Provide the licensed scene archive: {archive}")
        target = root / "scene_datasets/hm3d" / split
        target.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive) as f:
            f.extractall(target, filter="data")
        report.append({"local_archive": str(archive), "size": archive.stat().st_size})
        print(f"Prepared HM3D {split} scenes", flush=True)
    (root / "download_provenance.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
