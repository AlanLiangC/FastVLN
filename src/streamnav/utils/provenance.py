"""Snapshot current source/config/docs; credentials and datasets are excluded."""

import zipfile
from hashlib import sha256
from pathlib import Path

from streamnav.training import distributed as parallel


def snapshot_sources(root):
    record = None
    if parallel.rank() == 0:
        temp_source = root / ".source-writing.zip"
        with zipfile.ZipFile(temp_source, "w", zipfile.ZIP_DEFLATED) as archive:
            for folder in ("src", "services", "configs", "scripts", "tools", "docs", "tests"):
                for path in sorted(Path(folder).rglob("*")):
                    if path.is_file() and "__pycache__" not in str(path):
                        archive.write(path, str(path))
            for filename in ("README.md", "pyproject.toml", "uv.lock", "Makefile"):
                archive.write(filename, filename)
        source_hash = sha256(temp_source.read_bytes()).hexdigest()
        source_path = root / "source" / f"{source_hash}.zip"
        source_path.parent.mkdir(exist_ok=True)
        if source_path.exists():
            temp_source.unlink()
        else:
            temp_source.rename(source_path)
        link = root / ".source-new"
        link.unlink(missing_ok=True)
        link.symlink_to(f"source/{source_hash}.zip")
        link.replace(root / "source.zip")
        record = {"sha256": source_hash, "path": str(source_path.resolve())}
    return parallel.gather(record)[0]
