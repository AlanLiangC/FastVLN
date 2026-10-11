"""Download pinned public LightNav weights with verified HTTP byte ranges."""

import argparse
import hashlib
import json
import shutil
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from huggingface_hub import HfApi


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", default="826dc5fbfa37afa8293d2e336d329b6ffc0bfb64")
    parser.add_argument("--directory", default="checkpoints/lightnav0_base")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--parallelism", type=int, default=16)
    args = parser.parse_args()
    repo = "LightOriginsHQ/LightNav-0"
    info = HfApi().model_info(repo, revision=args.revision, files_metadata=True)
    entry = next(s for s in info.siblings if s.rfilename.endswith(".safetensors"))
    total, expected_hash = entry.size, entry.lfs.sha256
    root = Path(args.directory)
    root.mkdir(parents=True, exist_ok=True)
    parts = root / ".weight-parts"
    parts.mkdir(exist_ok=True)
    url = f"https://huggingface.co/{repo}/resolve/{args.revision}/{entry.rfilename}"

    def fetch(index):
        start, end = total * index // args.workers, total * (index + 1) // args.workers - 1
        path = parts / f"part-{index:02d}"
        if path.exists() and path.stat().st_size == end - start + 1:
            return index
        for attempt in range(12):
            offset = path.stat().st_size if path.exists() else 0
            if offset == end - start + 1:
                return index
            if offset > end - start + 1:
                raise RuntimeError("Partial file belongs to a different partition recipe")
            request = urllib.request.Request(
                url + f"?range_part={index}&offset={offset}",
                headers={"Range": f"bytes={start + offset}-{end}"},
            )
            try:
                with urllib.request.urlopen(request, timeout=120) as response:
                    if (
                        response.status != 206
                        or response.headers.get("Content-Range")
                        != f"bytes {start + offset}-{end}/{total}"
                    ):
                        raise ValueError("Server did not honor the requested byte range")
                    with path.open("ab") as output:
                        while block := response.read(8 * 1024 * 1024):
                            output.write(block)
            except (OSError, TimeoutError) as exc:
                print(
                    json.dumps(
                        {
                            "retry_range": index,
                            "attempt": attempt + 1,
                            "error_type": type(exc).__name__,
                        }
                    ),
                    flush=True,
                )
        if path.stat().st_size != end - start + 1:
            raise RuntimeError("Incomplete range after retries; partial data retained")
        return index

    with ThreadPoolExecutor(max_workers=min(args.workers, args.parallelism)) as executor:
        for future in as_completed([executor.submit(fetch, i) for i in range(args.workers)]):
            print(
                json.dumps({"completed_range": future.result(), "total_ranges": args.workers}),
                flush=True,
            )
    destination = root / (entry.rfilename + ".verified-new")
    digest = hashlib.sha256()
    with destination.open("wb") as output:
        for i in range(args.workers):
            with (parts / f"part-{i:02d}").open("rb") as part:
                while block := part.read(8 * 1024 * 1024):
                    digest.update(block)
                    output.write(block)
    if digest.hexdigest() != expected_hash:
        raise RuntimeError("Downloaded weights failed SHA256 verification")
    destination.replace(root / entry.rfilename)
    (root / "source_revision.json").write_text(
        json.dumps(
            {
                "repo": repo,
                "revision": args.revision,
                "weight_sha256": expected_hash,
                "weight_bytes": total,
            },
            indent=2,
        )
        + "\n"
    )
    shutil.rmtree(parts)
    print(
        json.dumps({"download_complete": str(root), "sha256_verified": expected_hash}), flush=True
    )


if __name__ == "__main__":
    main()
