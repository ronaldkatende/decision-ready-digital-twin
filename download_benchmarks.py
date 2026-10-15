from __future__ import annotations

import hashlib
import json
from pathlib import Path
import requests

from dtwinexp.config import DATA, NET3_PATH

NET3_URL = "https://raw.githubusercontent.com/USEPA/WNTR/main/examples/networks/Net3.inp"


def sha256(path: Path, block_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(block_size), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_required_data() -> dict:
    DATA.mkdir(parents=True, exist_ok=True)
    if NET3_PATH.exists() and NET3_PATH.stat().st_size > 0:
        status = "present"
    else:
        tmp = NET3_PATH.with_suffix(".inp.part")
        headers = {"User-Agent": "decision-ready-digital-twin-reproducibility/1.0"}
        with requests.get(NET3_URL, stream=True, timeout=120, headers=headers) as r:
            r.raise_for_status()
            with tmp.open("wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
        tmp.replace(NET3_PATH)
        status = "downloaded"

    manifest = {
        "source": NET3_URL,
        "file": "data/Net3.inp",
        "status": status,
        "bytes": NET3_PATH.stat().st_size,
        "sha256": sha256(NET3_PATH),
    }
    (DATA / "download_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest


if __name__ == "__main__":
    print(json.dumps(ensure_required_data(), indent=2))
