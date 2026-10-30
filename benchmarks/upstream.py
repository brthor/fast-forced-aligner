"""Fetch and verify pinned source files without importing upstream packages."""
from __future__ import annotations

import argparse
import ast
import hashlib
import os
from pathlib import Path
import tempfile
from typing import Any, Optional
from urllib.request import urlopen

CACHE = Path(os.environ.get("BENCHMARK_CACHE", Path(__file__).resolve().parent / ".local"))
ROOT = CACHE / "upstream"
SOURCES = {
    "nemo": {
        "repository": "NVIDIA-NeMo/Speech",
        "revision": "9bd9a949554abae9262327840bf72b03f0bd5b74",
        "path": "nemo/collections/asr/parts/utils/aligner_utils.py",
        "sha256": "9d963f8a09a722ff118326de25a7c61dee496aef6321a7d6141506917fb91fb5",
    },
    "ctc_cpp": {
        "repository": "MahmoudAshraf97/ctc-forced-aligner",
        "revision": "64293cc6d711e57666c4a8b098e9fd93b381fd88",
        "path": "ctc_forced_aligner/forced_align_impl.cpp",
        "sha256": "0f6410d89e5c1d3c1ee226d6b024fabc82f16a99baa0fe8defdf97f9875d2804",
    },
}


def source_path(name: str, *, fetch: bool = True) -> Path:
    spec = SOURCES[name]
    path = ROOT / name / spec["revision"] / Path(spec["path"]).name
    if not path.exists():
        if not fetch:
            raise FileNotFoundError(f"Run python benchmarks/upstream.py first: {path}")
        url = f"https://raw.githubusercontent.com/{spec['repository']}/{spec['revision']}/{spec['path']}"
        with urlopen(url, timeout=60) as response:
            data = response.read()
        if hashlib.sha256(data).hexdigest() != spec["sha256"]:
            raise ValueError(f"upstream hash mismatch: {name}")
        path.parent.mkdir(parents=True, exist_ok=True)
        # Publish only complete, verified files when simultaneous benchmark
        # invocations share a cache. Readers must never see a partial download.
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(data)
            temporary.replace(path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    if hashlib.sha256(path.read_bytes()).hexdigest() != spec["sha256"]:
        raise ValueError(f"cached upstream hash mismatch: {name}")
    return path


def nemo_function() -> Any:
    """Compile the original function AST, without executing model imports."""
    import torch
    path = source_path("nemo")
    module = ast.parse(path.read_text(), filename=str(path))
    definitions = [node for node in module.body
                   if isinstance(node, ast.FunctionDef) and node.name == "viterbi_decoding"]
    if len(definitions) != 1:
        raise ValueError("expected exactly one upstream viterbi_decoding function")
    namespace: dict[str, Any] = {"torch": torch, "Optional": Optional}
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(path), "exec"), namespace)
    return namespace["viterbi_decoding"]


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    for name in SOURCES:
        print(source_path(name, fetch=True))


if __name__ == "__main__":
    main()
