"""Original C++ comparator and explicitly corrected, threaded adaptation."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from functools import lru_cache
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sysconfig
from typing import Any

import numpy as np
import torch

from inputs import Case
from upstream import CACHE, SOURCES, source_path
from .common import DeviceBackend, admit

PATCHES = Path(__file__).resolve().parents[1] / "patches"


@lru_cache(maxsize=2)
def cpp_function(corrected: bool) -> tuple[Any, dict[str, Any]]:
    """Compile before timing, recording source, compiler, ABI and build flags."""
    upstream = source_path("ctc_cpp").resolve()
    source = (PATCHES / "cpu_corrected.cpp") if corrected else upstream
    module_name = "ffa_corrected_ctc_cpp" if corrected else "ctc_forced_aligner"
    compiler = os.environ.get("CXX", "c++")
    version = subprocess.run([compiler, "--version"], check=True, capture_output=True,
                             text=True).stdout.splitlines()[0]
    fingerprint = {
        "upstream": SOURCES["ctc_cpp"],
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "patch_sha256": hashlib.sha256((PATCHES / "cpu_corrected.patch").read_bytes()).hexdigest()
        if corrected else None,
        "compiler": version, "compiler_executable": compiler,
        "python_abi": sysconfig.get_config_var("SOABI"), "torch": torch.__version__,
        "flags": ["-O3", "-shared", "-std=c++17", "-fPIC"] + (["-pthread"] if corrected else []),
    }
    digest = hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode()).hexdigest()
    build = CACHE / "build" / module_name / digest[:20]
    build.mkdir(parents=True, exist_ok=True)
    output = build / (module_name + sysconfig.get_config_var("EXT_SUFFIX"))
    command = [compiler, *fingerprint["flags"], "-I" + sysconfig.get_paths()["include"],
               "-I" + str(Path(torch.__file__).parent / "include"), str(source), "-o", str(output)]
    manifest = {**fingerprint, "command": command}
    if not output.exists():
        # Replace only a completed shared object, including when two invocations
        # happen to populate the same cache concurrently.
        temporary = output.with_name(output.name + f".{os.getpid()}.tmp")
        try:
            subprocess.run([*command[:-1], str(temporary)], check=True)
            temporary.replace(output)
        finally:
            temporary.unlink(missing_ok=True)
    (build / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    spec = importlib.util.spec_from_file_location(module_name, output)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load compiled comparator: {output}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.forced_align, manifest


class CppBackend(DeviceBackend):
    """Include layout, allocation, dispatch, native DP and score recovery in run.

    The corrected primitive releases the GIL for native recurrence and traceback.
    Its persistent pool starts during warmup, submits one record per worker task,
    and collects results in input order. No transcript is split across workers.
    """

    def __init__(self, *, corrected: bool = False, workers: int = 1) -> None:
        if type(workers) is not int or workers < 1:
            raise ValueError("workers must be a positive integer")
        if not corrected and workers != 1:
            raise ValueError("unchanged upstream C++ is serial; workers requires corrected=True")
        self.device = torch.device("cpu")
        self.name = "ctc-corrected" if corrected else "ctc-cpp"
        self.workers = workers
        self.function, self.build_manifest = cpp_function(corrected)
        self._pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="ctc-cpu") if workers > 1 else None
        self._closed = False

    def prepare_batch(self, cases: list[Case]) -> list[Case]:
        admit(cases, cpp=True)
        # Dtype admission is excluded, as is initial CUDA placement for GPU lanes.
        # Contiguous layout and the native batch axis remain in the timed call.
        return [replace(case, emissions=np.asarray(case.emissions, dtype=np.float32),
                        targets=np.asarray(case.targets, dtype=np.int64)) for case in cases]

    def _run_one(self, item: tuple[np.ndarray, np.ndarray, int]):
        return self.function(*item)

    def run_batch(self, cases: list[Case]):
        if self._closed:
            raise RuntimeError("CPU backend is closed")
        prepared = [(np.ascontiguousarray(case.emissions[None]),
                     np.ascontiguousarray(case.targets[None]), int(case.blank_id)) for case in cases]
        if self._pool is None:
            return [self._run_one(item) for item in prepared]
        futures = [self._pool.submit(self._run_one, item) for item in prepared]
        outputs = []
        first_error: Exception | None = None
        for future in futures:
            try:
                outputs.append(future.result())
            except Exception as error:
                if first_error is None:
                    first_error = error
        if first_error is not None:
            raise first_error
        return outputs

    def materialize_batch(self, raw) -> list[tuple[np.ndarray, np.ndarray]]:
        return [(labels.reshape(-1), scores.reshape(-1)) for labels, scores in raw]

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=True)
        self._closed = True
