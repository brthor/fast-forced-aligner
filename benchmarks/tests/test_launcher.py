"""The launcher preserves argument boundaries and isolates writable state."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


@pytest.fixture
def launcher(tmp_path: Path):
    publish = tmp_path / "publish with spaces"
    benchmarks = publish / "benchmarks"
    benchmarks.mkdir(parents=True)
    script = benchmarks / "run.sh"
    shutil.copy2(Path(__file__).resolve().parents[1] / "run.sh", script)
    binaries = tmp_path / "bin"
    binaries.mkdir()
    capture = tmp_path / "docker-arguments.json"
    docker = binaries / "docker"
    docker.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "with open(os.environ['DOCKER_ARGUMENT_CAPTURE'], 'w') as output:\n"
        "    json.dump(sys.argv[1:], output)\n"
    )
    docker.chmod(0o755)
    cache = tmp_path / "cache with spaces"
    environment = {
        **os.environ,
        "PATH": f"{binaries}:{os.environ['PATH']}",
        "DOCKER_ARGUMENT_CAPTURE": str(capture),
        "BENCHMARK_CACHE": str(cache),
    }

    def invoke(*args: str, **overrides: str) -> list[str]:
        subprocess.run([str(script), *args], check=True, env={**environment, **overrides},
                       capture_output=True, text=True)
        return json.loads(capture.read_text())

    return invoke, publish, cache


def values_after(arguments: list[str], option: str) -> list[str]:
    return [arguments[index + 1] for index, value in enumerate(arguments[:-1]) if value == option]


def test_runtime_paths_and_spaces_keep_new_destination_uncreated(launcher, tmp_path: Path):
    invoke, publish, cache = launcher
    inputs = tmp_path / "frozen inputs"
    inputs.mkdir()
    destination = tmp_path / "result parent" / "new results"
    arguments = invoke("run", "--inputs", str(inputs), f"--output={destination}",
                       BENCHMARK_GPUS="none", BENCHMARK_NETWORK="none")
    mounts = values_after(arguments, "--mount")
    assert f"type=bind,src={publish},dst=/workspace/publish,readonly" in mounts
    assert any(f"src={inputs}," in mount and mount.endswith(",readonly") for mount in mounts)
    assert any(f"src={destination.parent}," in mount and not mount.endswith(",readonly") for mount in mounts)
    assert values_after(arguments, "--output")[-1].endswith("/new results")
    assert not destination.exists()
    assert not (cache / "results/latest").exists()
    assert "--gpus" not in arguments
    assert values_after(arguments, "--network") == ["none"]
    assert values_after(arguments, "--user") == [f"{os.getuid()}:{os.getgid()}"]
    assert "HOME=/cache/home" in values_after(arguments, "--env")


def test_prepare_mounts_bundle_readonly_and_only_creates_destination_parent(launcher, tmp_path: Path):
    invoke, _, _ = launcher
    bundle = tmp_path / "input bundle.zip"
    bundle.touch()
    destination = tmp_path / "prepared parent" / "prepared inputs"
    arguments = invoke("prepare", "--bundle", str(bundle), "--root", str(destination))
    assert not destination.exists()
    assert destination.parent.is_dir()
    assert any(f"src={bundle}," in mount and mount.endswith(",readonly")
               for mount in values_after(arguments, "--mount"))
    assert "--gpus" not in arguments


def test_gpu_selectors_and_test_command_are_explicit(launcher):
    invoke, _, _ = launcher
    arguments = invoke("python", "-c", "print('a b')", BENCHMARK_GPUS="device=0")
    assert values_after(arguments, "--gpus") == ["device=0"]
    assert arguments[-2:] == ["-c", "print('a b')"]
    arguments = invoke("python", "-c", "pass", BENCHMARK_GPUS="nvidia.com/gpu=all")
    assert values_after(arguments, "--device") == ["nvidia.com/gpu=all"]
    arguments = invoke("test", BENCHMARK_GPUS="all")
    assert "--gpus" not in arguments and "--device" not in arguments


def test_build_uses_only_publish_context_and_preserves_flags(launcher):
    invoke, publish, cache = launcher
    arguments = invoke("build", "--build-arg", "VALUE=two words")
    assert arguments[0] == "build"
    assert arguments[-1] == str(publish)
    assert values_after(arguments, "--file") == [str(publish / "benchmarks/Dockerfile")]
    assert values_after(arguments, "--build-arg") == ["VALUE=two words"]
    assert not cache.exists()


def test_sweep_uses_same_gpu_and_path_isolation(launcher, tmp_path):
    invoke, _, _ = launcher
    destination = tmp_path / "results" / "sweep"
    arguments = invoke("sweep", "--output", str(destination), BENCHMARK_GPUS="device=0")
    assert "nemo_sweep.py" in arguments
    assert values_after(arguments, "--gpus") == ["device=0"]
    assert not destination.exists()


def test_public_prepare_uses_separate_image_and_cuda_by_default(launcher):
    invoke, _, _ = launcher
    arguments = invoke("prepare", "--limit", "2")
    assert "fast-forced-aligner-prepare:local" in arguments
    assert "regenerate.py" in arguments and "inputs.py" not in arguments
    assert values_after(arguments, "--gpus") == ["all"]
    assert values_after(arguments, "--limit") == ["2"]
    assert "HF_HOME=/cache/huggingface" in values_after(arguments, "--env")
    arguments = invoke("prepare", BENCHMARK_GPUS="none", BENCHMARK_PREPARE_IMAGE="custom:prep")
    assert "custom:prep" in arguments and "--gpus" not in arguments
    assert values_after(arguments, "--device")[-1] == "cpu"


def test_prepare_build_and_default_suite(launcher):
    invoke, publish, _ = launcher
    arguments = invoke("build-prepare", "--build-arg", "VALUE=two words")
    assert values_after(arguments, "--file") == [str(publish / "benchmarks/Dockerfile.prepare")]
    assert values_after(arguments, "--tag") == ["fast-forced-aligner-prepare:local"]
    for command in ("run", "sweep", "verify"):
        arguments = invoke(command, BENCHMARK_GPUS="none")
        assert values_after(arguments, "--suite") == ["regenerated-readme"]
