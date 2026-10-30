#!/usr/bin/env bash
# Every runtime command uses an installed package and an unprivileged container.
set -euo pipefail
bench_dir="$(cd -P "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
publish_dir="$(dirname "$bench_dir")"
image="${BENCHMARK_IMAGE:-fast-forced-aligner-benchmarks:local}"
prepare_image="${BENCHMARK_PREPARE_IMAGE:-fast-forced-aligner-prepare:local}"
cache="${BENCHMARK_CACHE:-$publish_dir/.local/benchmarks}"
command="${1:-help}"
if [[ $# -gt 0 ]]; then shift; fi

if [[ "$command" == help || "$command" == --help || "$command" == -h ]]; then
    cat <<'USAGE'
Usage: benchmarks/run.sh build|build-prepare|prepare|verify|fetch|run|sweep|report|test|python [args]

  build     Build the pinned CUDA development environment and install this package.
  build-prepare  Build the separate stock Transformers/NeMo preparation image.
  prepare   Download public audio/models and regenerate inputs (GPU by default).
            --limit N makes an explicitly partial per-workload smoke selection.
            --bundle PATH or --url URL --sha256 HASH imports an archived bundle.
  verify    Verify the prepared input bundle.
  fetch     Check/download pinned upstream comparator source files.
  run       Measure the complete suite; defaults: B64, 2 rounds, 3 repetitions.
            --suite synthetic needs no prepared bundle.
  sweep     Compare five NeMo schedules, then remeasure winners and B64 control.
  report    Render the most recent results (Markdown and image).
  test      Run benchmark unit tests, without granting GPU access.
  python    Run a Python diagnostic in the same environment (GPU access enabled).

Defaults live in publish/.local/benchmarks. Override BENCHMARK_CACHE to put
inputs, results and extension caches on another disk. --bundle, --root,
--inputs, --input and --output accept host paths and are mounted automatically.
Use BENCHMARK_GPUS=none for CPU-only run/python; otherwise it is passed to
Docker --gpus (default: all). CDI identifiers such as nvidia.com/gpu=all use
--device instead. BENCHMARK_IMAGE overrides the image tag. DOCKER_HOST and
Docker context selection follow the normal Docker CLI configuration.
BENCHMARK_PREPARE_IMAGE overrides the preparation image tag.
BENCHMARK_CPU_WORKERS sets the default comparison worker count (default: 8).
BENCHMARK_MAX_JOBS controls extension build parallelism (default: 2).
BENCHMARK_NETWORK=none disables runtime networking after all inputs are cached.
USAGE
    exit 0
fi
if [[ "$command" == build || "$command" == build-prepare ]]; then
    dockerfile="$bench_dir/Dockerfile"
    if [[ "$command" == build-prepare ]]; then
        dockerfile="$bench_dir/Dockerfile.prepare"
        image="$prepare_image"
    fi
    exec docker build --file "$dockerfile" --tag "$image" "$@" "$publish_dir"
fi
prepare_bundle=false
if [[ "$command" == prepare ]]; then
    for argument in "$@"; do
        case "$argument" in --bundle|--bundle=*|--url|--url=*) prepare_bundle=true ;; esac
    done
    if [[ "$prepare_bundle" == false ]]; then image="$prepare_image"; fi
fi
case "$command" in
    prepare|verify|fetch|run|sweep|report|test|python) ;;
    *) echo "Unknown command: $command (use --help)" >&2; exit 2 ;;
esac
mkdir -p "$cache"
cache="$(cd -P "$cache" && pwd)"
mkdir -p "$cache/home" "$cache/extensions" "$cache/matplotlib" "$cache/results"
mounts=(--mount "type=bind,src=$publish_dir,dst=/workspace/publish,readonly"
        --mount "type=bind,src=$cache,dst=/cache")
args=()
# Explicit data paths can live outside the checkout/cache (including large bundles).
# Bind each path independently so spaces survive and host paths never leak into CLI.
while [[ $# -gt 0 ]]; do
    option="$1"; shift
    case "$option" in
        --bundle=*|--root=*|--inputs=*|--input=*|--output=*)
            set -- "${option#*=}" "$@"; option="${option%%=*}" ;;
    esac
    case "$option" in
        --bundle|--root|--inputs|--input|--output)
            [[ $# -gt 0 ]] || { echo "$option requires a host path" >&2; exit 2; }
            path="$1"; shift
            if [[ "$option" == --output || ( "$option" == --root && "$command" == prepare ) ]]; then
                # Mount the parent: the runner creates its destination atomically
                # and rejects existing results rather than overwriting evidence.
                parent="$(dirname "$path")"
                mkdir -p "$parent"
                parent="$(cd -P "$parent" && pwd)"
                target="/external/${#args[@]}"
                mounts+=(--mount "type=bind,src=$parent,dst=$target")
                args+=("$option" "$target/$(basename "$path")")
                continue
            fi
            [[ -e "$path" ]] || { echo "Path does not exist: $path" >&2; exit 2; }
            path="$(realpath "$path")"
            target="/external/${#args[@]}"
            access=,readonly
            mounts+=(--mount "type=bind,src=$path,dst=$target$access")
            args+=("$option" "$target")
            ;;
        *) args+=("$option") ;;
    esac
done

runtime=(--rm --user "$(id -u):$(id -g)" --shm-size 2g
         --workdir /workspace/publish/benchmarks
         --env BENCHMARK_CACHE=/cache --env HOME=/cache/home
         --env HF_HOME=/cache/huggingface --env NEMO_CACHE_DIR=/cache/nemo
         --env TORCH_EXTENSIONS_DIR=/cache/extensions --env MPLCONFIGDIR=/cache/matplotlib
         --env "MAX_JOBS=${BENCHMARK_MAX_JOBS:-2}"
         --env OMP_NUM_THREADS=1 --env MKL_NUM_THREADS=1 --env OPENBLAS_NUM_THREADS=1)
if [[ -n "${BENCHMARK_NETWORK:-}" ]]; then runtime+=(--network "$BENCHMARK_NETWORK"); fi
if [[ "$command" == run || "$command" == sweep || "$command" == python ||
      ( "$command" == prepare && "$prepare_bundle" == false ) ]]; then
    gpu_selector="${BENCHMARK_GPUS:-all}"
    if [[ "$gpu_selector" == nvidia.com/gpu=* ]]; then
        runtime+=(--device "$gpu_selector")
    elif [[ "$gpu_selector" != none ]]; then
        runtime+=(--gpus "$gpu_selector")
    fi
fi
case "$command" in
    prepare)
        if [[ "$prepare_bundle" == true ]]; then
            invocation=(python inputs.py prepare --root /cache/inputs)
        else
            invocation=(python regenerate.py --root /cache/inputs --device cuda)
            if [[ "${BENCHMARK_GPUS:-all}" == none ]]; then invocation+=(--device cpu); fi
        fi ;;
    verify) invocation=(python inputs.py verify --root /cache/inputs --suite regenerated-readme) ;;
    fetch) invocation=(python upstream.py) ;;
    run) invocation=(python benchmark.py --inputs /cache/inputs --suite regenerated-readme
                      --output /cache/results/latest --batch-size 64 --rounds 2
                      --repetitions 3 --workers "${BENCHMARK_CPU_WORKERS:-8}") ;;
    sweep) invocation=(python nemo_sweep.py --inputs /cache/inputs --suite regenerated-readme
                        --output /cache/results/nemo-sweep --workers "${BENCHMARK_CPU_WORKERS:-8}") ;;
    report) invocation=(python report.py --input /cache/results/latest --output /cache/results/latest/report) ;;
    test) invocation=(python -m pytest -c /dev/null -o cache_dir=/cache/pytest tests) ;;
    python) invocation=(python) ;;
esac
# Seed only missing source files; adapters verify hashes on every access. A cached
# modification is deliberately not overwritten, so corruption is visible.
if [[ "$command" == prepare && "$prepare_bundle" == false ]]; then
    exec docker run "${runtime[@]}" "${mounts[@]}" "$image" "${invocation[@]}" "${args[@]}"
fi
exec docker run "${runtime[@]}" "${mounts[@]}" "$image" bash -c \
    'mkdir -p /cache/upstream; cp -rn /opt/benchmark-seed/upstream/. /cache/upstream/; exec "$@"' \
    benchmark "${invocation[@]}" "${args[@]}"
