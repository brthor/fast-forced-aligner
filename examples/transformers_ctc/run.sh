#!/usr/bin/env bash
set -euo pipefail

example_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd -- "$example_dir/../.." && pwd)"
image="${RUNTIME_IMAGE:-fast-forced-aligner-transformers:torch280}"

if [[ "${1:-}" == build ]]; then
  args=(--pull=false --file "$example_dir/Dockerfile" --tag "$image")
  args+=(--build-arg "ALIGNER_REF=${ALIGNER_REF:-HEAD}")
  if [[ -n "${PYTORCH_IMAGE:-}" ]]; then
    args+=(--build-arg "PYTORCH_IMAGE=$PYTORCH_IMAGE")
  fi
  exec docker build "${args[@]}" "$project_dir"
fi

cache_dir="$project_dir/.local/transformers-example"
mkdir -p "$cache_dir/home" "$cache_dir/huggingface" "$cache_dir/extensions"
exec docker run --rm --gpus "${GPU_DEVICE:+device=}${GPU_DEVICE:-all}" \
  --user "$(id -u):$(id -g)" --shm-size 2g \
  --mount "type=bind,source=$cache_dir,target=/cache" \
  --env HOME=/cache/home --env HF_HOME=/cache/huggingface \
  --env TORCH_EXTENSIONS_DIR=/cache/extensions \
  "$image" "$@"
