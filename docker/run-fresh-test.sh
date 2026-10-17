#!/usr/bin/env bash
# Test the install steps from scratch in a throwaway Linux container with the GPU attached (Linux / Git Bash).
#   docker/run-fresh-test.sh --accept-license [--keep-models]
# See run-fresh-test.ps1 for what the options mean.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
repo="$(cd "$here/.." && pwd)"
accept=0
keep=()
for arg in "$@"; do
  case "$arg" in
    --accept-license) accept=1 ;;
    --keep-models) keep=(-v qie-fresh-models:/home/tester/.local/share/qwen-image-editor/models -v qie-fresh-uv-cache:/home/tester/.cache/uv) ;;
    *) echo "unknown option: $arg"; exit 2 ;;
  esac
done

docker info >/dev/null 2>&1 || { echo "Docker isn't running."; exit 1; }
docker build -t qie-fresh-test -f "$here/fresh-test.Dockerfile" "$here"

out="$repo/outputs/fresh-test/$(date +%Y%m%d-%H%M%S)"
mkdir -p "$out"
docker run --rm --gpus all --ulimit memlock=-1 --shm-size 8g \
  -v "$repo:/src:ro" -v "$out:/results" -e ACCEPT_LICENSE=$accept "${keep[@]}" \
  qie-fresh-test 2>&1 | tee "$out/log.txt"
echo
echo "Results and log: $out"
