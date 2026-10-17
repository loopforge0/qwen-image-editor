#!/bin/sh
# Pull code changes from the PC into the container (stands in for `git pull`).
# Needs the repository mounted read-only at /src. Files deleted on the PC are not deleted here.
set -e
if [ ! -d /src ]; then
  echo "Start the container with the repository mounted:  -v <repo>:/src:ro"
  exit 1
fi
tar -C /src --exclude=./.git --exclude=./.venv --exclude=./outputs --exclude='*/__pycache__' --exclude=./.playwright-mcp -cf - . \
  | tar -C "$HOME/qwen-image-editor" -xf -
echo "Code updated from /src."
