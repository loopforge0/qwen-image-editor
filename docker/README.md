# Fresh-machine tests

Docker Desktop must be in **Linux containers** mode (tray icon → "Switch to Linux containers…").

## By hand

A bare Ubuntu machine with the project already cloned and nothing else installed. Open a shell in it and follow
the main README as a new user would:

```powershell
docker build -t qie-manual -f docker/manual.Dockerfile .
docker run -it --rm --gpus all --ulimit memlock=-1 --shm-size 8g -p 127.0.0.1:7861:7861 qie-manual
```

Inside, start the editor with `uv run qie app --host 0.0.0.0 --no-browser` (the container's own localhost isn't
reachable from Windows), then open http://127.0.0.1:7861 on the PC. Port 7861 is published to the PC's localhost
only. `--rm` throws everything away on exit, including the 17 GB of downloaded models.

To resume quickly after a bug, keep the downloads in volumes and mount the repo so fixes reach the container:

```powershell
docker run -it --rm --gpus all --ulimit memlock=-1 --shm-size 8g -p 127.0.0.1:7861:7861 -v qie-models:/home/tester/.local/share/qwen-image-editor/models -v qie-uv-cache:/home/tester/.cache/uv -v "C:\path\to\qwen-image-editor:/src:ro" qie-manual
```

One line on purpose, and with the repo's full path (change it to yours) so it works the same in Command Prompt
and PowerShell. `${PWD}` is PowerShell-only; Command Prompt's equivalent is `%cd%`.

- `qie-models` keeps the 17 GB of models; `qie setup` finds them and finishes in seconds.
- `qie-uv-cache` keeps downloaded packages; `uv sync` reinstalls from it.
- The settings file is not kept, so every new container still goes through `qie setup` and the license prompt.
- After changing code on the PC, run `update-code` inside the container (the `git pull` of this setup).
- Start over completely: `docker volume rm qie-models qie-uv-cache`.

## Scripted

Runs the README's install steps in a throwaway Ubuntu container that starts with nothing but git and curl, with
the GPU attached, and reports each step:

```
GPU visible · install uv · get the code · uv sync · doctor says not set up · qie setup ·
doctor says ready · text-to-image · edit with a reference · editor serves the page
```

Windows (Docker Desktop with the WSL2 backend):

```powershell
.\docker\run-fresh-test.ps1 -AcceptLicense              # everything from scratch, including the 17 GB download
.\docker\run-fresh-test.ps1 -AcceptLicense -KeepModels  # keep models and packages between runs
```

Linux (Docker with the NVIDIA Container Toolkit): `docker/run-fresh-test.sh --accept-license [--keep-models]`

`-AcceptLicense` accepts the Qwen Research License (non-commercial use only) inside the test container; the models
can't be downloaded without it. The log and generated images go to `outputs/fresh-test/<time>/`.

Notes:
- This tests the Linux install. The repository is copied in the way a clone would be (no `.venv`, `outputs`).
- WSL2 gets half the machine's RAM by default; the engine pins about 15 GB, so give it more in `%USERPROFILE%\.wslconfig`
  (`[wsl2]` then `memory=28GB`, then `wsl --shutdown`).
- Delete the kept volumes with `docker volume rm qie-fresh-models qie-fresh-uv-cache`.
