<#
.SYNOPSIS
  Test the install steps from scratch in a throwaway Linux container with the GPU attached.

.DESCRIPTION
  Builds a bare Ubuntu image (git and curl only), then inside it: installs uv, copies the repo the way a clone
  would, runs uv sync, qie setup, two generations and the editor. Results go to outputs\fresh-test\<time>.

.PARAMETER AcceptLicense
  Accept the Qwen Research License (non-commercial use only) for the test; needed to download the models.

.PARAMETER KeepModels
  Keep downloaded models and uv's package cache in Docker volumes, so later runs skip the 17 GB download.
  Without it, every run starts from nothing.

.EXAMPLE
  .\docker\run-fresh-test.ps1 -AcceptLicense
#>
param([switch]$AcceptLicense, [switch]$KeepModels)

$ErrorActionPreference = "Stop"
$repo = (Resolve-Path "$PSScriptRoot\..").Path

docker info --format "{{.ServerVersion}}" *> $null
if ($LASTEXITCODE -ne 0) { Write-Host "Docker isn't running. Start Docker Desktop and try again."; exit 1 }

$memGiB = [math]::Round([double](docker info --format "{{.MemTotal}}") / 1GB, 1)
if ($memGiB -lt 28) {
  Write-Host "Note: Docker has $memGiB GB of RAM. The engine pins about 15 GB, so the test may fail or be slow."
  Write-Host "      To give WSL more, add to $env:USERPROFILE\.wslconfig:  [wsl2]  memory=28GB   then run: wsl --shutdown"
}

$usedMiB = [int](nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | Select-Object -First 1)
if ($usedMiB -gt 2000) {
  Write-Host "Warning: $usedMiB MiB of GPU memory is already in use (the editor, ComfyUI, a game?)."
  Write-Host "         Close it first: two GPU jobs at once make the test slow or fail."
}

docker build -t qie-fresh-test -f "$PSScriptRoot\fresh-test.Dockerfile" "$PSScriptRoot"
if ($LASTEXITCODE -ne 0) { exit 1 }

$out = Join-Path $repo ("outputs\fresh-test\" + (Get-Date -Format "yyyyMMdd-HHmmss"))
New-Item -ItemType Directory -Force $out | Out-Null

$run = @("run", "--rm", "--gpus", "all", "--ulimit", "memlock=-1", "--shm-size", "8g",
         "-v", "${repo}:/src:ro", "-v", "${out}:/results",
         "-e", ("ACCEPT_LICENSE=" + [int]$AcceptLicense.IsPresent))
if ($KeepModels) {
  $run += @("-v", "qie-fresh-models:/home/tester/.local/share/qwen-image-editor/models",
            "-v", "qie-fresh-uv-cache:/home/tester/.cache/uv")
}
$run += "qie-fresh-test"

docker @run 2>&1 | Tee-Object -FilePath (Join-Path $out "log.txt")
$code = $LASTEXITCODE
Write-Host ""
Write-Host "Results and log: $out"
exit $code
