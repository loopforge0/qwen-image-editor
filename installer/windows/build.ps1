<#
.SYNOPSIS
  Build the Windows installer: dist\QwenImageEditor-Setup-<version>.exe

.DESCRIPTION
  Stages the editor's code, the launcher, the icon and uv into installer\windows\build\stage, then compiles
  QwenImageEditor.iss with Inno Setup 6 (https://jrsoftware.org/isinfo.php; winget install JRSoftware.InnoSetup).
  Nothing of Qwen's is included: the models are downloaded on the user's PC by the editor's setup page.
#>
param([string]$UvVersion = "0.11.28")

$ErrorActionPreference = "Stop"
$root = (Resolve-Path "$PSScriptRoot\..\..").Path
$build = Join-Path $PSScriptRoot "build"
$stage = Join-Path $build "stage"
$version = (Select-String -Path "$root\pyproject.toml" -Pattern '^version = "(.+)"').Matches[0].Groups[1].Value

if (Test-Path $stage) { Remove-Item -Recurse -Force $stage }
New-Item -ItemType Directory -Force $stage | Out-Null

# The editor: what a user needs to run it, not the development and test material.
foreach ($item in "engine", "app", "pyproject.toml", "uv.lock", "README.md", "LICENSE") {
  Copy-Item -Recurse (Join-Path $root $item) $stage
}
Get-ChildItem $stage -Recurse -Directory -Filter "__pycache__" | Remove-Item -Recurse -Force
Copy-Item "$root\installer\assets\icon.ico" $stage

# uv: installs Python and the editor's packages on first start.
$uvZip = Join-Path $build "uv-$UvVersion.zip"
if (-not (Test-Path $uvZip)) {
  Write-Host "Downloading uv $UvVersion"
  Invoke-WebRequest "https://github.com/astral-sh/uv/releases/download/$UvVersion/uv-x86_64-pc-windows-msvc.zip" -OutFile $uvZip
}
Expand-Archive $uvZip -DestinationPath (Join-Path $build "uv") -Force
Copy-Item (Join-Path $build "uv\uv.exe") $stage
Copy-Item (Join-Path $build "uv\uvw.exe") $stage  # uv without a console window: starts the tray app

$iscc = @("${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe", "$env:ProgramFiles\Inno Setup 6\ISCC.exe",
          "$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe") | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $iscc) {
  Write-Host "Staged in $stage."
  Write-Host "To build the installer, install Inno Setup 6 (winget install JRSoftware.InnoSetup) and run this again."
  exit 1
}
& $iscc "/DAppVersion=$version" "$PSScriptRoot\QwenImageEditor.iss"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Write-Host "Built $root\dist\QwenImageEditor-Setup-$version.exe"
