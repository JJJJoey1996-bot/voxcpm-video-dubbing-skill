$ErrorActionPreference = "Stop"

$RootDir = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
Set-Location $RootDir

if (Get-Command py -ErrorAction SilentlyContinue) {
  $PythonBin = "py"
} elseif (Get-Command python -ErrorAction SilentlyContinue) {
  $PythonBin = "python"
} else {
  throw "Python 3.10+ is required."
}

& $PythonBin ".\skills\short-video-dubbing\scripts\install.py" @args
