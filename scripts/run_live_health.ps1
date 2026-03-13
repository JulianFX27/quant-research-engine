param(
    [switch]$Watch,
    [int]$Interval = 15,
    [switch]$NoClear,
    [switch]$NoColor
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

if (Test-Path ".\.venv\Scripts\Activate.ps1") {
    . .\.venv\Scripts\Activate.ps1
}

$cmd = @("-m", "src.tools.live_health_check")

if ($Watch) {
    $cmd += "--watch"
    $cmd += "--interval"
    $cmd += "$Interval"
}

if ($NoClear) {
    $cmd += "--no-clear"
}

if ($NoColor) {
    $cmd += "--no-color"
}

python @cmd