# Run the M39 weighted-MAIN experiment detached (nohup under Git-Bash does not survive
# session teardown here -- it silently killed two long jobs already).
param(
    [double]$Boost = 3.0,
    [int]$Seeds = 3,
    [string]$Log = ""
)
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$env:PYTHONPATH = Join-Path $root "src"
$env:PYTHONIOENCODING = "utf-8"
if (-not $Log) { $Log = "scratchpad/_m39_boost$($Boost -replace '\.','p').log" }

$argv = @("-u", "scratchpad/train_weighted_main.py", "--boost", $Boost, "--seeds", $Seeds)
$p = Start-Process -FilePath "python" -ArgumentList $argv `
    -WorkingDirectory $root -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput $Log -RedirectStandardError "$Log.err"
Write-Output "launched PID $($p.Id) boost=$Boost -> $Log"
