# Launch the M38 RL loop as a process that OUTLIVES this session.
#
# `nohup ... &` from the bash tool does NOT survive here: the job is a child of the shell
# Claude Code spawns, so a session teardown kills it. It happened twice -- once to the
# Pareto measurement, once to the first RL run, both silently (empty log, no traceback,
# no exit code). Start-Process detaches from the caller's process tree.
#
# Usage:  powershell -ExecutionPolicy Bypass -File scratchpad/run_rl_detached.ps1
#         powershell -ExecutionPolicy Bypass -File scratchpad/run_rl_detached.ps1 -Iters 12 -StartIter 9 -StartCkpt data/models/rl_alakazam_iter8.json

# Defaults are the MEASURED settings for the Alakazam base, not M16's raw numbers:
#   lr 3e-5 / 10 epochs -- swept on a real trajectory (_sweep_update.py). M16's
#     1e-3 / 25 collapses this model in one update (bc_acc 0.733 -> 0.168) because its
#     scores are 2.0-2.4x more compressed than TR-650's.
#   batch 800 -- featurising a batch costs ~63 ms/decision single-threaded and dominates
#     the iteration (~95 min at batch 2000). 800 games keeps an iteration near 45 min so
#     8 of them fit in one night.
param(
    [int]$Iters = 8,
    [int]$StartIter = 1,
    [string]$StartCkpt = "data/models/bc_alakazam_fetch.json",
    [int]$Batch = 800,
    [double]$Lambda = 0.10,
    [double]$Lr = 3e-5,
    [int]$Epochs = 10,
    [int]$Workers = 4,
    # evaluate_vs_init() does max(EvalN, 200); 300 actually tightens the head-to-head CI
    # instead of the previous 120->200 clamp doing nothing (it was already floored at 200).
    [int]$EvalN = 300,
    [string]$Log = "scratchpad/_m38_rl.log"
)

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$env:PYTHONPATH = Join-Path $root "src"
$env:PYTHONIOENCODING = "utf-8"

$argv = @(
    "-u", "scratchpad/rl_selfplay.py", "--setup", "alakazam", "iterate",
    "--start-ckpt", $StartCkpt, "--start-iter", $StartIter, "--iters", $Iters,
    "--batch", $Batch, "--epochs", $Epochs, "--lambda", $Lambda, "--lr", $Lr,
    "--eval-every", 2, "--eval-n", $EvalN, "--workers", $Workers
)

$p = Start-Process -FilePath "python" -ArgumentList $argv `
    -WorkingDirectory $root -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput $Log -RedirectStandardError "$Log.err"

Write-Output "launched PID $($p.Id)  ->  $Log"
Write-Output "  base=$StartCkpt  iters=$StartIter..$($StartIter + $Iters - 1)"
Write-Output "  batch=$Batch  lambda=$Lambda  lr=$Lr  epochs=$Epochs  workers=$Workers"
