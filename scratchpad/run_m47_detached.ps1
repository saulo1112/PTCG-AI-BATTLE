# Launch the M47 RL loop as a process that OUTLIVES this session.
#
# Same detachment reason as run_rl_detached.ps1: `nohup ... &` from the bash tool is a
# child of the shell Claude Code spawns and dies silently on session teardown (it cost
# M38 two long jobs, empty log and no exit code both times). Start-Process detaches.
#
# Usage:  powershell -ExecutionPolicy Bypass -File scratchpad/run_m47_detached.ps1
#
# EVERY DEFAULT BELOW WAS MEASURED TODAY, not inherited:
#
#   --setup alakazam_final  base = the SHIPPED champion (MAIN sha identical to the
#     `fetch` payload M38 used, but with the M42/M43 heads, so self-play now generates
#     the state distribution of the agent we actually deploy). Opponent mix reverted to
#     M38's 0.40/0.35/0.25 -- the 0.30/0.30/0.40 still sitting in the `alakazam` setup is
#     M39's reweight, which M39 measured as confidently WORSE.
#
#   --tau0 0.3   M38 collected at tau=1.0 and measured 33.3% of MAIN decisions off-argmax
#     against M16's healthy ~16% -- it fixed lr for the model's 2x-compressed score scale
#     and left tau alone. Measured curve today: 1.0->0.333, 0.6->0.256, 0.5->0.217,
#     0.4->0.201, 0.3->0.169, 0.25->0.151. 0.3 lands on M16's profile.
#
#   --critic v_alakazam_v2.json   refit on the 282k self-play decisions already on disk.
#     Held-out AUC 0.6526 -> 0.7902. Every advantage in M38/M39 was computed with the
#     0.65 critic.
#
#   --clip 0.2 --lr 1e-4   swept on a tau=0.3 batch collected by this exact base
#     (m47_sweep_update.py). The clip visibly saturates displacement -- unclipped goes
#     2.82%->4.83% from lr 1e-4->3e-4, clipped stays 1.85%->1.98% -- which is the trust
#     region working. 1e-4 sits where the UNCLIPPED arm would still be safe, so the run
#     does not depend on the mask being perfect.
#
#   --adv-norm   corrects a measured +0.093 offset in the mean advantage (the refit
#     critic was fit on tau=1.0 games, where the sampling policy won 40%). A
#     state-independent baseline shift cannot bias the policy gradient; it only removes
#     variance. Verified inert on the step itself (Adam is scale-invariant).
#
#   --eval-every 99   no in-loop arena. M38 spent ~30 min per eval on n=200 reads, and
#     M41 later proved that at n=200 this engine reads a mirror against ITSELF as 0.435.
#     The decisive measurement is scratchpad/m47_arena.py at n=600, run afterwards on
#     whichever checkpoints the log says are worth it. K2 (bc_acc drift, relative floor)
#     still guards the loop.
param(
    [int]$Iters = 18,
    [int]$StartIter = 1,
    [string]$StartCkpt = "data/models/bc_alakazam_final.json",
    [int]$Batch = 800,
    [double]$Lambda = 0.10,
    [double]$Lr = 1e-4,
    [double]$Tau = 0.3,
    [double]$Clip = 0.2,
    [int]$Epochs = 10,
    [int]$Workers = 4,
    [string]$Critic = "data/models/v_alakazam_v2.json",
    [string]$Tag = "",
    [string]$Log = "scratchpad/_m47_rl.log"
)

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$env:PYTHONPATH = Join-Path $root "src"
$env:PYTHONIOENCODING = "utf-8"

$argv = @(
    "-u", "scratchpad/rl_selfplay.py", "--setup", "alakazam_final", "iterate",
    "--start-ckpt", $StartCkpt, "--start-iter", $StartIter, "--iters", $Iters,
    "--batch", $Batch, "--epochs", $Epochs, "--lambda", $Lambda, "--lr", $Lr,
    "--tau0", $Tau, "--clip", $Clip, "--adv-norm", "--critic", $Critic,
    "--eval-every", 99, "--workers", $Workers
)
if ($Tag) { $argv += @("--tag", $Tag) }

$p = Start-Process -FilePath "python" -ArgumentList $argv `
    -WorkingDirectory $root -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput $Log -RedirectStandardError "$Log.err"

Write-Output "launched PID $($p.Id)  ->  $Log"
Write-Output "  base=$StartCkpt  iters=$StartIter..$($StartIter + $Iters - 1)"
Write-Output "  batch=$Batch lambda=$Lambda lr=$Lr tau=$Tau clip=$Clip epochs=$Epochs workers=$Workers"
Write-Output "  critic=$Critic"
Write-Output "WATCH: probe_agree per iteration is printed next to M38's value for the same"
Write-Output "iteration. If we are not above that curve by iter 6, the clip is not doing what"
Write-Output "the retroactive test predicted -- stop and go to plan B."
