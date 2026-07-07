# Full-scale TSGM C-MAPSS run: data prep -> train (iter_pre=5000/iter_main=10000) ->
# recursive PC sampling (n_steps=1000) -> 10-seed discriminative/predictive/t-SNE evaluation.
# See docs/reproduction_plan.md for the reproduction plan this follows.

$ErrorActionPreference = "Stop"

Set-Location (Join-Path $PSScriptRoot "..")

$Python = ".venv\Scripts\python.exe"
$Config = "configs/cmapss.yaml"
$CkptDir = "outputs/checkpoints/cmapss_full"
$Device = if ($env:DEVICE) { $env:DEVICE } else { "cpu" }
$NSamples = if ($env:N_SAMPLES) { $env:N_SAMPLES } else { "100" }
$NSteps = if ($env:N_STEPS) { $env:N_STEPS } else { "1000" }
$NSeeds = if ($env:N_SEEDS) { $env:N_SEEDS } else { "10" }

if (-not (Test-Path $Python)) {
    Write-Host "== creating venv and installing dependencies =="
    uv venv .venv --python 3.11
    uv pip install -p $Python -r requirements.txt
}

Write-Host "== M1: preparing C-MAPSS data =="
& $Python scripts/prepare_data.py `
    --subset FD001 `
    --out data/processed/cmapss `
    --T 24

Write-Host "== M2/M3: full training (iter_pre=5000, iter_main=10000) on device=$Device =="
& $Python scripts/train.py `
    --config $Config `
    --out $CkptDir `
    --device $Device

Write-Host "== M4: recursive PC sampling (n_samples=$NSamples, n_steps=$NSteps) =="
& $Python scripts/sample.py `
    --config $Config `
    --checkpoint "$CkptDir/ckpt_latest.pt" `
    --n-samples $NSamples `
    --n-steps $NSteps `
    --out outputs/samples/cmapss.npy `
    --device $Device

Write-Host "== M5: evaluation (n_seeds=$NSeeds) =="
& $Python scripts/evaluate.py `
    --config $Config `
    --checkpoint "$CkptDir/ckpt_latest.pt" `
    --n-seeds $NSeeds `
    --device $Device

Write-Host "== done. reports in outputs/reports/ =="
