# Full-scale TSGM C-MAPSS run: data prep -> train -> recursive PC sampling (n_steps=1000)
# -> 10-seed discriminative/predictive/t-SNE evaluation. Iteration counts, score-net type,
# SDE, etc. all come from configs/cmapss.yaml. See docs/reproduction_plan.md.

$ErrorActionPreference = "Stop"

Set-Location (Join-Path $PSScriptRoot "..")

$Python = ".venv\Scripts\python.exe"
$Config = "configs/cmapss.yaml"
$CkptDir = "outputs/checkpoints/cmapss_full"
$NSamples = if ($env:N_SAMPLES) { $env:N_SAMPLES } else { "100" }
$NSteps = if ($env:N_STEPS) { $env:N_STEPS } else { "1000" }
$NSeeds = if ($env:N_SEEDS) { $env:N_SEEDS } else { "10" }

if (-not (Test-Path $Python)) {
    Write-Host "== creating venv and installing dependencies =="
    uv venv .venv --python 3.11
    uv pip install -p $Python -r requirements.txt
}

$Device = if ($env:DEVICE) { $env:DEVICE } else { (& $Python -c "from scripts.config_utils import get_default_device; print(get_default_device())").Trim() }

Write-Host "== M1: preparing C-MAPSS data =="
& $Python scripts/prepare_data.py `
    --subset FD001 `
    --out data/processed/cmapss `
    --T 24

Write-Host "== M2/M3: full training (counts from $Config) on device=$Device =="
& $Python scripts/train.py `
    --config $Config `
    --out $CkptDir `
    --device $Device

# Prefer the best-by-val checkpoint if selection produced one, else the latest.
$Ckpt = if (Test-Path "$CkptDir/ckpt_best.pt") { "$CkptDir/ckpt_best.pt" } else { "$CkptDir/ckpt_latest.pt" }
Write-Host "== using checkpoint $Ckpt =="

Write-Host "== M4: recursive PC sampling (n_samples=$NSamples, n_steps=$NSteps) =="
& $Python scripts/sample.py `
    --config $Config `
    --checkpoint $Ckpt `
    --n-samples $NSamples `
    --n-steps $NSteps `
    --out outputs/samples/cmapss.npy `
    --device $Device

Write-Host "== M5: evaluation (n_seeds=$NSeeds) =="
& $Python scripts/evaluate.py `
    --config $Config `
    --checkpoint $Ckpt `
    --n-seeds $NSeeds `
    --device $Device

Write-Host "== done. reports in outputs/reports/ =="
