# Full-scale TSGM ESA run: data prep -> train -> recursive PC sampling (n_steps=1000)
# -> 10-seed discriminative/predictive/t-SNE evaluation. Iteration counts, score-net type,
# SDE, etc. all come from configs/esa.yaml. See docs/reproduction_plan.md.

$ErrorActionPreference = "Stop"

Set-Location (Join-Path $PSScriptRoot "..")

$Python = ".venv\Scripts\python.exe"
$Config = "configs/esa.yaml"
$CkptDir = "outputs/checkpoints/esa_full"
$NSamples = if ($env:N_SAMPLES) { $env:N_SAMPLES } else { "100" }
$NSteps = if ($env:N_STEPS) { $env:N_STEPS } else { "1000" }
$NSeeds = if ($env:N_SEEDS) { $env:N_SEEDS } else { "10" }

if (-not (Test-Path $Python)) {
    Write-Host "== creating venv and installing dependencies =="
    uv venv .venv --python 3.11
    uv pip install -p $Python -r requirements.txt
}

$Device = if ($env:DEVICE) { $env:DEVICE } else { (& $Python -c "from scripts.config_utils import get_default_device; print(get_default_device())").Trim() }

Write-Host "== M7: preparing ESA (Mission1) data =="
& $Python scripts/prepare_data_esa.py `
    --out data/processed/esa `
    --T 24

Write-Host "== training (counts from $Config) on device=$Device =="
& $Python scripts/train.py `
    --config $Config `
    --out $CkptDir `
    --device $Device

$Ckpt = if (Test-Path "$CkptDir/ckpt_best.pt") { "$CkptDir/ckpt_best.pt" } else { "$CkptDir/ckpt_latest.pt" }
Write-Host "== using checkpoint $Ckpt =="

Write-Host "== recursive PC sampling (n_samples=$NSamples, n_steps=$NSteps) =="
& $Python scripts/sample.py `
    --config $Config `
    --checkpoint $Ckpt `
    --n-samples $NSamples `
    --n-steps $NSteps `
    --out outputs/samples/esa.npy `
    --device $Device

Write-Host "== evaluation (n_seeds=$NSeeds) =="
& $Python scripts/evaluate.py `
    --config $Config `
    --checkpoint $Ckpt `
    --n-seeds $NSeeds `
    --device $Device

Write-Host "== done. reports in outputs/reports/ =="
