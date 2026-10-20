# GeoSense-X -- BEN-14K Evaluation Launcher (PowerShell)
# Activates .venv311 and runs the evaluation with CUDA.
#
# Usage:
#   .\run_eval.ps1                         (full evaluation)
#   .\run_eval.ps1 --max-samples 500       (quick smoke test)
#   .\run_eval.ps1 --batch-size 16         (lower VRAM batch)

$projectDir = $PSScriptRoot
$venvPython = Join-Path $projectDir ".venv311\Scripts\python.exe"

if (-not (Test-Path $venvPython)) {
    Write-Error "[ERROR] Virtual environment not found at $venvPython"
    Write-Error "Create it: python -m venv .venv311"
    exit 1
}

$env:KMP_DUPLICATE_LIB_OK = "TRUE"
$env:PYTHONIOENCODING = "utf-8"
$env:_GEOSENSE_VENV_ACTIVE = "1"

Write-Host "[GeoSense-X] Python: $venvPython"
& $venvPython "$projectDir\ml\evaluate_ben14k.py" --device cuda --batch-size 8 @args
exit $LASTEXITCODE
