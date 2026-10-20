@echo off
:: GeoSense-X — BEN-14K Evaluation Launcher
:: Activates the project venv (.venv311) and runs the evaluation script.
::
:: Usage:
::   run_eval.bat                          (full evaluation, all 14,832 samples)
::   run_eval.bat --max-samples 500        (quick smoke test)
::   run_eval.bat --batch-size 16          (smaller batch for low VRAM)
::
:: All extra arguments are forwarded to ml/evaluate_ben14k.py

setlocal

:: Set UTF-8 output
chcp 65001 >nul 2>&1

:: Locate project root (this script's directory)
set PROJECT_DIR=%~dp0
set VENV_PYTHON=%PROJECT_DIR%.venv311\Scripts\python.exe

:: Check the venv exists
if not exist "%VENV_PYTHON%" (
    echo [ERROR] Virtual environment not found at .venv311\Scripts\python.exe
    echo         Create it with:  python -m venv .venv311
    echo         Then install:    .venv311\Scripts\pip install torch==2.11.0+cu128 ...
    exit /b 1
)

:: Set environment variables for compatibility
set KMP_DUPLICATE_LIB_OK=TRUE
set PYTHONIOENCODING=utf-8

echo [GeoSense-X] Using Python: %VENV_PYTHON%
"%VENV_PYTHON%" "%PROJECT_DIR%ml\evaluate_ben14k.py" --device cuda --batch-size 8 %*
exit /b %ERRORLEVEL%
