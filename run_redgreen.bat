@echo off
cd /d "%~dp0"
if exist pyenv\Scripts\activate.bat (
    call pyenv\Scripts\activate.bat
    python run_vep_redgreen.py
    pause
) else (
    echo Virtual environment not found. Run: python run_vep_redgreen.py
    python run_vep_redgreen.py
    pause
)
