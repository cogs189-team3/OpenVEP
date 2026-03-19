@echo off
cd /d "%~dp0"
if exist pyenv\Scripts\activate.bat (
    call pyenv\Scripts\activate.bat
)
python preview_keyboards.py
pause
