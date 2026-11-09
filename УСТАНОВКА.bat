@echo off
rem Installing dependencies via Python (see install.py)
where py >nul 2>nul
if not errorlevel 1 (
    py -3 "%~dp0install.py"
    pause
    exit /b 0
)
where python >nul 2>nul
if not errorlevel 1 (
    python "%~dp0install.py"
    pause
    exit /b 0
)
echo [X] Python not found. Install it from python.org (Add to PATH).
pause
