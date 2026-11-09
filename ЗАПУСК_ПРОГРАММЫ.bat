@echo off
set VENV=
if exist "%~dp0.venv\Scripts\pythonw.exe" set VENV=%~dp0.venv
if not defined VENV if exist "%~dp0..\.venv\Scripts\pythonw.exe" set VENV=%~dp0..\.venv
if not defined VENV (
    echo [X] Not installed yet. Run USTANOVKA.bat first.
    pause
    exit /b 1
)
"%VENV%\Scripts\python.exe" -c "import customtkinter, torch, cv2" >nul 2>nul
if errorlevel 1 (
    echo [X] Packages missing. Run USTANOVKA.bat again.
    pause
    exit /b 1
)
start "" "%VENV%\Scripts\pythonw.exe" "%~dp0app_gui.py"
