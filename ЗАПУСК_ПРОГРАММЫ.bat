@echo off
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  echo [x] Python venv not found. Create it first:
  echo     1. python -m venv .venv
  echo     2. .venv\Scriptsctivate
  echo     3. pip install -r requirements.txt
  echo Read README.md section "Kak zapustit" for details.
  pause
  exit /b 1
)
.venv\Scripts\python.exe -c "import customtkinter, torch, cv2" 2>nul
if errorlevel 1 (
  echo [x] Python packages missing. Install:
  echo     .venv\Scripts\pip install -r requirements.txt
  pause
  exit /b 1
)
start "" .venv\Scripts\pythonw.exe app_gui.py
