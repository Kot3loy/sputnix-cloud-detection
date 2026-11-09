@echo off
cd /d "%~dp0"
set VENV=.venv
if not exist "%VENV%\Scripts\python.exe" if exist "..\.venv\Scripts\python.exe" set VENV=..\.venv
if not exist "%VENV%\Scripts\python.exe" (
  echo [x] Python venv not found here or in the parent folder.
  echo Create it once:
  echo     1. python -m venv .venv
  echo     2. .venv\Scriptsctivate
  echo     3. pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
  echo     4. pip install -r requirements.txt
  echo See README.md, section "Kak zapustit".
  pause
  exit /b 1
)
"%VENV%\Scripts\python.exe" -c "import customtkinter, torch, cv2" 2>nul
if errorlevel 1 (
  echo [x] Python packages missing. Install:
  echo     "%VENV%\Scripts\pip" install -r requirements.txt
  pause
  exit /b 1
)
start "" "%VENV%\Scripts\pythonw.exe" app_gui.py
