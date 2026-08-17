@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Missing .venv. Run setup_windows.bat first.
  pause
  exit /b 1
)
"%~dp0.venv\Scripts\python.exe" "%~dp0build_playbook.py"
if errorlevel 1 pause
