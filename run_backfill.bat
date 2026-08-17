@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Virtual environment not found. Run setup_windows.bat first.
  exit /b 1
)
if not exist "logs" mkdir logs
".venv\Scripts\python.exe" main.py >> "logs\latest.log" 2>&1
exit /b %errorlevel%

