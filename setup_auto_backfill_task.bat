@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Virtual environment not found. Install dependencies first.
  exit /b 1
)
if not exist "logs" mkdir logs
".venv\Scripts\python.exe" auto_backfill.py
exit /b %errorlevel%
