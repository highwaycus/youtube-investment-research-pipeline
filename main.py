@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Virtual environment not found. Install dependencies first.
  exit /b 1
)
".venv\Scripts\python.exe" backfill_status.py
exit /b %errorlevel%
