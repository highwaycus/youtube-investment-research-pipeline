@echo off
setlocal
cd /d "%~dp0"
where py >nul 2>nul || goto :no_python
py -3.11 -c "import sys" >nul 2>nul
if not errorlevel 1 (
  set "PY_CMD=py -3.11"
) else (
  py -3.9 -c "import sys" >nul 2>nul || goto :no_python
  set "PY_CMD=py -3.9"
)
%PY_CMD% -m venv .venv || exit /b 1
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip || exit /b 1
python -m pip install -r requirements.txt || exit /b 1
if not exist ".env" copy ".env.example" ".env"
echo.
echo Setup complete. Edit .env, then run run.bat.
exit /b 0

:no_python
echo A working Python 3.11 or 3.9 installation was not found.
exit /b 1
