@echo off
setlocal
cd /d "%~dp0"

echo Downloading the latest official yt-dlp for Windows...
powershell.exe -NoProfile -ExecutionPolicy Bypass -Command "$ProgressPreference='SilentlyContinue'; Invoke-WebRequest -UseBasicParsing -Uri 'https://github.com/yt-dlp/yt-dlp/releases/latest/download/yt-dlp.exe' -OutFile 'yt-dlp.exe'"
if errorlevel 1 (
  echo.
  echo Download failed. Check the internet connection and try again.
  pause
  exit /b 1
)

echo.
echo Installed version:
"%~dp0yt-dlp.exe" --version
echo.
echo yt-dlp is ready. You can now run run_backfill.bat.
pause
