@echo off
setlocal

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\dev-up.ps1"

if errorlevel 1 (
  echo.
  echo Startup failed.
  pause
)

endlocal
