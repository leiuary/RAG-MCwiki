@echo off
setlocal

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\dev-down.ps1"

if errorlevel 1 (
  echo.
  echo Stop failed.
  pause
)

endlocal
