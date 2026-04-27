[CmdletBinding()]
param(
  [switch]$SkipBackend,
  [switch]$SkipFrontend,
  [switch]$NoInstall,
  [switch]$DryRun
)

$ErrorActionPreference = "Stop"

$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$frontendDir = Join-Path $root "frontend"
$venvActivate = Join-Path $root ".venv\Scripts\Activate.ps1"

if (-not (Test-Path $frontendDir)) {
  throw "找不到前端目录: $frontendDir"
}

$backendCommand = @"
Set-Location '$root'
if (Test-Path '$venvActivate') { . '$venvActivate' }
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
"@

$frontendLines = @("Set-Location '$frontendDir'")
if (-not $NoInstall) {
  $frontendLines += "if (-not (Test-Path 'node_modules')) { npm install }"
}
$frontendLines += "npm run dev"
$frontendCommand = $frontendLines -join "`n"

if ($DryRun) {
  if (-not $SkipBackend) {
    Write-Host "[DryRun] Backend command:"
    Write-Host $backendCommand
  }
  if (-not $SkipFrontend) {
    Write-Host "[DryRun] Frontend command:"
    Write-Host $frontendCommand
  }
  return
}

if (-not $SkipBackend) {
  Start-Process powershell -ArgumentList "-NoExit", "-ExecutionPolicy", "Bypass", "-Command", $backendCommand
  Write-Host "Backend started at http://localhost:8000"
}

if (-not $SkipFrontend) {
  Start-Process powershell -ArgumentList "-NoExit", "-ExecutionPolicy", "Bypass", "-Command", $frontendCommand
  Write-Host "Frontend started at http://localhost:3000"
}

Write-Host "Done."
