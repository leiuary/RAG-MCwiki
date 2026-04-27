[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"

function Stop-PortProcess {
  param(
    [Parameter(Mandatory = $true)]
    [int]$Port
  )

  $connections = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
  if (-not $connections) {
    Write-Host "Port $Port has no listening process."
    return
  }

  $processIds = $connections | Select-Object -ExpandProperty OwningProcess -Unique
  foreach ($processId in $processIds) {
    try {
      $proc = Get-Process -Id $processId -ErrorAction Stop
      Stop-Process -Id $processId -Force -ErrorAction Stop
      Write-Host "Stopped $($proc.ProcessName) on port $Port (PID=$processId)"
    } catch {
      Write-Host "Failed to stop PID=$processId on port $Port"
    }
  }
}

Stop-PortProcess -Port 8000
Stop-PortProcess -Port 3000

Write-Host "Done."
