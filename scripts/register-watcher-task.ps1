# Register sync-watcher.py as a Windows scheduled task that starts at logon.
# Run this script as Administrator (right-click PowerShell -> "Run as Administrator").
#
# Usage:
#   .\register-watcher-task.ps1 [-Unregister]

param(
    [switch]$Unregister
)

$TaskName = "watch-sync-watcher"
$ScriptPath = "$env:USERPROFILE\.claude\skills\watch\scripts\sync_watcher.py"
$VenvPython = "$env:USERPROFILE\.claude\skills\watch\.venv\Scripts\python.exe"

if ($Unregister) {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue
    Write-Host "Unregistered task: $TaskName"
    exit 0
}

if (-not (Test-Path $VenvPython)) {
    Write-Error "venv Python not found at: $VenvPython"
    Write-Error "Run: python -m venv $env:USERPROFILE\.claude\skills\watch\.venv && pip install -r requirements.txt"
    exit 1
}

if (-not (Test-Path $ScriptPath)) {
    Write-Error "sync_watcher.py not found at: $ScriptPath"
    exit 1
}

$Action = New-ScheduledTaskAction `
    -Execute $VenvPython `
    -Argument "`"$ScriptPath`""

$Trigger = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME

$Settings = New-ScheduledTaskSettingsSet `
    -StartWhenAvailable `
    -RestartCount 3 `
    -RestartInterval (New-TimeSpan -Minutes 5) `
    -ExecutionTimeLimit ([TimeSpan]::Zero)   # no time limit

$Principal = New-ScheduledTaskPrincipal `
    -UserId $env:USERNAME `
    -LogonType Interactive `
    -RunLevel Limited

Register-ScheduledTask `
    -TaskName $TaskName `
    -Action $Action `
    -Trigger $Trigger `
    -Settings $Settings `
    -Principal $Principal `
    -Force

Write-Host "Registered scheduled task: $TaskName"
Write-Host "Starting task now..."

Start-ScheduledTask -TaskName $TaskName
Start-Sleep -Seconds 2

$Status = (Get-ScheduledTask -TaskName $TaskName).State
Write-Host "Task state: $Status"

if ($Status -eq "Running") {
    Write-Host "sync-watcher is running. It will auto-start at every future logon."
} else {
    Write-Warning "Task may not have started. Check Event Viewer > Task Scheduler for errors."
}
