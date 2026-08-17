$ErrorActionPreference = "Stop"

$TaskName = "YouTube Investment Auto Backfill"
$ProjectDirectory = Split-Path -Parent $MyInvocation.MyCommand.Path
$BatchFile = Join-Path $ProjectDirectory "run_auto_backfill.bat"

if (-not (Test-Path $BatchFile)) {
    throw "run_auto_backfill.bat was not found in $ProjectDirectory"
}

$Action = New-ScheduledTaskAction `
    -Execute "cmd.exe" `
    -Argument "/d /c `"$BatchFile`"" `
    -WorkingDirectory $ProjectDirectory

$Trigger = New-ScheduledTaskTrigger `
    -Once `
    -At (Get-Date).AddHours(1) `
    -RepetitionInterval (New-TimeSpan -Hours 1) `
    -RepetitionDuration (New-TimeSpan -Days 3650)

$Settings = New-ScheduledTaskSettingsSet `
    -StartWhenAvailable `
    -MultipleInstances IgnoreNew `
    -ExecutionTimeLimit (New-TimeSpan -Hours 1)

Register-ScheduledTask `
    -TaskName $TaskName `
    -Action $Action `
    -Trigger $Trigger `
    -Settings $Settings `
    -Description "Safely backfill up to five YouTube investment streams every hour." `
    -Force | Out-Null

Write-Host ""
Write-Host "Scheduled task created: $TaskName"
Write-Host "First run: about one hour from now"
Write-Host "Then: every hour"
Write-Host ""
Write-Host "A YouTube block causes a 24-hour automatic cooldown."
Write-Host "When the backfill is complete, future launches send no requests."
