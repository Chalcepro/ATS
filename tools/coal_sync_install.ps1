# Start coal_sync.ps1 at every logon (a shortcut in your Startup folder) and start it now.
# To stop it for good: delete "Coal Sync" from shell:startup.
$ps = "$env:WINDIR\System32\WindowsPowerShell\v1.0\powershell.exe"
$args = "-NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -File `"$PSScriptRoot\coal_sync.ps1`""
$lnk = Join-Path ([Environment]::GetFolderPath("Startup")) "Coal Sync.lnk"
$s = (New-Object -ComObject WScript.Shell).CreateShortcut($lnk)
$s.TargetPath = $ps; $s.Arguments = $args; $s.WindowStyle = 7; $s.Description = "Copies the newest Coal to the phone when it is plugged in"
$s.Save()
Start-Process $ps -ArgumentList $args -WindowStyle Hidden
"Installed: $lnk"
