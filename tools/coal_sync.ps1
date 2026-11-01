# Coal to the phone: whenever the phone is plugged in (USB, "File transfer"), the newest Coal is
# copied to its Internal storage\Coal - only if it changed. Laptop -> phone only.
# Started at logon by a shortcut in the Startup folder (tools/coal_sync_install.ps1); one copy runs.
$ErrorActionPreference = "SilentlyContinue"
$made = $false
$one = New-Object System.Threading.Mutex($true, "CoalSync", [ref]$made)
if (-not $made) { exit }

$ats = Split-Path $PSScriptRoot -Parent
$py = "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe"
$log = "$env:APPDATA\CoalSync.log"
$tmp = Join-Path $env:TEMP "coalsync"
$sh = New-Object -ComObject Shell.Application
function Say($m) { Add-Content $log ("{0:yyyy-MM-dd HH:mm}  {1}" -f (Get-Date), $m) }

function Phone {
    # a phone shows under "This PC" as a portable device with a storage folder inside
    foreach ($d in $sh.Namespace(17).Items()) {
        if ($d.Path -like "::*" -and $d.IsFolder) {
            $store = $d.GetFolder.Items() | Where-Object { $_.IsFolder } | Select-Object -First 1
            if ($store) { return @{ name = $d.Name; store = $store.GetFolder } }
        }
    }
}

$seen = $null
while ($true) {
    $p = Phone
    if (-not $p) { $seen = $null }
    elseif ($seen -ne $p.name) {
        $seen = $p.name
        $version = (& $py (Join-Path $ats "tools\pack_coal.py")).Trim()
        $theirs = ""
        $coal = $p.store.Items() | Where-Object { $_.Name -eq "Coal" }
        if ($coal) {
            $v = $coal.GetFolder.Items() | Where-Object { $_.Name -like "coal_version*" }
            if ($v) {
                Remove-Item $tmp -Recurse -Force; New-Item -ItemType Directory $tmp | Out-Null
                $sh.Namespace($tmp).CopyHere($v, 4 + 16 + 1024); Start-Sleep 3
                $theirs = (Get-Content (Join-Path $tmp "coal_version.json") -Raw | ConvertFrom-Json).version
            }
        }
        if ($theirs -eq $version) { Say "$($p.name): Coal already up to date ($version)" }
        else {
            Say "$($p.name): updating Coal $theirs -> $version"
            $p.store.CopyHere((Join-Path $ats "phone_pack\Coal"), 4 + 16 + 512 + 1024)   # no dialogs, yes to all
            Say "$($p.name): copy started (Windows finishes it in the background)"
        }
    }
    Start-Sleep 5
}
