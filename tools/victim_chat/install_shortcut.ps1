# Creates a Desktop shortcut for Victim Chat.
#
# Targets pythonw.exe, not python.exe, so launching it opens the window with no
# console behind it — the thing that makes it feel like an installed app rather
# than a script. Errors still surface: the model-load failure path is shown in
# the window itself, and app.py logs to logs/app.log when there is no console.

$ErrorActionPreference = 'Stop'

$root    = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)   # repo root
$pythonw = Join-Path $root '.venv\Scripts\pythonw.exe'
$script  = Join-Path $PSScriptRoot 'app.py'
$icon    = Join-Path $PSScriptRoot 'victim_chat.ico'

foreach ($required in @($pythonw, $script)) {
    if (-not (Test-Path $required)) {
        Write-Error "Missing $required - run this from the project folder."
    }
}

# GetFolderPath, not "$env:USERPROFILE\Desktop": a OneDrive-redirected Desktop
# lives somewhere else entirely and the hardcoded path silently creates a
# shortcut nobody ever sees.
$desktop  = [Environment]::GetFolderPath('Desktop')
$linkPath = Join-Path $desktop 'Victim Chat.lnk'

$shell = New-Object -ComObject WScript.Shell
$link  = $shell.CreateShortcut($linkPath)
$link.TargetPath       = $pythonw
$link.Arguments        = '"' + $script + '"'
$link.WorkingDirectory = $root
$link.Description      = 'Chat with the local Qwen2.5-VL victim model'
$link.WindowStyle      = 1
if (Test-Path $icon) { $link.IconLocation = $icon }
$link.Save()

Write-Host ""
Write-Host "  Shortcut created:" -ForegroundColor Green
Write-Host "  $linkPath"
Write-Host ""
Write-Host "  Double-click it to start Victim Chat. To keep it handy, right-click"
Write-Host "  the shortcut and choose 'Pin to Start' or 'Pin to taskbar'."
Write-Host ""
