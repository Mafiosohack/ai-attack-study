@echo off
REM Puts a "Victim Chat" shortcut on the Desktop that launches the app like any
REM other Windows program: no console window, own icon, own taskbar entry.
REM Re-run this if you move the project folder.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\install_shortcut.ps1"
pause
