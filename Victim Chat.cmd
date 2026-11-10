@echo off
REM Launch the local Victim Chat window. The console stays open so model
REM load progress and any CUDA error are visible instead of vanishing.
cd /d "%~dp0"
set HF_HUB_OFFLINE=1
".venv\Scripts\python.exe" app.py
if errorlevel 1 pause
