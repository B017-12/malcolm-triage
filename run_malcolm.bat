@echo off
cd /d "%~dp0"
if exist "MalcolmLauncher.exe" (
    "MalcolmLauncher.exe"
) else (
    echo [ERROR] MalcolmLauncher.exe not found.
    pause
)
