@echo off
cd /d "%~dp0"
echo [3-RRR] Starting Screen Manual Pad...
"E:\anaconda\envs\pytorch\python.exe" screen_manual_pad.py
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Screen Manual Pad exited with error code %ERRORLEVEL%
    pause
)
