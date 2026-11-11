@echo off
cd /d "%~dp0"
echo [3-RRR] Starting Full-Screen Aim Trainer GUI...
"E:\anaconda\envs\pytorch\python.exe" fps_aim_trainer.py
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Aim Trainer exited with error code %ERRORLEVEL%
    pause
)
