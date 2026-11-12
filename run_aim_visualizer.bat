@echo off
title 3-RRR 100ms 3200DPI FPS 打靶精度检验可视化平台
cd /d "%~dp0"
echo ====================================================================
echo   启动 3-RRR 打靶精度与 R 曲线离线交互式检验系统 (无需连接硬件)...
echo ====================================================================
"E:\anaconda\envs\pytorch\python.exe" visualize_aim_benchmark.py
pause

