@echo off
title DataDoctor Auto-Backup Service
echo Starting Real-Time Auto-Backup Service for dataDoctor...
echo Keep this window open while you work to ensure your code is backed up every 30 seconds.
echo.
python "%~dp0auto_backup.py"
pause
