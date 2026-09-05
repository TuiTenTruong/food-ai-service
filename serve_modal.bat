@echo off
REM Shortcut to run Modal Dev Server for Food AI Service
REM Tắt khi không cần sử dụng bằng cách nhấn Ctrl + C

set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8

if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" serve_modal.py %*
) else (
    python serve_modal.py %*
)
