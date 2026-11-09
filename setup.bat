@echo off
rem One-step setup + run on a new Windows machine. Double-click it.
rem Uses whatever Python the py launcher finds (3.12+ is fine).
cd /d "%~dp0"
py -m pip install -r requirements.txt
if errorlevel 1 (
  echo.
  echo pip install failed - read the error above.
  pause
  exit /b 1
)
if not exist .env (
  copy .env.example .env >nul
  echo Created .env - set MCP_AUTH_TOKEN in it, save, close Notepad, then run setup.bat again.
  notepad .env
  exit /b 0
)
py server.py
pause
