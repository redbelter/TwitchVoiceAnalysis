@echo off
setlocal
cd /d "%~dp0"

rem ---- per-machine settings (create vodpipe.local.bat next to this file, see README) ----
if exist "%~dp0vodpipe.local.bat" call "%~dp0vodpipe.local.bat"

rem ---- server already up? ----
curl -s -o nul http://127.0.0.1:5001/api/jobs
if %errorlevel%==0 goto open

echo Starting VODPipe server (minimized window = live log)...
start "VODPipe server" /min python vodpipe_web.py --port 5001
timeout /t 4 /nobreak >nul

:open
start "" http://127.0.0.1:5001
exit /b
