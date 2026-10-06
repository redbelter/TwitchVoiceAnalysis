@echo off
setlocal
cd /d "%~dp0"
set "PY=python"
if exist ".venv\Scripts\python.exe" set "PY=.venv\Scripts\python.exe"
if exist "%~dp0vodpipe.local.bat" call "%~dp0vodpipe.local.bat"

if "%~1"=="" goto ask
set "TARGET=%~1"
goto run

:ask
echo   - paste a Twitch VOD/clip URL
echo   - or type a channel login to queue EVERYTHING (live + all VODs)
set /p TARGET=Twitch URL or channel login (or drag a video file onto this .bat): 
if "%TARGET%"=="" exit /b 1

:run
echo ====================================================
echo  VODPipe: %TARGET%
echo  Safe to close this window? NO — analysis dies too.
echo  Resume anytime: re-run the same command / re-drop.
echo ====================================================
"%PY%" vodpipe.py "%TARGET%"
echo.
pause
