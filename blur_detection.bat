@echo off
setlocal

echo ===================================================
echo    EXTERNAL SSD MOUNT TOOL FOR BLUR PIPELINE
echo ===================================================
echo.

:: 1. Launch a background PowerShell monitor linked to this specific window instance
powershell -NoProfile -Command "Start-Job -ScriptBlock { $parentPid = %%%%; while ($true) { Start-Sleep -Seconds 1; if (-not (Get-Process -Id $parentPid -ErrorAction SilentlyContinue)) { Set-Location -Path '%~dp0'; docker compose down; break } } }" >nul 2>&1

echo Choose your data drive setup:
echo [1] D:\ Drive - External SSD (Default)
echo [2] D:\ Drive - Internal Local Drive
echo [3] Manual Configuration (Other Drive Letters)
echo.
choice /c 123 /n /m "Press 1, 2, or 3: "

if errorlevel 3 goto :MANUAL_PROMPT
if errorlevel 2 goto :D_INTERNAL
if errorlevel 1 goto :D_EXTERNAL

:D_EXTERNAL
set "DRIVE=D"
goto :DO_EXTERNAL_MOUNT

:D_INTERNAL
set "DRIVE=D"
goto :DO_INTERNAL_RUN

:MANUAL_PROMPT
echo.
echo Enter your drive letter (Just type a single letter like C, E, F):
set /p "DRIVE="
echo.
echo Is this manual drive external or internal?
echo [E] External USB Device
echo [I] Internal Local Drive
echo.
choice /c EI /n /m "Press E or I: "

if errorlevel 2 goto :DO_INTERNAL_RUN
goto :DO_EXTERNAL_MOUNT


:DO_EXTERNAL_MOUNT
echo ---------------------------------------------------
echo Target Configured: %DRIVE%:\ Drive (EXTERNAL)
echo ---------------------------------------------------
if not exist %DRIVE%:\ goto :ERROR_MISSING
echo Mounting %DRIVE%:\ to Docker WSL Kernel with UTF-8...
wsl -d docker-desktop mkdir -p /mnt/wsl/external_d
wsl -d docker-desktop mount -t drvfs %DRIVE%: /mnt/wsl/external_d -o codepage=936,iocharset=utf8,metadata
goto :LAUNCH_DOCKER


:DO_INTERNAL_RUN
echo ---------------------------------------------------
echo Target Configured: %DRIVE%:\ Drive (INTERNAL)
echo ---------------------------------------------------
if not exist %DRIVE%:\ goto :ERROR_MISSING
echo Skipping WSL mount layer...
goto :LAUNCH_DOCKER


:LAUNCH_DOCKER
echo Starting Blur Detection Pipeline Container...
cd /d "%~dp0"
docker compose up -d >nul 2>&1

timeout /t 2 /nobreak >nul
start "" "http://localhost:8501"

cls
echo ===================================================
echo    EXTERNAL SSD MOUNT TOOL FOR BLUR PIPELINE
echo ===================================================
echo.
echo Status: SUCCESS
echo Target: %DRIVE%:\ drive is ready.
echo.
echo ---------------------------------------------------
echo Pipeline is running! Interface opened at:
echo http://localhost:8501
echo ---------------------------------------------------
echo.
echo WARNING: Pressing any key HERE or CLOSING this window 
echo will automatically STOP the Docker pipeline containers.
echo ---------------------------------------------------
echo.
pause

:: Handles normal exit (Pressing any key inside the terminal)
echo.
echo Stopping Docker pipeline cleanly...
docker compose down >nul 2>&1
exit

:ERROR_MISSING
color 0C
echo ERROR: Windows cannot see the %DRIVE%:\ drive. 
echo Please check your connections or drive letter and try again.
echo.
pause
color 07
exit