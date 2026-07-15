@echo off
:: Force the terminal execution engine to parse strings using UTF-8 encoding
chcp 65001 >nul
setlocal enabledelayedexpansion

:: Force set a completely unique title for this terminal window session
title BLUR_PIPE_MAIN_WINDOW

:: Set working directory immediately to the script's location
cd /d "%~dp0"

:: ===================================================
::  PRE-FLIGHT CLEANUP: Nuke stale environment states
:: ===================================================
if exist .env (
    echo Cleaning up stale .env file from previous runs...
    del /f /q .env
)

:: Check if the Docker daemon is responding
docker info >nul 2>&1
if %errorlevel% neq 0 (
    echo Docker Desktop is not running. Launching engine automatically...
    
    start "" "C:\Program Files\Docker\Docker\Docker Desktop.exe"
    
    echo Waiting for Docker daemon to fully initialize...
    echo This may take up to 20-30 seconds depending on system drive speed.
    echo.
    
    :WAIT_FOR_DOCKER
    timeout /t 3 /nobreak >nul
    docker info >nul 2>&1
    if %errorlevel% neq 0 (
        set /a count+=1
        echo . [Attempt !count!] Still initializing...
        if !count! geq 15 (
            color 0C
            echo.
            echo ERROR: Docker Desktop timed out or failed to start up.
            echo Please open Docker Desktop manually and try again.
            pause
            exit /b
        )
        goto :WAIT_FOR_DOCKER
    )
    echo Docker engine successfully initialized!
    echo.
)

echo ===================================================
echo     DRIVE AND FOLDER CONFIGURATION TOOL
echo ===================================================
echo.

echo Is your data drive external or internal?
echo [E] External USB Device (e.g., Portable SSD)
echo [I] Internal Local Drive (e.g., Local C or D Drive)
echo.
choice /c EI /n /m "Press E or I: "

if errorlevel 2 (set "IS_EXTERNAL=N") else (set "IS_EXTERNAL=Y")

echo.
echo Launching Folder Browser... Please select your INPUT folder.
echo ---------------------------------------------------

set "CHOOSER_CODE=[Console]::OutputEncoding = [System.Text.Encoding]::UTF8; Add-Type -AssemblyName System.Windows.Forms; $f = New-Object System.Windows.Forms.FolderBrowserDialog; $f.Description = 'Select Input Directory'; $f.ShowNewFolderButton = $true; if($f.ShowDialog() -eq 'OK'){write-host $f.SelectedPath}"
for /f "usebackq tokens=*" %%A in (`powershell -NoProfile -Command "%CHOOSER_CODE%"`) do set "RAW_INPUT_PATH=%%A"

if "%RAW_INPUT_PATH%"=="" (
    color 0C
    echo ERROR: No input folder was selected. Exiting configuration.
    pause
    exit /b
)

set "DRIVE=%RAW_INPUT_PATH:~0,1%"

if "%IS_EXTERNAL%"=="Y" (
    set "RAW_OUTPUT_PATH=%DRIVE%:\pipeline_outputs"
    set "FOLDER_TRAIL=%RAW_INPUT_PATH:~3%"
    if not "!FOLDER_TRAIL!"=="" (
        set "RAW_OUTPUT_PATH=!RAW_OUTPUT_PATH!\!FOLDER_TRAIL!"
    )
) else (
    set "RAW_OUTPUT_PATH=%RAW_INPUT_PATH%\pipeline_outputs"
)

:: --- CRITICAL PATH SANITIZATION ---
:: Convert backslashes to forward slashes for Docker compatibility
set "SAFE_INPUT_PATH=!RAW_INPUT_PATH:\=/!"
set "SAFE_OUTPUT_PATH=!RAW_OUTPUT_PATH:\=/!"

echo ---------------------------------------------------
echo Configuration Summary:
echo   Target Drive : %DRIVE%:\
echo   Input Folder : %SAFE_INPUT_PATH%
echo   Output Folder: %SAFE_OUTPUT_PATH%
echo ---------------------------------------------------
echo.

if not exist "%RAW_INPUT_PATH%" goto :ERROR_MISSING
if not exist "%RAW_OUTPUT_PATH%" mkdir "%RAW_OUTPUT_PATH%"

echo Preparing paths for Docker Desktop...
goto :LAUNCH_DOCKER


:LAUNCH_DOCKER
echo Starting Blur Detection Pipeline Container...

:: Write sanitized paths directly to the fresh .env file
(
  echo DOCKER_INPUT_PATH=!SAFE_INPUT_PATH!
  echo DOCKER_OUTPUT_PATH=!SAFE_OUTPUT_PATH!
) > .env

:: Get current CMD process ID (PID)
for /f "usebackq tokens=*" %%A in (`powershell -NoProfile -Command "$parent_id = (Get-WmiObject Win32_Process | Where-Object {$_.ProcessID -eq $PID}).ParentProcessId; (Get-WmiObject Win32_Process | Where-Object {$_.ProcessID -eq $parent_id}).ParentProcessId"`) do set "MY_PID=%%A"


:: DETACHED CLOSE MONITOR (Watches CMD process ID to automatically run docker compose down on exit)
if not "!MY_PID!"=="" (
    start "" powershell -WindowStyle Hidden -NoProfile -Command "$pid_to_watch = !MY_PID!; while ($true) { Start-Sleep -Seconds 1; $proc = Get-Process -Id $pid_to_watch -ErrorAction SilentlyContinue; if (-not $proc) { Start-Process cmd.exe -ArgumentList '/c docker compose down' -WorkingDirectory '!CD!' -WindowStyle Hidden; break } }" >nul 2>&1
)

:: Force remove stale container and prune dangling networks to clear mount conflicts
echo Cleaning up stale Docker state...
docker rm -f blur_processor >nul 2>&1
docker network prune -f >nul 2>&1

:: Spin up the stack in background (detached)
docker compose up -d

timeout /t 2 /nobreak >nul
start "" "http://localhost:8501"

echo Starting Blur Detection Pipeline Container...
echo.
echo ---------------------------------------------------
echo Pipeline is running! Interface opened at:
echo http://localhost:8501
echo ---------------------------------------------------
echo WARNING: Closing this terminal window/tab or pressing
echo any key here will automatically STOP the Docker container.
echo ---------------------------------------------------
echo.
pause

echo.
echo Stopping Docker pipeline cleanly...
docker compose down

if exist .env del /f /q .env

echo.
echo You may now safely close this window.
goto :EOF

:ERROR_MISSING
color 0C
echo ERROR: Specified directory path could not be verified.
echo Please check your hardware connections or drive paths and try again.
echo.
pause
color 07
exit