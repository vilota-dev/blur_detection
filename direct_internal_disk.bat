@echo off
:: Force the terminal execution engine to parse strings using UTF-8 encoding
chcp 65001 >nul
setlocal enabledelayedexpansion

:: Force set a completely unique title for this terminal window session
title BLUR_PIPE_MAIN_WINDOW

:: =========================================================================
:: 🛠️ DIRECT CONFIGURATION ZONE (CHANGE YOUR PATHS HERE)
:: =========================================================================
:: Set to 'N' because you are running on an Internal Local Drive
set "IS_EXTERNAL=N"

:: Change this to your exact local storage directory containing your images
set "RAW_INPUT_PATH=C:\Users\Administrator\vilota\600D"
:: =========================================================================

echo ===================================================
echo    DIRECT BLUR PIPELINE STARTUP ENGINE (INTERNAL)
echo ===================================================
echo.

:: Automatically extract the drive letter and convert it to lowercase for Docker formatting
set "DRIVE=%RAW_INPUT_PATH:~0,1%"
for %%L in (a b c d e f g h i j k l m n o p q r s t u v w x y z) do (
    if /i "%DRIVE%"=="%%L" set "DRIVE_LOWER=%%L"
)

:: Calculate Output Paths based on drive selection type
if "%IS_EXTERNAL%"=="Y" (
    set "RAW_OUTPUT_PATH=%DRIVE%:\pipeline_outputs"
    set "FOLDER_TRAIL=%RAW_INPUT_PATH:~3%"
    if not "!FOLDER_TRAIL!"=="" (
        set "RAW_OUTPUT_PATH=!RAW_OUTPUT_PATH!\!FOLDER_TRAIL!"
    )
) else (
    set "RAW_OUTPUT_PATH=%RAW_INPUT_PATH%"
)

:PROCESS_PATHS
if "%IS_EXTERNAL%"=="Y" (
    set "TRAILING_PATH=%RAW_INPUT_PATH:~3%"
    set "TRAILING_PATH=!TRAILING_PATH:\=/!"
    set "DOCKER_INPUT_PATH=/mnt/wsl/external_d/!TRAILING_PATH!"
    
    set "OUT_TRAILING_PATH=%RAW_OUTPUT_PATH:~3%"
    set "OUT_TRAILING_PATH=!OUT_TRAILING_PATH:\=/!"
    set "DOCKER_OUTPUT_PATH=/mnt/wsl/external_d/!OUT_TRAILING_PATH!"
) else (
    set "TRAILING_PATH=%RAW_INPUT_PATH:~3%"
    set "TRAILING_PATH=!TRAILING_PATH:\=/!"
    set "DOCKER_INPUT_PATH=/!DRIVE_LOWER!/!TRAILING_PATH!"
    
    set "OUT_TRAILING_PATH=%RAW_OUTPUT_PATH:~3%"
    set "OUT_TRAILING_PATH=!OUT_TRAILING_PATH:\=/!"
    set "DOCKER_OUTPUT_PATH=/!DRIVE_LOWER!/!OUT_TRAILING_PATH!"
)

:: Export variables directly to Docker Compose engine environment block
set "DOCKER_INPUT_PATH=%DOCKER_INPUT_PATH%"
set "DOCKER_OUTPUT_PATH=%DOCKER_OUTPUT_PATH%"

echo ---------------------------------------------------
echo Execution Summary:
echo   Target Drive : %DRIVE%:\
echo   Input Folder : %RAW_INPUT_PATH%
echo   Output Folder: %RAW_OUTPUT_PATH%
echo ---------------------------------------------------
echo.

if not exist "%RAW_INPUT_PATH%" goto :ERROR_MISSING
if not exist "%RAW_OUTPUT_PATH%" mkdir "%RAW_OUTPUT_PATH%"

if "%IS_EXTERNAL%"=="Y" (
    echo Mounting %DRIVE%:\ to Docker WSL Kernel with UTF-8...
    wsl -d docker-desktop mkdir -p /mnt/wsl/external_d
    wsl -d docker-desktop mount -t drvfs %DRIVE%: /mnt/wsl/external_d -o codepage=936,iocharset=utf8,metadata
) else (
    echo Internal Drive Context Verified. Skipping WSL mount layer...
)

echo Starting Blur Detection Pipeline Container...
cd /d "%~dp0"
docker compose up -d >nul 2>&1

:: DETACHED CLOSE MONITOR: Kills container immediately if window is clicked shut.
start /b "" powershell -NoProfile -Command "$currentTitle = 'BLUR_PIPE_MAIN_WINDOW'; while ($true) { Start-Sleep -Seconds 1; $proc = Get-Process | Where-Object { $_.MainWindowTitle -eq $currentTitle }; if (-not $proc) { Start-Process cmd.exe -ArgumentList '/c docker stop blur_processor && docker rm blur_processor' -WindowStyle Hidden; break } }" >nul 2>&1

timeout /t 2 /nobreak >nul
start "" "http://localhost:8501"

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

echo.
echo Stopping Docker pipeline cleanly...

:: DETACHED MANUAL EXIT: Spins off a background cleanup process thread
start /b "" cmd /c "docker stop blur_processor >nul 2>&1 && docker rm blur_processor >nul 2>&1"

:: CLEAN AUTO-CLOSE: End the script context parameter tree cleanly
goto :EOF

:ERROR_MISSING
color 0C
echo ERROR: Specified directory path could not be verified.
echo Please check your file paths configuration and try again.
echo.
pause
color 07
exit


:: Using terminal
:: $env:DOCKER_INPUT_PATH="/c/Users/Vilota/mingjie/600D/IMG"; $env:DOCKER_OUTPUT_PATH="/c/Users/Vilota/mingjie/output/600D"; docker compose up -d