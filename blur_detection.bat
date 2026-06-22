@echo off
:: Force the terminal execution engine to parse strings using UTF-8 encoding
chcp 65001 >nul
setlocal enabledelayedexpansion

:: Force set a completely unique title for this terminal window session
title BLUR_PIPE_MAIN_WINDOW

:: Check if the Docker daemon is responding
docker info >nul 2>&1
if %errorlevel% neq 0 (
    echo Docker Desktop is not running. Launching engine automatically...
    
    :: Start Docker Desktop using its default Windows installation path
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

:: 1. Ask straight away whether the target drive is External or Internal
echo Is your data drive external or internal?
echo [E] External USB Device (e.g., Portable SSD)
echo [I] Internal Local Drive (e.g., Local C or D Drive)
echo.
choice /c EI /n /m "Press E or I: "

if errorlevel 2 (set "IS_EXTERNAL=N") else (set "IS_EXTERNAL=Y")

echo.
echo Launching Folder Browser... Please select your INPUT folder.
echo ---------------------------------------------------

:: 2. Launch the native Windows folder visual browser using explicit UTF-8 output parsing
set "CHOOSER_CODE=[Console]::OutputEncoding = [System.Text.Encoding]::UTF8; Add-Type -AssemblyName System.Windows.Forms; $f = New-Object System.Windows.Forms.FolderBrowserDialog; $f.Description = 'Select Input Directory'; $f.ShowNewFolderButton = $true; if($f.ShowDialog() -eq 'OK'){write-host $f.SelectedPath}"
for /f "usebackq tokens=*" %%A in (`powershell -NoProfile -Command "%CHOOSER_CODE%"`) do set "RAW_INPUT_PATH=%%A"

if "%RAW_INPUT_PATH%"=="" (
    color 0C
    echo ERROR: No input folder was selected. Exiting configuration.
    pause
    exit /b
)

:: 3. Extract the drive letter and convert it to lowercase for Docker formatting
set "DRIVE=%RAW_INPUT_PATH:~0,1%"
for %%L in (a b c d e f g h i j k l m n o p q r s t u v w x y z) do (
    if /i "%DRIVE%"=="%%L" set "DRIVE_LOWER=%%L"
)

:: 4. Calculate Output Paths based on drive type
if "%IS_EXTERNAL%"=="Y" (
    :: External Drive: Keep the automated /pipeline_outputs mapping structure
    set "RAW_OUTPUT_PATH=%DRIVE%:\pipeline_outputs"
    set "FOLDER_TRAIL=%RAW_INPUT_PATH:~3%"
    if not "!FOLDER_TRAIL!"=="" (
        set "RAW_OUTPUT_PATH=!RAW_OUTPUT_PATH!\!FOLDER_TRAIL!"
    )
) else (
    :: Internal Drive: Output directly into the exact same folder as the input directory
    set "RAW_OUTPUT_PATH=%RAW_INPUT_PATH%\pipeline_outputs"
)

:PROCESS_PATHS
:: Cross-platform native forward-slash layouts for Docker Desktop volume mounting stability
set "TRAILING_PATH=%RAW_INPUT_PATH:~3%"
set "TRAILING_PATH=!TRAILING_PATH:\=/!"
set "DOCKER_INPUT_PATH=%DRIVE_LOWER%:/!TRAILING_PATH!"

set "OUT_TRAILING_PATH=%RAW_OUTPUT_PATH:~3%"
set "OUT_TRAILING_PATH=!OUT_TRAILING_PATH:\=/!"
set "DOCKER_OUTPUT_PATH=%DRIVE_LOWER%:/!OUT_TRAILING_PATH!"

goto :DO_MOUNT_CHECK

:DO_MOUNT_CHECK
echo ---------------------------------------------------
echo Configuration Summary:
echo   Target Drive : %DRIVE%:\
echo   Input Folder : %RAW_INPUT_PATH%
echo   Output Folder: %RAW_OUTPUT_PATH%
echo ---------------------------------------------------
echo.

if not exist "%RAW_INPUT_PATH%" goto :ERROR_MISSING
if not exist "%RAW_OUTPUT_PATH%" mkdir "%RAW_OUTPUT_PATH%"

echo Preparing paths for Docker Desktop...
goto :LAUNCH_DOCKER


:LAUNCH_DOCKER
echo Starting Blur Detection Pipeline Container...
cd /d "%~dp0"

:: Step 1: Write variables directly to a local file cleanly without trailing spaces
(
  echo DOCKER_INPUT_PATH=%DOCKER_INPUT_PATH%
  echo DOCKER_OUTPUT_PATH=%DOCKER_OUTPUT_PATH%
) > .env

:: Step 2: [FIXED] Force clean up any leftover containers first to prevent mounting/port conflicts
docker compose down >nul 2>&1

:: Step 3: Spin up the entire stack detached safely
docker compose up -d

:: Step 4: Delete the temporary environment declaration block right away
if exist .env del .env

:: DETACHED CLOSE MONITOR: Kills container immediately if terminal execution window is clicked closed.
start /b "" powershell -NoProfile -Command "$currentTitle = 'BLUR_PIPE_MAIN_WINDOW'; while ($true) { Start-Sleep -Seconds 1; $proc = Get-Process | Where-Object { $_.MainWindowTitle -eq $currentTitle }; if (-not $proc) { Start-Process cmd.exe -ArgumentList '/c docker stop blur_processor && docker rm blur_processor' -WindowStyle Hidden; break } }" >nul 2>&1

timeout /t 2 /nobreak >nul
start "" "http://localhost:8501"

echo.
echo ===================================================
echo     EXTERNAL SSD MOUNT TOOL FOR BLUR PIPELINE
echo ===================================================
echo.
echo Status: SUCCESS
echo Target Folders successfully mapped into pipeline array.
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
echo.

:: [FIXED] Synchronous exit: Block the script until the container is completely stopped and unmounted
docker compose down

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