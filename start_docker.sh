#!/bin/bash

# Ensure script stops on unhandled errors
set -e

# Clear screen and set terminal title (if supported by terminal emulator)
clear
echo -ne "\033]0;BLUR_PIPE_MAIN_WINDOW\007"

# --- STEP 1: Check Docker Daemon Status ---
if ! docker info >/dev/null 2>&1; then
    echo "Docker engine is not running. Attempting to start docker service..."
    sudo systemctl start docker
    
    # Wait and check if it initialized successfully
    count=0
    while ! docker info >/dev/null 2>&1; do
        sleep 2
        ((count++))
        echo ". [Attempt $count] Still initializing Docker engine..."
        if [ $count -ge 10 ]; then
            echo -e "\n\033[0;31mERROR: Docker service timed out or failed to start.\033[0m"
            echo "Please start the Docker daemon manually (e.g., 'sudo systemctl start docker') and retry."
            exit 1
        fi
    done
    echo -e "Docker engine successfully initialized!\n"
fi

echo "==================================================="
echo "     DRIVE AND FOLDER CONFIGURATION TOOL (LINUX) "
echo "==================================================="
echo ""

# --- STEP 2: Determine Drive Association ---
echo "Is your data drive external or internal?"
echo "[E] External USB Device (e.g., Mounted SSD under /media/ or /mnt/)"
echo "[I] Internal Local Drive (e.g., Home directory space)"
echo ""

while true; do
    read -p "Press E or I: " -n 1 -r choice
    echo ""
    case "$choice" in 
        [Ee]* ) IS_EXTERNAL="Y"; break;;
        [Ii]* ) IS_EXTERNAL="N"; break;;
        * ) echo "Invalid choice. Please press E or I.";;
    esac
done

echo -e "\n---------------------------------------------------"
echo "Select your INPUT folder."
echo "---------------------------------------------------"

# Attempt to use visual folder picker (zenity) if available, otherwise fall back to manual terminal text input
if command -v zenity >/dev/null 2>&1 && [ -n "$DISPLAY" ]; then
    RAW_INPUT_PATH=$(zenity --file-selection --directory --title="Select Input Directory" 2>/dev/null)
else
    echo "Type (or drag & drop) the absolute path to your input folder, then hit Enter:"
    read -r RAW_INPUT_PATH
fi

# Clean up trailing slashes and spaces if any
RAW_INPUT_PATH="${RAW_INPUT_PATH%/}"

if [ -z "$RAW_INPUT_PATH" ] || [ ! -d "$RAW_INPUT_PATH" ]; then
    echo -e "\n\033[0;31mERROR: Invalid or empty folder path selected. Exiting config.\033[0m"
    exit 1
fi

# --- STEP 3: Structural Path Calculations ---
# Linux absolute paths mount directly into Docker without drive-letter conversions
DOCKER_INPUT_PATH="$RAW_INPUT_PATH"

if [ "$IS_EXTERNAL" = "Y" ]; then
    # External Drive Layout Tracker
    # Extracts the mount point root (e.g., /media/vilota/YOUR_SSD) to separate it from the file trail
    if [[ "$RAW_INPUT_PATH" =~ ^(/media/[^/]+/[^/]+|/mnt/[^/]+) ]]; then
        MOUNT_ROOT="${BASH_REMATCH[1]}"
        FOLDER_TRAIL="${RAW_INPUT_PATH#$MOUNT_ROOT}"
        RAW_OUTPUT_PATH="$MOUNT_ROOT/pipeline_outputs${FOLDER_TRAIL}"
    else
        # Fallback if mounted outside standard hubs
        RAW_OUTPUT_PATH="$RAW_INPUT_PATH/pipeline_outputs"
    fi
else
    # Internal Drive Layout
    RAW_OUTPUT_PATH="$RAW_INPUT_PATH/pipeline_outputs"
fi

DOCKER_OUTPUT_PATH="$RAW_OUTPUT_PATH"

# Ensure runtime target folders exist
mkdir -p "$RAW_OUTPUT_PATH"

echo "---------------------------------------------------"
echo "Configuration Summary:"
echo "  Input Folder : $RAW_INPUT_PATH"
echo "  Output Folder: $RAW_OUTPUT_PATH"
echo "---------------------------------------------------"
echo ""

# --- STEP 4: Setup Cleanup Traps (The Window Close/Exit Handler) ---
cleanup() {
    echo -e "\n\nStopping Docker pipeline cleanly..."
    docker compose down
    if [ -f .env ]; then rm -f .env; fi
    echo "Pipeline safely terminated."
    exit
}
# Trap terminal termination flags, window closes, and Ctrl+C key events
trap cleanup EXIT SIGINT SIGTERM

# --- STEP 5: Launch Pipeline Container Stack ---
echo "Preparing paths for Docker Desktop / Core Engine..."
cd "$(dirname "$0")"

# Write operational variables to runtime configuration file
cat << EOF > .env
DOCKER_INPUT_PATH=$DOCKER_INPUT_PATH
DOCKER_OUTPUT_PATH=$DOCKER_OUTPUT_PATH
EOF

echo "Starting Blur Detection Pipeline Container..."
# Clear conflicting residue blocks
docker compose down >/dev/null 2>&1

# Spin up detached pipeline array
docker compose up -d

# Erase ephemeral configurations safely
rm -f .env

# Automatically trigger visual user interface
sleep 2
if command -v xdg-open >/dev/null 2>&1; then
    xdg-open "http://localhost:8501" >/dev/null 2>&1
fi

echo ""
echo "==================================================="
echo "     EXTERNAL SSD MOUNT TOOL FOR BLUR PIPELINE"
echo "==================================================="
echo ""
echo "Status: SUCCESS"
echo "Target Folders successfully mapped into pipeline array."
echo ""
echo "---------------------------------------------------"
echo "Pipeline is running! Interface opened at:"
echo "http://localhost:8501"
echo "---------------------------------------------------"
echo ""
echo "WARNING: Pressing Ctrl+C HERE or CLOSING this window"
echo "will automatically STOP the Docker pipeline containers."
echo "---------------------------------------------------"
echo ""

# Block execution process to allow operational run until manually broken out
while true; do
    sleep 1
done