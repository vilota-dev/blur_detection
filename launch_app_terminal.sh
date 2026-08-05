#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

"$SCRIPT_DIR/start_app.sh"
status=$?

if [ "$status" -ne 0 ] && [ "$status" -ne 130 ]; then
    echo
    echo "Blur Detection exited with status $status."
    read -r -p "Press Enter to close this terminal..."
fi

exit "$status"
