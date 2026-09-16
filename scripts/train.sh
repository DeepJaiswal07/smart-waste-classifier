#!/usr/bin/env bash
# Shell script to run training
set -e

# Default to quick mode if requested
if [ "$1" == "--quick" ]; then
    echo "Running quick development training..."
    python -m src.main train --quick
else
    echo "Running full model training..."
    python -m src.main train
fi
