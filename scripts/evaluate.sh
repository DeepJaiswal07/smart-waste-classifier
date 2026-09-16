#!/usr/bin/env bash
# Shell script to run test set evaluation
set -e

echo "Evaluating best checkpoint on held-out test split..."
python -m src.main evaluate
