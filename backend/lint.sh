#!/bin/bash
# Backend linting and formatting scripts

set -e

cd "$(dirname "$0")"
source venv/bin/activate

case "$1" in
    "check")
        echo "Running black check..."
        black --check app/
        echo "Running ruff check..."
        ruff check app/
        ;;
    "fix")
        echo "Running black format..."
        black app/
        echo "Running ruff fix..."
        ruff check --fix app/
        ;;
    *)
        echo "Usage: $0 {check|fix}"
        echo "  check - Check code formatting and linting"
        echo "  fix   - Auto-fix formatting and linting issues"
        exit 1
        ;;
esac

echo "✨ Done!"
