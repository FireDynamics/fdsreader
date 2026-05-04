#!/bin/bash
set -euo pipefail

# Run all fdsreader tests locally.
# Automatically extracts test data archives if not already done.
# Works with uv, python3, or python.

TESTS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$TESTS_DIR/.." && pwd )"

# Detect Python environment manager
if command -v uv >/dev/null 2>&1; then
    PYTEST_CMD="uv run pytest"
elif command -v python3 >/dev/null 2>&1; then
    PYTEST_CMD="python3 -m pytest"
elif command -v python >/dev/null 2>&1; then
    PYTEST_CMD="python -m pytest"
else
    echo "Error: No Python interpreter found (install uv or python3)" >&2
    exit 1
fi

# Extract test data archives if needed
echo "Preparing test data..."
cd "$TESTS_DIR/cases"

NEED_EXTRACT=false
for tgz_file in *.tgz; do
    [ -f "$tgz_file" ] || continue
    dir_name="${tgz_file%.tgz}"
    if [ ! -d "$dir_name" ]; then
        NEED_EXTRACT=true
        break
    fi
done

if [ "$NEED_EXTRACT" = true ]; then
    echo "  Extracting archives..."
    for f in *.tgz; do
        [ -f "$f" ] || continue
        echo "    $f"
        tar -xzf "$f"
    done
    echo "  Done."
else
    echo "  Already extracted."
fi

# Run all tests
cd "$PROJECT_ROOT"
echo ""
echo "Running tests..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
$PYTEST_CMD tests/ -v --cov=fdsreader --cov-report=term-missing
