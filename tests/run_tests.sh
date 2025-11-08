#!/bin/bash
set -euo pipefail

PYTEST_ARGS="-v -W ignore::UserWarning"

# Get the tests directory (where this script is located)
TESTS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$TESTS_DIR/.." && pwd )"


# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Detect Python environment manager
if command_exists uv; then
    PYTHON_CMD="uv run"
elif command_exists python3; then
    PYTHON_CMD="python3 -m"
elif command_exists python; then
    PYTHON_CMD="python -m"
else
    echo "Error: No Python interpreter found (install uv or python3)" >&2
    exit 1
fi

echo ""
echo -e "Preparing test data..."
cd "$TESTS_DIR/cases" || { echo -e "${RED} Cannot cd to $TESTS_DIR/cases${NC}"; exit 1; }

shopt -s nullglob

EXPECTED_DIRS=()
for f in *.tgz; do
    [[ -f "$f" ]] || continue
    EXPECTED_DIRS+=("${f%.tgz}")
done

NEED_EXTRACT=false

# Check if any expected directory is missing
for tgz_file in *.tgz; do
    if [ -f "$tgz_file" ]; then
        DIR_NAME="${tgz_file%.tgz}"
        if [ ! -d "$DIR_NAME" ]; then
            NEED_EXTRACT=true
            break
        fi
    fi
done

if [ "$NEED_EXTRACT" = true ]; then
    echo "  Extracting test data files..."
    for f in *.tgz; do
        if [ -f "$f" ]; then
            echo "    Extracting $f..."
            tar -xzf "$f"
            if [ $? -ne 0 ]; then
                echo -e "Failed to extract $f$"
                exit 1
            fi
        fi
    done
    echo -e "Test data extracted"
else
    echo -e "Test data already extracted"
fi

echo ""
echo -e "Running acceptance tests..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

cd "$TESTS_DIR/cases"

# Run the tests
if command_exists uv; then
    uv run pytest ../acceptance_tests/ $PYTEST_ARGS
else
    $PYTHON_CMD pytest ../acceptance_tests/ $PYTEST_ARGS
fi
