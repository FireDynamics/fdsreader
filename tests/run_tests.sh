#!/bin/bash
set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Get the tests directory (where this script is located)
TESTS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$TESTS_DIR/.." && pwd )"

# Function to show help
show_help() {
    echo -e "${BLUE}FDSreader Test Runner${NC}"
    echo ""
    echo "Usage: $0 [OPTIONS] [PYTEST_ARGS]"
    echo ""
    echo "Run FDSreader acceptance tests with automatic setup and data extraction."
    echo ""
    echo -e "${CYAN}Options:${NC}"
    echo "  -h, --help          Show this help message and exit"
    echo "  -f, --force-extract Force re-extraction of test data from .tgz files"
    echo ""
    echo -e "${CYAN}Examples:${NC}"
    echo "  $0                  Run all tests with default settings"
    echo "  $0 -v               Run tests in verbose mode"
    echo "  $0 -f               Force re-extract data and run tests"
    echo "  $0 -x               Stop on first test failure"
    echo "  $0 -k test_bndf     Run only tests matching 'test_bndf'"
    echo "  $0 --tb=short       Use shorter traceback format"
    echo "  $0 -f -v -x         Combine multiple options"
    echo ""
    echo -e "${CYAN}Common pytest arguments:${NC}"
    echo "  -v, --verbose       Increase verbosity"
    echo "  -q, --quiet         Decrease verbosity"
    echo "  -x, --exitfirst     Exit on first failure"
    echo "  -k EXPRESSION       Run tests matching expression"
    echo "  --tb=style          Traceback style (short/long/native/no)"
    echo "  --lf                Run last failed tests"
    echo "  --ff                Run failed tests first"
    echo "  -s                  Don't capture stdout (show print statements)"
    echo "  --collect-only      Only collect tests, don't run them"
    echo ""
    echo -e "${CYAN}What this script does:${NC}"
    echo "  1. Installs dependencies (pytest, numpy, fdsreader)"
    echo "  2. Extracts test data from .tgz files if needed"
    echo "  3. Runs acceptance tests from the correct directory"
    echo "  4. Reports test results with colored output"
    echo ""
    echo -e "${CYAN}Note:${NC} Test data is cached after extraction. Use -f to force re-extraction."
}

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Parse command line arguments
FORCE_EXTRACT=false
PYTEST_ARGS=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            show_help
            exit 0
            ;;
        -f|--force-extract)
            FORCE_EXTRACT=true
            shift
            ;;
        *)
            PYTEST_ARGS="$PYTEST_ARGS $1"
            shift
            ;;
    esac
done

# Default to verbose if no pytest args provided
if [ -z "$PYTEST_ARGS" ]; then
    PYTEST_ARGS="-v"
fi

echo -e "${BLUE}╔══════════════════════════════════════╗${NC}"
echo -e "${BLUE}║   FDSreader Acceptance Test Runner   ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════╝${NC}"
echo ""

# Detect Python environment manager
if command_exists uv; then
    PYTHON_CMD="uv run"
    PIP_INSTALL="uv pip install"
    echo -e "${GREEN} Using uv for Python environment${NC}"
elif command_exists python3; then
    PYTHON_CMD="python3 -m"
    PIP_INSTALL="python3 -m pip install"
    echo -e "${YELLOW} Using system python3 (consider using uv)${NC}"
elif command_exists python; then
    PYTHON_CMD="python -m"
    PIP_INSTALL="python -m pip install"
    echo -e "${YELLOW} Using system python${NC}"
else
    echo -e "${RED} Error: No Python interpreter found${NC}"
    exit 1
fi

# Step 1: Install dependencies
echo ""
echo -e "${GREEN}Step 1: Installing dependencies...${NC}"

# Install pytest and numpy
$PIP_INSTALL pytest numpy > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo -e "${RED} Failed to install pytest and numpy${NC}"
    echo -e "${YELLOW}  Try running: $PIP_INSTALL pytest numpy${NC}"
    exit 1
fi

# Install fdsreader package in editable mode from project root
cd "$PROJECT_ROOT"
$PIP_INSTALL -e . > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo -e "${RED} Failed to install fdsreader package${NC}"
    echo -e "${YELLOW}  Try running: $PIP_INSTALL -e .${NC}"
    exit 1
fi
echo -e "${GREEN} Dependencies installed${NC}"

# Step 2: Extract test data if needed
echo ""
echo -e "${GREEN}Step 2: Preparing test data...${NC}"
cd "$TESTS_DIR/cases" || { echo -e "${RED} Cannot cd to $TESTS_DIR/cases${NC}"; exit 1; }

# Make globs like *.tgz expand to nothing (not the literal string) if no match
shopt -s nullglob

# Build EXPECTED_DIRS dynamically from available .tgz files
EXPECTED_DIRS=()
for f in *.tgz; do
    [[ -f "$f" ]] || continue
    EXPECTED_DIRS+=("${f%.tgz}")
done

# Check if we need to extract
NEED_EXTRACT=false

if [ "$FORCE_EXTRACT" = true ]; then
    echo "  Force extraction requested, cleaning old data..."
    # Remove existing extracted directories
    for dir in "${EXPECTED_DIRS[@]}"; do
        if [ -d "$dir" ]; then
            rm -rf "$dir"
        fi
    done
    NEED_EXTRACT=true
else
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
fi

if [ "$NEED_EXTRACT" = true ]; then
    echo "  Extracting test data files..."
    for f in *.tgz; do
        if [ -f "$f" ]; then
            echo "    Extracting $f..."
            tar -xzf "$f"
            if [ $? -ne 0 ]; then
                echo -e "${RED}✗ Failed to extract $f${NC}"
                exit 1
            fi
        fi
    done
    echo -e "${GREEN} Test data extracted${NC}"
else
    echo -e "${GREEN} Test data already extracted (use -f to force re-extract)${NC}"
fi

# Step 3: Run tests
echo ""
echo -e "${GREEN}Step 3: Running acceptance tests...${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Run from cases directory (where test data is located)
cd "$TESTS_DIR/cases"

# Run the tests
if command_exists uv; then
    uv run pytest ../acceptance_tests/ $PYTEST_ARGS
else
    $PYTHON_CMD pytest ../acceptance_tests/ $PYTEST_ARGS
fi

TEST_RESULT=$?

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Step 4: Report results
if [ $TEST_RESULT -eq 0 ]; then
    echo -e "${GREEN} All tests passed!${NC}"
else
    echo -e "${RED} Tests failed (exit code: $TEST_RESULT)${NC}"
    echo ""
    echo -e "${YELLOW}Tip: Use '$0 --lf' to rerun only failed tests${NC}"
fi

exit $TEST_RESULT
