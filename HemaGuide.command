#!/bin/bash
# HemaGuide Launcher
# Double-click this file to start HemaGuide

set -e

# Change to the directory containing this script
cd "$(dirname "$0")"

echo ""
echo "  ╔══════════════════════════════════════╗"
echo "  ║         HemaGuide Launcher           ║"
echo "  ╚══════════════════════════════════════╝"
echo ""

# --- Check Python 3 + venv (auto-install Xcode CLT if needed) ---
if ! python3 -c "import venv" 2>/dev/null; then
    echo "  [SETUP] Installing Python tools (one-time, ~5 minutes)..."
    echo "  A system dialog will appear — click 'Install'."
    xcode-select --install 2>/dev/null
    echo "  Waiting for installation to finish..."
    TRIES=0
    while ! python3 -c "import venv" 2>/dev/null; do
        sleep 5
        TRIES=$((TRIES + 1))
        if [ $TRIES -ge 120 ]; then
            echo "  [ERROR] Python tools installation timed out."
            echo "  Please restart and click 'Install' when the dialog appears."
            echo "  Press any key to exit..."
            read -n 1
            exit 1
        fi
    done
    echo "  Python tools installed."
fi

# --- Require Python 3.12+ (HemaGuide minimum) ---
PYTHON_CMD=""
for CMD in python3.14 python3.13 python3.12 python3; do
    if command -v "$CMD" &> /dev/null && "$CMD" -c "import sys; sys.exit(0 if sys.version_info >= (3, 12) else 1)" 2>/dev/null; then
        PYTHON_CMD="$CMD"
        break
    fi
done

if [ -z "$PYTHON_CMD" ]; then
    FOUND_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || echo "none")
    echo "  [ERROR] HemaGuide requires Python 3.12 or newer (found: $FOUND_VERSION)."
    echo ""
    echo "  Install Python 3.12+ from: https://www.python.org/downloads/"
    echo "  Press any key to open the download page..."
    read -n 1
    open "https://www.python.org/downloads/"
    exit 1
fi

# --- Check Ollama ---
if ! command -v ollama &> /dev/null; then
    echo "  [ERROR] Ollama is not installed."
    echo ""
    echo "  Please install Ollama from: https://ollama.com/download"
    echo "  Press any key to open the download page..."
    read -n 1
    open "https://ollama.com/download"
    exit 1
fi

# Start Ollama if not running
if ! curl -s http://localhost:11434/api/tags &> /dev/null; then
    echo "  [INFO] Starting Ollama..."
    open -a Ollama
    echo "  Waiting for Ollama to start..."
    TRIES=0
    while ! curl -s http://localhost:11434/api/tags &> /dev/null; do
        sleep 2
        TRIES=$((TRIES + 1))
        if [ $TRIES -ge 30 ]; then
            echo "  [ERROR] Ollama did not start within 60 seconds."
            echo "  Please start Ollama manually and try again."
            echo "  Press any key to exit..."
            read -n 1
            exit 1
        fi
    done
    echo "  Ollama is ready."
fi

# --- Check Ollama models ---
DECISION_MODEL="${DECISION_MODEL:-qwen3:8b}"
EMBEDDING_MODEL="embeddinggemma:300m"

for MODEL in "$DECISION_MODEL" "$EMBEDDING_MODEL"; do
    if ! ollama list 2>/dev/null | grep -q "$MODEL"; then
        echo "  [SETUP] Downloading model: $MODEL ..."
        if ! ollama pull "$MODEL"; then
            echo "  [ERROR] Failed to download $MODEL."
            echo "  Check your internet connection and try again."
            echo "  Press any key to exit..."
            read -n 1
            exit 1
        fi
    fi
done

# --- Setup Python venv (first run only) ---
# Recreate venv if it was built with an older Python version
if [ -d "venv" ] && ! venv/bin/python -c "import sys; sys.exit(0 if sys.version_info >= (3, 12) else 1)" 2>/dev/null; then
    VENV_VERSION=$(venv/bin/python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || echo "unknown")
    echo "  [SETUP] Existing venv uses Python $VENV_VERSION (< 3.12). Recreating with $PYTHON_CMD..."
    rm -rf venv
fi

if [ ! -d "venv" ]; then
    echo "  [SETUP] First run: creating Python environment with $PYTHON_CMD..."
    echo "  This may take 2-3 minutes."
    echo ""
    "$PYTHON_CMD" -m venv venv
    source venv/bin/activate
    pip install --upgrade pip -q
    echo "  Installing dependencies..."
    pip install -r requirements.txt -q
    echo "  Python environment ready."
    echo ""
else
    source venv/bin/activate
fi

# --- Check port 8000 ---
if curl -sf http://localhost:8000/api/health &> /dev/null; then
    echo "  [INFO] HemaGuide is already running!"
    echo "  Opening browser..."
    open "http://localhost:8000"
    echo "  Press any key to exit..."
    read -n 1
    exit 0
fi

# --- Start HemaGuide ---
echo "  [INFO] Starting HemaGuide server..."

# Store the backend PID so we can stop it later
python backend/main.py &
BACKEND_PID=$!

# Cleanup on exit
cleanup() {
    echo ""
    echo "  Stopping HemaGuide..."
    kill $BACKEND_PID 2>/dev/null
    wait $BACKEND_PID 2>/dev/null
    echo "  HemaGuide stopped."
}
trap cleanup EXIT INT TERM

# Wait for server to be ready
echo "  Waiting for server to start..."
TRIES=0
while ! curl -s http://localhost:8000/api/health &> /dev/null; do
    sleep 1
    TRIES=$((TRIES + 1))
    if [ $TRIES -ge 30 ]; then
        echo "  [ERROR] Server did not start within 30 seconds."
        echo "  Check the logs above for errors."
        echo "  Press any key to exit..."
        read -n 1
        exit 1
    fi
done

echo ""
echo "  ╔══════════════════════════════════════╗"
echo "  ║  HemaGuide is running!               ║"
echo "  ║  Browser opening...                  ║"
echo "  ║                                      ║"
echo "  ║  Close this window to stop.          ║"
echo "  ╚══════════════════════════════════════╝"
echo ""

# Open browser
open "http://localhost:8000"

# Keep running until Terminal is closed
wait $BACKEND_PID
