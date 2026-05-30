#!/bin/bash
set -e

echo "=== ERL & SC-ERL Experiment Environment Setup ==="

# 1. Install 'task' if not present
if ! command -v task &> /dev/null; then
    echo "Installing Taskfile runner (task)..."
    # Installs to bin/task in current folder if normal user, or globally if root
    if [ "$EUID" -ne 0 ]; then
        echo "Not root. Installing locally to ./bin..."
        sh -c "$(curl --location https://taskfile.dev/install.sh)" -- -d -b ./bin
        export PATH="$PWD/bin:$PATH"
        echo "Task installed to ./bin. Added to PATH."
    else
        echo "Root detected. Installing globally to /usr/local/bin..."
        sh -c "$(curl --location https://taskfile.dev/install.sh)" -- -d -b /usr/local/bin
        echo "Task installed to /usr/local/bin."
    fi
else
    echo "Taskfile runner (task) is already installed."
fi

# 2. Install 'uv' if not present
if ! command -v uv &> /dev/null; then
    echo "Installing uv package manager..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Source env path
    if [ -f "$HOME/.local/bin/env" ]; then
        source "$HOME/.local/bin/env"
    elif [ -f "$HOME/.cargo/env" ]; then
        source "$HOME/.cargo/env"
    fi
else
    echo "uv package manager is already installed."
fi

# 3. Run uv sync to synchronize python dependencies
echo "Synchronizing python dependencies with uv sync..."
uv sync

echo "=== Setup Completed Successfully! ==="
echo "You can now run tasks. Examples:"
if command -v task &> /dev/null; then
    echo "  task --list"
else
    echo "  ./bin/task --list"
fi
