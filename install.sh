#!/bin/bash
# Install graft and associated utilities

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Installing graft Python package..."
pip install -e "$SCRIPT_DIR"

echo "Creating ~/.claude-archive directory..."
mkdir -p ~/.claude-archive/raw

echo "Symlinking utilities to ~/.local/bin..."
mkdir -p ~/.local/bin
ln -sf "$SCRIPT_DIR/bin/claude-sub" ~/.local/bin/
ln -sf "$SCRIPT_DIR/bin/claude-session-to-graft" ~/.local/bin/
ln -sf "$SCRIPT_DIR/bin/claude-archive-list" ~/.local/bin/

echo ""
echo "Done! Make sure ~/.local/bin is in your PATH."
echo "You may need to add this to your ~/.bashrc:"
echo '  export PATH="$HOME/.local/bin:$PATH"'
