#!/bin/bash
#
# Jarvis Installer
# Usage: curl -fsSL https://raw.githubusercontent.com/rezaulhreza/jarvis/main/install.sh | bash
#

set -e

REPO="rezaulhreza/jarvis"
INSTALL_DIR="$HOME/.jarvis"
BIN_DIR="$HOME/.local/bin"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}"
echo "     _                  _"
echo "    | | __ _ _ ____   _(_)___"
echo " _  | |/ _\` | '__\\ \\ / / / __|"
echo "| |_| | (_| | |   \\ V /| \\__ \\"
echo " \\___/ \\__,_|_|    \\_/ |_|___/"
echo -e "${NC}"
echo "Local AI Assistant Installer"
echo "=============================="
echo

# Check Python version
check_python() {
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
        MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
        MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

        if [ "$MAJOR" -ge 3 ] && [ "$MINOR" -ge 10 ]; then
            echo -e "${GREEN}✓${NC} Python $PYTHON_VERSION found"
            return 0
        fi
    fi

    echo -e "${RED}✗${NC} Python 3.10+ required"
    echo "  Install from: https://www.python.org/downloads/"
    exit 1
}

# Check Ollama
check_ollama() {
    if command -v ollama &> /dev/null; then
        echo -e "${GREEN}✓${NC} Ollama found"
        return 0
    fi

    echo -e "${YELLOW}!${NC} Ollama not found"
    echo "  Install from: https://ollama.ai"
    echo "  Continuing anyway - you'll need it to run Jarvis"
    return 0
}

# Install Jarvis
install_jarvis() {
    echo
    echo "Installing Jarvis..."

    # Create directories
    mkdir -p "$INSTALL_DIR"
    mkdir -p "$BIN_DIR"

    # Clone or download
    if command -v git &> /dev/null; then
        echo "Cloning repository..."
        if [ -d "$INSTALL_DIR/src" ]; then
            cd "$INSTALL_DIR/src" && git pull
        else
            git clone "https://github.com/$REPO.git" "$INSTALL_DIR/src"
        fi
    else
        echo "Downloading..."
        curl -sL "https://github.com/$REPO/archive/main.tar.gz" | tar xz -C "$INSTALL_DIR"
        mv "$INSTALL_DIR/jarvis-main" "$INSTALL_DIR/src"
    fi

    # Create virtual environment
    echo "Setting up Python environment..."
    python3 -m venv "$INSTALL_DIR/venv"
    source "$INSTALL_DIR/venv/bin/activate"

    # Install package
    pip install --upgrade pip > /dev/null
    pip install -e "$INSTALL_DIR/src" > /dev/null

    # Create wrapper script
    cat > "$BIN_DIR/jarvis" << 'WRAPPER'
#!/bin/bash
source "$HOME/.jarvis/venv/bin/activate"
python -m jarvis.cli "$@"
WRAPPER
    chmod +x "$BIN_DIR/jarvis"

    echo -e "${GREEN}✓${NC} Jarvis installed"
}

# Add to PATH if needed
setup_path() {
    if [[ ":$PATH:" != *":$BIN_DIR:"* ]]; then
        echo
        echo "Adding $BIN_DIR to PATH..."

        SHELL_RC=""
        if [ -n "$ZSH_VERSION" ] || [ -f "$HOME/.zshrc" ]; then
            SHELL_RC="$HOME/.zshrc"
        elif [ -f "$HOME/.bashrc" ]; then
            SHELL_RC="$HOME/.bashrc"
        elif [ -f "$HOME/.bash_profile" ]; then
            SHELL_RC="$HOME/.bash_profile"
        fi

        if [ -n "$SHELL_RC" ]; then
            echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$SHELL_RC"
            echo -e "${GREEN}✓${NC} Added to $SHELL_RC"
            echo -e "${YELLOW}!${NC} Run: source $SHELL_RC"
        else
            echo -e "${YELLOW}!${NC} Add to your shell config: export PATH=\"\$HOME/.local/bin:\$PATH\""
        fi
    fi
}

# Download recommended models
setup_models() {
    echo
    read -p "Download recommended Ollama models? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        if command -v ollama &> /dev/null; then
            echo "Downloading models (this may take a while)..."
            ollama pull qwen3:4b || true
            ollama pull llava || true
            ollama pull functiongemma || true
            echo -e "${GREEN}✓${NC} Models downloaded"
        else
            echo -e "${YELLOW}!${NC} Ollama not available, skipping"
        fi
    fi
}

# Main
main() {
    check_python
    check_ollama
    install_jarvis
    setup_path
    setup_models

    echo
    echo -e "${GREEN}Installation complete!${NC}"
    echo
    echo "Usage:"
    echo "  jarvis          # Start CLI"
    echo "  jarvis --dev    # Start Web UI"
    echo "  jarvis --help   # Show all options"
    echo
    echo "If 'jarvis' command not found, restart your terminal."
}

main
