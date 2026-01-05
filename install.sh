#!/bin/bash
# Install SoupaWhisper on Linux
# Supports: Ubuntu, Pop!_OS, Debian, Fedora, Arch

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_DIR="$HOME/.config/soupawhisper"
SERVICE_DIR="$HOME/.config/systemd/user"

# Detect package manager
detect_package_manager() {
    if command -v apt &> /dev/null; then
        echo "apt"
    elif command -v dnf &> /dev/null; then
        echo "dnf"
    elif command -v pacman &> /dev/null; then
        echo "pacman"
    elif command -v zypper &> /dev/null; then
        echo "zypper"
    else
        echo "unknown"
    fi
}

# Install system dependencies
install_deps() {
    local pm=$(detect_package_manager)

    echo "Detected package manager: $pm"
    echo "Installing system dependencies..."

    case $pm in
        apt)
            sudo apt update
            sudo apt install -y alsa-utils xclip xdotool libnotify-bin
            ;;
        dnf)
            sudo dnf install -y alsa-utils xclip xdotool libnotify
            ;;
        pacman)
            sudo pacman -S --noconfirm alsa-utils xclip xdotool libnotify
            ;;
        zypper)
            sudo zypper install -y alsa-utils xclip xdotool libnotify-tools
            ;;
        *)
            echo "Unknown package manager. Please install manually:"
            echo "  alsa-utils xclip xdotool libnotify"
            ;;
    esac
}

# Detect Python package manager
detect_python_manager() {
    if command -v uv &> /dev/null; then
        echo "uv"
    elif command -v poetry &> /dev/null; then
        echo "poetry"
    else
        echo "unknown"
    fi
}

# Install Python dependencies
install_python() {
    echo ""
    echo "Installing Python dependencies..."

    local pm=$(detect_python_manager)

    case $pm in
        uv)
            echo "Using uv..."
            uv sync
            ;;
        poetry)
            echo "Using Poetry..."
            poetry install
            ;;
        *)
            echo "Neither uv nor Poetry found. Please install one:"
            echo "  uv:     curl -LsSf https://astral.sh/uv/install.sh | sh"
            echo "  Poetry: curl -sSL https://install.python-poetry.org | python3 -"
            exit 1
            ;;
    esac
}

# Setup config file
setup_config() {
    echo ""
    echo "Setting up config..."
    mkdir -p "$CONFIG_DIR"

    if [ ! -f "$CONFIG_DIR/config.ini" ]; then
        cp "$SCRIPT_DIR/config.example.ini" "$CONFIG_DIR/config.ini"
        echo "Created config at $CONFIG_DIR/config.ini"
    else
        echo "Config already exists at $CONFIG_DIR/config.ini"
    fi
}

# Install systemd service
install_service() {
    echo ""
    echo "Installing systemd user service..."

    mkdir -p "$SERVICE_DIR"

    # Get current display settings
    local display="${DISPLAY:-:0}"
    local xauthority="${XAUTHORITY:-$HOME/.Xauthority}"
    local venv_path="$SCRIPT_DIR/.venv"

    # Detect Python package manager and get venv path
    local pm=$(detect_python_manager)
    case $pm in
        uv)
            # UV uses .venv in the script directory
            venv_path="$SCRIPT_DIR/.venv"
            ;;
        poetry)
            # Try to get Poetry's venv path, fallback to .venv
            venv_path=$(poetry env info --path 2>/dev/null || echo "$SCRIPT_DIR/.venv")
            ;;
        *)
            # Fallback to .venv
            venv_path="$SCRIPT_DIR/.venv"
            ;;
    esac

    cat > "$SERVICE_DIR/soupawhisper.service" << EOF
[Unit]
Description=SoupaWhisper Voice Dictation
After=graphical-session.target

[Service]
Type=simple
WorkingDirectory=$SCRIPT_DIR
ExecStart=$venv_path/bin/python $SCRIPT_DIR/dictate.py
Restart=on-failure
RestartSec=5

# X11 display access
Environment=DISPLAY=$display
Environment=XAUTHORITY=$xauthority

[Install]
WantedBy=default.target
EOF

    echo "Created service at $SERVICE_DIR/soupawhisper.service"

    # Reload and enable
    systemctl --user daemon-reload
    systemctl --user enable soupawhisper

    echo ""
    echo "Service installed! Commands:"
    echo "  systemctl --user start soupawhisper   # Start"
    echo "  systemctl --user stop soupawhisper    # Stop"
    echo "  systemctl --user status soupawhisper  # Status"
    echo "  journalctl --user -u soupawhisper -f  # Logs"
}

# Main
main() {
    echo "==================================="
    echo "  SoupaWhisper Installer"
    echo "==================================="
    echo ""

    install_deps
    install_python
    setup_config

    echo ""
    read -p "Install as systemd service? [y/N] " -n 1 -r
    echo ""

    if [[ $REPLY =~ ^[Yy]$ ]]; then
        install_service
    fi

    echo ""
    echo "==================================="
    echo "  Installation complete!"
    echo "==================================="
    echo ""
    echo "To run manually:"
    local pm=$(detect_python_manager)
    case $pm in
        uv)
            echo "  uv run python dictate.py"
            ;;
        poetry)
            echo "  poetry run python dictate.py"
            ;;
        *)
            echo "  python dictate.py  # (after activating venv)"
            ;;
    esac
    echo ""
    echo "Config: $CONFIG_DIR/config.ini"
    echo "Hotkey: F10 (hold to record)"
    echo "Exit:   Ctrl+C"
}

main "$@"
