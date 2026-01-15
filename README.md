# SoupaWhisper

A simple push-to-talk voice dictation tool using faster-whisper. Hold a key to record, release to transcribe, and it automatically copies to clipboard and types into the active input.

## Requirements

- Python 3.10+
- Poetry

## Supported Platforms

- **macOS** (arm64 and x86_64)
- **Linux** with X11 (ALSA audio)
  - Ubuntu / Pop!_OS / Debian (apt)
  - Fedora (dnf)
  - Arch Linux (pacman)
  - openSUSE (zypper)

## Installation

```bash
git clone https://github.com/ksred/soupawhisper.git
cd soupawhisper
chmod +x install.sh
./install.sh
```

The installer will:
1. Detect your package manager
2. Install system dependencies
3. Install Python dependencies via Poetry
4. Set up the config file
5. Optionally install as a systemd service

### macOS Installation

```bash
# Install sox for audio recording
brew install sox

# Install Python dependencies
poetry install
```

**Note:** On first run, you may need to grant permissions in **System Settings > Privacy & Security**:
- **Accessibility** - for auto-typing feature
- **Microphone** - for audio recording

### Linux Installation

```bash
# Ubuntu/Debian
sudo apt install alsa-utils xclip xdotool libnotify-bin

# Fedora
sudo dnf install alsa-utils xclip xdotool libnotify

# Arch
sudo pacman -S alsa-utils xclip xdotool libnotify

# Then install Python deps
poetry install
```

### GPU Support (Optional)

For NVIDIA GPU acceleration, install cuDNN 9:

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install libcudnn9-cuda-12
```

Then edit `~/.config/soupawhisper/config.ini`:
```ini
device = cuda
compute_type = float16
```

## Usage

```bash
poetry run python dictate.py
```

- Hold **F12** to record
- Release to transcribe → copies to clipboard and types into active input
- Press **Ctrl+C** to quit (when running manually)

### macOS: Run as Background Process

To use with other terminal apps like Claude Code, Codex etc:

1. Start the dictation service in the background:
   ```bash
   nohup poetry run python dictate.py > /dev/null 2>&1 &
   ```

2. Switch to your target app (iTerm, Claude Code, Codex, etc.)

3. Hold **F12** to record, release to transcribe - text types at your cursor

To stop the background process:
```bash
pkill -f "python dictate.py"
```

To use from non-terminal apps, hold F12 to record and transcribe, the text is copied to clipboard.

### Linux: Run as a systemd Service

The installer can set this up automatically. If you skipped it, run:

```bash
./install.sh  # Select 'y' when prompted for systemd
```

### Service Commands

```bash
systemctl --user start soupawhisper     # Start
systemctl --user stop soupawhisper      # Stop
systemctl --user restart soupawhisper   # Restart
systemctl --user status soupawhisper    # Status
journalctl --user -u soupawhisper -f    # View logs
```

## Configuration

Edit `~/.config/soupawhisper/config.ini`:

```ini
[whisper]
# Model size: tiny.en, base.en, small.en, medium.en, large-v3
model = base.en

# Device: cpu or cuda (cuda requires cuDNN)
device = cpu

# Compute type: int8 for CPU, float16 for GPU
compute_type = int8

[hotkey]
# Key to hold for recording: f12, scroll_lock, pause, etc.
key = f12

[behavior]
# Type text into active input field
auto_type = true

# Show desktop notification
notifications = true
```

Create the config directory and file if it doesn't exist:
```bash
mkdir -p ~/.config/soupawhisper
cp /path/to/soupawhisper/config.example.ini ~/.config/soupawhisper/config.ini
```

## Troubleshooting

**No audio recording:**
```bash
# Check your input device
arecord -l

# Test recording
arecord -d 3 test.wav && aplay test.wav
```

**Permission issues with keyboard:**
```bash
sudo usermod -aG input $USER
# Then log out and back in
```

**cuDNN errors with GPU:**
```
Unable to load any of {libcudnn_ops.so.9...}
```
Install cuDNN 9 (see GPU Support section above) or switch to CPU mode.

## Model Sizes

| Model | Size | Speed | Accuracy |
|-------|------|-------|----------|
| tiny.en | ~75MB | Fastest | Basic |
| base.en | ~150MB | Fast | Good |
| small.en | ~500MB | Medium | Better |
| medium.en | ~1.5GB | Slower | Great |
| large-v3 | ~3GB | Slowest | Best |

For dictation, `base.en` or `small.en` is usually the sweet spot.
