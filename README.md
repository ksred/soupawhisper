# SoupaWhisper

A simple local press-to-toggle voice dictation tool for Linux using faster-whisper. Press a key to start transcribing, press again to stop. It automatically copies to clipboard and types into the active input. Supports both streaming and non-streaming modes.

Note that push-to-talk is possible but can't be used with terminal applications.

## Requirements

- Python 3.10+
- Poetry or uv
- Linux with X11 (ALSA audio)

## Supported Distros

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
3. Install Python dependencies via Poetry or uv
4. Set up the config file
5. Optionally install as a systemd service

### Manual Installation

```bash
# Ubuntu/Debian
sudo apt install alsa-utils xclip xdotool libnotify-bin

# Fedora
sudo dnf install alsa-utils xclip xdotool libnotify

# Arch
sudo pacman -S alsa-utils xclip xdotool libnotify

# Then install Python deps
poetry install
# OR with uv:
uv sync
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

### With Poetry

```bash
poetry run python dictate.py
```

### With uv

```bash
uv run python dictate.py
```

### Using Makefile

```bash
make run           # Run in default mode
make run-stream    # Run in streaming mode
make run-no-stream # Run in non-streaming mode
make test          # Run tests in virtual environment
```

### Operation

- Press the configured hotkey (default: **F10**) to start transcribing
- Press again to stop → transcribes, copies to clipboard, and types into active input
- Press **Ctrl+C** to quit (when running manually)

### Streaming vs Non-Streaming Mode

**Non-Streaming Mode** (default):
- Records entire audio, then transcribes all at once
- Text appears only after you stop recording
- Better for short, precise dictations
- Slightly more accurate for complete sentences
- Use `--no-streaming` flag or set `default_streaming = false` in config

**Streaming Mode**:
- Transcribes audio in real-time chunks while recording
- Text appears incrementally as you speak
- Better for longer dictations
- Lower latency for seeing results
- Configured via `default_streaming = true` in config

## Run as a systemd Service

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
# Key to press to toggle recording: f10, f12, scroll_lock, pause, etc.
key = f10

[behavior]
# Enable streaming mode by default
default_streaming = true

# Show desktop notification
notifications = true

# Copy resulting text to clipboard
clipboard = true

# Type text into active input field
auto_type = true

# Delay between typing characters in seconds
typing_delay = 0.01

[streaming]
# Length of audio chunk in seconds (bigger = better quality, longer delay)
streaming_chunk_seconds = 3.0

# Overlap between chunks in seconds (should be less than chunk_seconds)
streaming_overlap_seconds = 1.5

# Threshold for matching words in overlapping chunks (seconds)
streaming_match_words_threshold_seconds = 0.1
```

Create the config directory and file if it doesn't exist:
```bash
mkdir -p ~/.config/soupawhisper
cp config.example.ini ~/.config/soupawhisper/config.ini
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
| tiny.en  |   ~75MB | Fastest | Basic |
| base.en  |  ~150MB | Fast    | Good |
| small.en |  ~500MB | Medium  | Better |
| medium.en | ~1.5GB | Slower  | Great |
| large-v3 |    ~3GB | Slowest | Best |

For dictation, `base.en` or `small.en` is usually the sweet spot.

## Testing

### With Poetry

```bash
poetry run pytest dictate_tests.py
```

### With uv

```bash
uv run pytest dictate_tests.py
```

### Using Makefile

```bash
make test
```

