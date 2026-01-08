# SoupaWhisper

A simple local push-to-talk (not streaming) or press-to-toggle (streaming) voice dictation tool for Linux using faster-whisper.
It automatically copies to clipboard and types into the active input. Streaming transcribes speech chunks as some silence is detected. Not-streaming transcribes the entire audio file at once.

Note that push-to-talk is possible in a streaming mode but can't be used with terminal applications with paralell typing.

## Requirements

- Python 3.10+
- Poetry or uv
- (not-streaming) Linux with X11 (ALSA audio)
- (streaming) Any Linux supported by PyAudio

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
sudo apt install alsa-utils xclip xdotool libnotify-bin portaudio19-dev

# Fedora
sudo dnf install alsa-utils xclip xdotool libnotify portaudio-devel

# Arch
sudo pacman -S alsa-utils xclip xdotool libnotify portaudio

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

- Press the configured hotkey (default: **F12**) to start transcribing
- (not-streaming) Release the hotkey → transcribes, copies to clipboard, and types into active input.
- (streaming) Press the hotkey again to stop transcribing.
- Press **Ctrl+C** to quit (when running manually)

### Streaming vs Non-Streaming Mode

**Non-Streaming Mode** (default):
- Records entire audio, then transcribes all at once
- Uses `arecord` to record audio (only ALSA supported)
- Text appears only after you stop recording
- Better for short, precise dictations
- Slightly more accurate for complete sentences
- Use `--no-streaming` flag or set `default_streaming = false` in config

**Streaming Mode**:
- Uses WebRTC VAD (Voice Activity Detection) to detect when you speak
- Uses `pyaudio` to record audio (any audio backend supported by PyAudio)
- Transcribes natural voice segments (sentences / phrases) one by one as they are detected
- Text appears incrementally as you speak, but only after short silences
- Better for longer dictations and more natural sentence boundaries
- Use `--streaming` flag or set `default_streaming = true` in config

## Run as a systemd Service

The installer can set this up automatically. If you skipped it, run:

```bash
./install.sh  # Select 'y' when prompted for systemd
```

### Service Commands

Use `make` commands or directly:

```bash
systemctl --user start soupawhisper     # Start
systemctl --user stop soupawhisper      # Stop
systemctl --user restart soupawhisper   # Restart
systemctl --user status soupawhisper    # Status
journalctl --user -u soupawhisper -f    # View logs
```

## Configuration

Edit `~/.config/soupawhisper/config.ini` - it contains explanations for each option.

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

