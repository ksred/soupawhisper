# SoupaWhisper

A simple push-to-talk voice dictation tool for Linux using faster-whisper. Hold a key to record, release to transcribe, and it automatically copies to clipboard and types into the active input.

## Requirements

- Python 3.10+
- Poetry
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
3. Install Python dependencies via Poetry
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
compute_type = float16 (or float32)
```

## Usage

```bash
poetry run python dictate.py
```

- Hold **F12** to record
- Release to transcribe → copies to clipboard and types into active input
- Press **Ctrl+C** to quit (when running manually)

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

### Control Panel

A TUI control panel is included for easy management. `install.sh` symlinks it into `~/.local/bin`, so after installing it's just:

```bash
soupawhisper-ctl
```

(Or run it directly from the repo with `./soupawhisper-ctl` without installing.)

Features:
- Start/stop/restart the service
- Switch Whisper models (standard + distilled)
- Change hotkey (F12, F11, or detect any key)
- Enable/disable auto-start
- View logs

The control panel edits `~/.config/soupawhisper/config.ini` and restarts the service to apply changes.

## Configuration

Edit `~/.config/soupawhisper/config.ini`:

```ini
[whisper]
# Model size: tiny.en, base.en, small.en, medium.en, large-v3
model = base.en

# Device: cpu or cuda (cuda requires cuDNN)
device = cpu

# Compute type: int8 for CPU, float16 (or float32) for GPU
compute_type = int8

[hotkey]
# Key to hold for recording: f12, scroll_lock, pause, etc.
key = f12

[behavior]
# Type text into active input field
auto_type = true

# Copy transcription to clipboard (set false to type only, leaving clipboard untouched)
copy_to_clipboard = true

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

### Standard Models
| Model | Size | Speed | Accuracy |
|-------|------|-------|----------|
| tiny.en | ~75MB | Fastest | Basic |
| base.en | ~150MB | Fast | Good |
| small.en | ~500MB | Medium | Better |
| medium.en | ~1.5GB | Slower | Great |
| large-v3 | ~3GB | Slowest | Best |

### Distilled Models (Recommended)
| Model | Speed vs large-v3 | Accuracy |
|-------|-------------------|----------|
| distil-large-v3 | ~2x faster | ~99% of large-v3 |
| distil-large-v2 | ~2x faster | ~99% of large-v2 |
| large-v3-turbo | ~4x faster | ~97% of large-v3 |

Distilled models offer near-original quality with significantly better speed. For dictation with GPU, `large-v3-turbo` or `distil-large-v3` are excellent choices. For CPU or low VRAM, `small.en` or `base.en` work well.

The distilled/turbo entries in `soupawhisper-ctl`'s model menu (`Systran/faster-distil-whisper-large-v3`, `Systran/faster-distil-whisper-large-v2`, `deepdml/faster-whisper-large-v3-turbo-ct2`) are third-party CTranslate2 conversions hosted on Hugging Face, not maintained by this project or by OpenAI/Systran/deepdml as an official faster-whisper release. Repo names, availability, and conversion quality may change upstream without notice.

## Groq Cloud Backend

For cloud-based transcription without local GPU or model downloads, use Groq's hosted Whisper API.

Set the model to `groq:<model-name>` in your config:

```ini
[whisper]
model = groq:whisper-large-v3-turbo
```

Requires `GROQ_API_KEY` in your environment. Falls back to reading a `GROQ_API_KEY=...` line from `~/.config/soupawhisper/.env` if the env var is unset.

Available Groq models:
- `whisper-large-v3-turbo` - fast, high quality
- `whisper-large-v3` - highest accuracy

Audio is sent to Groq's API for transcription. No local compute needed.
