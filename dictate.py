#!/usr/bin/env python3
"""
SoupaWhisper - Voice dictation tool using faster-whisper or Groq API.
Hold the hotkey to record, release to transcribe and copy to clipboard.
"""

import argparse
import configparser
import subprocess
import tempfile
import threading
import signal
import sys
import os
from pathlib import Path

from pynput import keyboard

# Optional imports based on backend
WhisperModel = None
requests = None

__version__ = "0.1.0"

# Load configuration
CONFIG_PATH = Path.home() / ".config" / "soupawhisper" / "config.ini"


def load_config():
    config = configparser.ConfigParser()

    # Defaults
    defaults = {
        "backend": "local",  # "local" (faster-whisper) or "groq" (cloud API)
        "model": "base.en",
        "device": "cpu",
        "compute_type": "int8",
        "api_key": "",       # Groq API key
        "language": "en",    # Language for Groq API
        "key": "f12",
        "auto_type": "true",
        "notifications": "true",
    }

    if CONFIG_PATH.exists():
        config.read(CONFIG_PATH)

    return {
        "backend": config.get("whisper", "backend", fallback=defaults["backend"]),
        "model": config.get("whisper", "model", fallback=defaults["model"]),
        "device": config.get("whisper", "device", fallback=defaults["device"]),
        "compute_type": config.get("whisper", "compute_type", fallback=defaults["compute_type"]),
        "api_key": config.get("groq", "api_key", fallback=defaults["api_key"]),
        "language": config.get("groq", "language", fallback=defaults["language"]),
        "key": config.get("hotkey", "key", fallback=defaults["key"]),
        "auto_type": config.getboolean("behavior", "auto_type", fallback=True),
        "notifications": config.getboolean("behavior", "notifications", fallback=True),
    }


CONFIG = load_config()


def get_hotkey(key_name):
    """Map key name to pynput key."""
    key_name = key_name.lower()
    if hasattr(keyboard.Key, key_name):
        return getattr(keyboard.Key, key_name)
    elif len(key_name) == 1:
        return keyboard.KeyCode.from_char(key_name)
    else:
        print(f"Unknown key: {key_name}, defaulting to f12")
        return keyboard.Key.f12


HOTKEY = get_hotkey(CONFIG["key"])
BACKEND = CONFIG["backend"]
MODEL_SIZE = CONFIG["model"]
DEVICE = CONFIG["device"]
COMPUTE_TYPE = CONFIG["compute_type"]
API_KEY = CONFIG["api_key"]
LANGUAGE = CONFIG["language"]
AUTO_TYPE = CONFIG["auto_type"]
NOTIFICATIONS = CONFIG["notifications"]

# Groq API settings
GROQ_API_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
GROQ_MODEL = "whisper-large-v3"  # Groq's Whisper model


def transcribe_groq(audio_path):
    """Transcribe audio using Groq Cloud API."""
    global requests
    if requests is None:
        import requests as req
        requests = req

    headers = {"Authorization": f"Bearer {API_KEY}"}

    with open(audio_path, "rb") as f:
        response = requests.post(
            GROQ_API_URL,
            headers=headers,
            files={"file": ("audio.wav", f, "audio/wav")},
            data={"model": GROQ_MODEL, "language": LANGUAGE},
            timeout=30,
        )

    if not response.ok:
        raise Exception(f"Groq API error {response.status_code}: {response.text}")

    return response.json().get("text", "").strip()


class Dictation:
    def __init__(self):
        self.recording = False
        self.record_process = None
        self.temp_file = None
        self.model = None
        self.model_loaded = threading.Event()
        self.model_error = None
        self.running = True
        self.backend = BACKEND

        if self.backend == "groq":
            # Groq API - no model to load
            if not API_KEY:
                print("ERROR: Groq API key not configured!")
                print("Add [groq] section with api_key to your config.ini")
                print("Get your key at: https://console.groq.com/")
                sys.exit(1)
            self.model_loaded.set()
            hotkey_name = HOTKEY.name if hasattr(HOTKEY, 'name') else HOTKEY.char
            print(f"Using Groq Cloud API (language: {LANGUAGE})")
            print(f"Ready for dictation!")
            print(f"Hold [{hotkey_name}] to record, release to transcribe.")
            print("Press Ctrl+C to quit.")
        else:
            # Local faster-whisper model
            print(f"Loading Whisper model ({MODEL_SIZE})...")
            threading.Thread(target=self._load_model, daemon=True).start()

    def _load_model(self):
        """Load local faster-whisper model."""
        global WhisperModel
        try:
            if WhisperModel is None:
                from faster_whisper import WhisperModel as WM
                WhisperModel = WM
            self.model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
            self.model_loaded.set()
            hotkey_name = HOTKEY.name if hasattr(HOTKEY, 'name') else HOTKEY.char
            print(f"Model loaded. Ready for dictation!")
            print(f"Hold [{hotkey_name}] to record, release to transcribe.")
            print("Press Ctrl+C to quit.")
        except Exception as e:
            self.model_error = str(e)
            self.model_loaded.set()
            print(f"Failed to load model: {e}")
            if "cudnn" in str(e).lower() or "cuda" in str(e).lower():
                print("Hint: Try setting device = cpu in your config, or install cuDNN.")

    def notify(self, title, message, icon="dialog-information", timeout=2000):
        """Send a desktop notification."""
        if not NOTIFICATIONS:
            return
        subprocess.run(
            [
                "notify-send",
                "-a", "SoupaWhisper",
                "-i", icon,
                "-t", str(timeout),
                "-h", "string:x-canonical-private-synchronous:soupawhisper",
                title,
                message
            ],
            capture_output=True
        )

    def start_recording(self):
        if self.recording or (self.backend == "local" and self.model_error):
            return

        self.recording = True
        self.temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        self.temp_file.close()

        # Record using arecord (ALSA) - works on most Linux systems
        self.record_process = subprocess.Popen(
            [
                "arecord",
                "-f", "S16_LE",  # Format: 16-bit little-endian
                "-r", "16000",   # Sample rate: 16kHz (what Whisper expects)
                "-c", "1",       # Mono
                "-t", "wav",
                self.temp_file.name
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        print("Recording...")
        hotkey_name = HOTKEY.name if hasattr(HOTKEY, 'name') else HOTKEY.char
        self.notify("Recording...", f"Release {hotkey_name.upper()} when done", "audio-input-microphone", 30000)

    def stop_recording(self):
        if not self.recording:
            return

        self.recording = False

        if self.record_process:
            self.record_process.terminate()
            self.record_process.wait()
            self.record_process = None

        print("Transcribing...")
        if self.backend == "groq":
            self.notify("Transcribing...", "Sending to Groq API", "emblem-synchronizing", 30000)
        else:
            self.notify("Transcribing...", "Processing your speech", "emblem-synchronizing", 30000)

        # Wait for model if not loaded yet (local backend)
        self.model_loaded.wait()

        if self.backend == "local" and self.model_error:
            print(f"Cannot transcribe: model failed to load")
            self.notify("Error", "Model failed to load", "dialog-error", 3000)
            return

        # Transcribe
        try:
            if self.backend == "groq":
                text = transcribe_groq(self.temp_file.name)
            else:
                segments, info = self.model.transcribe(
                    self.temp_file.name,
                    beam_size=5,
                    vad_filter=True,
                )
                text = " ".join(segment.text.strip() for segment in segments)

            if text:
                # Copy to clipboard using xclip
                process = subprocess.Popen(
                    ["xclip", "-selection", "clipboard"],
                    stdin=subprocess.PIPE
                )
                process.communicate(input=text.encode())

                # Type it into the active input field
                if AUTO_TYPE:
                    subprocess.run(["xdotool", "type", "--clearmodifiers", text])

                print(f"Copied: {text}")
                self.notify("Copied!", text[:100] + ("..." if len(text) > 100 else ""), "emblem-ok-symbolic", 3000)
            else:
                print("No speech detected")
                self.notify("No speech detected", "Try speaking louder", "dialog-warning", 2000)

        except Exception as e:
            print(f"Error: {e}")
            self.notify("Error", str(e)[:50], "dialog-error", 3000)
        finally:
            # Cleanup temp file
            if self.temp_file and os.path.exists(self.temp_file.name):
                os.unlink(self.temp_file.name)

    def on_press(self, key):
        if key == HOTKEY:
            self.start_recording()

    def on_release(self, key):
        if key == HOTKEY:
            self.stop_recording()

    def stop(self):
        print("\nExiting...")
        self.running = False
        os._exit(0)

    def run(self):
        with keyboard.Listener(
            on_press=self.on_press,
            on_release=self.on_release
        ) as listener:
            listener.join()


def check_dependencies():
    """Check that required system commands are available."""
    missing = []

    for cmd in ["arecord", "xclip"]:
        if subprocess.run(["which", cmd], capture_output=True).returncode != 0:
            pkg = "alsa-utils" if cmd == "arecord" else cmd
            missing.append((cmd, pkg))

    if AUTO_TYPE:
        if subprocess.run(["which", "xdotool"], capture_output=True).returncode != 0:
            missing.append(("xdotool", "xdotool"))

    if missing:
        print("Missing dependencies:")
        for cmd, pkg in missing:
            print(f"  {cmd} - install with: sudo apt install {pkg}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="SoupaWhisper - Push-to-talk voice dictation"
    )
    parser.add_argument(
        "-v", "--version",
        action="version",
        version=f"SoupaWhisper {__version__}"
    )
    parser.parse_args()

    print(f"SoupaWhisper v{__version__}")
    print(f"Config: {CONFIG_PATH}")

    check_dependencies()

    dictation = Dictation()

    # Handle Ctrl+C gracefully
    def handle_sigint(sig, frame):
        dictation.stop()

    signal.signal(signal.SIGINT, handle_sigint)

    dictation.run()


if __name__ == "__main__":
    main()
