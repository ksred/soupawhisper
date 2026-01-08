#!/usr/bin/env python3

import argparse
import configparser
import subprocess
import tempfile
import threading
import signal
import sys
import os
import queue
import time
import wave
import logging
from pathlib import Path
from typing import Optional, Iterable, Tuple
from concurrent.futures import ThreadPoolExecutor
from faster_whisper.transcribe import Segment, Word

import numpy as np
import pyaudio
import webrtcvad

from pynput import keyboard
from faster_whisper import WhisperModel

__version__ = "0.1.0"

# Logger will be configured in main()
logger = logging.getLogger(__name__)

# Load configuration
CONFIG_PATH = Path.home() / ".config" / "soupawhisper" / "config.ini"
DEFAULT_HOTKEY = "f12"


def load_config():
    config = configparser.ConfigParser()

    if CONFIG_PATH.exists():
        config.read(CONFIG_PATH)

    return {
        # Whisper
        "model": config.get("whisper", "model", fallback="base.en"),
        "device": config.get("whisper", "device", fallback="cpu"),
        "compute_type": config.get("whisper", "compute_type", fallback="int8"),
        # Hotkey
        "key": config.get("hotkey", "key", fallback=DEFAULT_HOTKEY),
        # Behavior
        "default_streaming": config.getboolean("behavior", "default_streaming", fallback=True),
        "notifications": config.getboolean("behavior", "notifications", fallback=True),
        "clipboard": config.getboolean("behavior", "clipboard", fallback=True),
        "auto_type": config.getboolean("behavior", "auto_type", fallback=True),
        "auto_sentence": config.getboolean("behavior", "auto_sentence", fallback=True),
        "typing_delay": config.getfloat("behavior", "typing_delay", fallback=0.01),
        "save_recordings": config.getboolean("behavior", "save_recordings", fallback=False),
        # Streaming
        "vad_silence_threshold_seconds": config.getfloat("streaming", "vad_silence_threshold_seconds", fallback=1.0),
        "vad_sample_rate": config.getint("streaming", "vad_sample_rate", fallback=16000),
        "vad_chunk_size_ms": config.getfloat("streaming", "vad_chunk_size_ms", fallback=512.0),
        "vad_threshold": config.getfloat("streaming", "vad_threshold", fallback=0.5),
        "audio_input_device": config.get("streaming", "audio_input_device", fallback=None),
    }


def get_hotkey(key_name):
    """Map key name to pynput key."""
    key_name = key_name.lower()
    if hasattr(keyboard.Key, key_name):
        return getattr(keyboard.Key, key_name)
    elif len(key_name) == 1:
        return keyboard.KeyCode.from_char(key_name)
    else:
        logger.warning(f"Unknown key: {key_name}, defaulting to {DEFAULT_HOTKEY}")
        return get_hotkey(DEFAULT_HOTKEY)


class Typer:
    """Types and removes characters in any inputs via xdotool."""

    def __init__(self, delay_ms: int = 10, start_delay_ms: int = 250):
        self.delay_ms = max(1, int(delay_ms))
        self.start_delay_ms = int(start_delay_ms)
        self.enabled = subprocess.run(["which", "xdotool"], capture_output=True).returncode == 0
        if not self.enabled:
            logger.warning("[typer] xdotool not found, typing disabled")

    def type_rewrite(self, text: str, previous_length: int = 0):
        """
        Type text using xdotool.

        Args:
            text: The text to type.
            previous_length: The number of characters to delete before typing the new text.
        """
        if not self.enabled or not text:
            return
        if self.start_delay_ms > 0:
            time.sleep(self.start_delay_ms / 1000.0)
        # Remove previous characters if needed.
        if previous_length > 0:
            subprocess.run(
                ["xdotool", "key", "BackSpace", "--clearmodifiers", "--repeat", str(previous_length)],#, "--repeat-delay", str(self.delay_ms)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=2
            )
        # Type the new text.
        subprocess.run(
            ["xdotool", "type", "--delay", str(self.delay_ms), "--clearmodifiers", text],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=10
        )


class Dictation:
    def __init__(self, config: dict):
        self.config = config
        self.hotkey = get_hotkey(config["key"])
        self.recording = False
        self.record_process = None
        self.temp_file = None
        self.model = None
        self.model_loaded = threading.Event()
        self.model_error = None
        self.running = True
        self.typer: Optional[Typer] = None

        if self.config["auto_type"]:
            self.typer = Typer(
                delay_ms=int(self.config["typing_delay"] * 1000),
                start_delay_ms=100  # Delay to avoid modifiers from hotkey
            )

        # Load model in background.
        logger.debug(f"Loading Whisper model ({config['model']})...")
        threading.Thread(target=self._load_model, daemon=True).start()

    def get_hotkey_name(self):
        return getattr(self.hotkey, 'name', None) or getattr(self.hotkey, 'char', DEFAULT_HOTKEY)

    def _load_model(self):
        try:
            self.model = WhisperModel(self.config["model"], device=self.config["device"], compute_type=self.config["compute_type"])
            self.model_loaded.set()
            logger.info(f"Model {self.config['model']} ({self.config['device']}, {self.config['compute_type']}) loaded.")
            self._finish_model_loading()
        except Exception as e:
            self.model_error = str(e)
            self.model_loaded.set()
            logger.error(f"Failed to load model: {e}", exc_info=True)
            if "cudnn" in str(e).lower() or "cuda" in str(e).lower():
                logger.info("Hint: Try setting device = cpu in your config, or install cuDNN.")

    def _finish_model_loading(self):
        logger.info(f"Hold [{self.get_hotkey_name()}] to start dictation, release to transcribe. Press Ctrl+C to quit.")

    def notify(self, title, message, icon="dialog-information", timeout=2000):
        """Send a desktop notification."""
        if not self.config["notifications"]:
            return
        logger.debug("[idle] Showing notification: %s, %s, %s, %s", title, message, icon, timeout)
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

    def _segments_to_text(self, segments: Iterable[Segment], auto_sentence: bool) -> str:
        """Format text as a sentence: capitalize first letter and add period at end if needed."""
        text = " ".join(segment.text.strip() for segment in segments)
        if not text or not self.config.get("auto_sentence", False):
            return text
        text = text.strip()
        if not text:
            return text
        if auto_sentence and len(text) > 0:
            text = text[0].upper() + text[1:]
        if auto_sentence and not text.endswith(('.', '!', '?', ':', ';')):
            text = text + '.'
        return text

    def _transcribe_audio_file(self, audio_file_path: str) -> Tuple[str, float]:
        """
        Wait for model to load and transcribe an audio file.
        
        Args:
            audio_file_path: Path to the audio file to transcribe
            
        Returns:
            Tuple of (transcribed_text, audio_duration)
            
        Raises:
            RuntimeError: If model failed to load or is not available
            FileNotFoundError: If audio file does not exist
            Exception: If transcription fails
        """
        # Wait for model loading to finish.
        self.model_loaded.wait()
        if self.model_error:
            logger.error("Cannot transcribe: model failed to load")
            self.notify("Error", "Model failed to load", "dialog-error", 3000)
            return "", 0.0
        if self.model is None:
            logger.error("Cannot transcribe: model not loaded")
            return "", 0.0
        # Transcribe audio.
        segments, info = self.model.transcribe(
            audio_file_path,
            beam_size=5,
            vad_filter=True,
        )
        # Convert segments to text.
        text = self._segments_to_text(segments, self.config["auto_sentence"])
        return text, info.duration

    def on_press(self, key):
        if key == self.hotkey:
            self.start_recording()

    def on_release(self, key):
        if key == self.hotkey:
            self.stop_recording()

    def stop(self):
        logger.info("\nExiting...")
        self.running = False

    def run(self):
        with keyboard.Listener(
            on_press=self.on_press,
            on_release=self.on_release,
        ) as listener:
            listener.join()

    def start_record_process(self, output_file: str, duration: Optional[float] = None, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL):
        """
        Start arecord process with shared parameters, writing WAV file.
        
        Args:
            output_file: Path to write WAV file to
            duration: Optional duration in seconds (if None, records until terminated)
            stdout: Where to redirect stdout (default: DEVNULL)
            stderr: Where to redirect stderr (default: DEVNULL)
        
        Returns:
            subprocess.Popen instance
        """
        cmd = [
            "arecord",
            "-f", "S16_LE",  # Format: 16-bit little-endian
            "-r", "16000",   # Sample rate: 16kHz (what Whisper expects)
            "-c", "1",       # Mono
            "-t", "wav",
        ]
        if duration is not None:
            cmd.extend(["-d", str(int(duration))])
        cmd.append(output_file)
        return subprocess.Popen(
            cmd,
            stdout=stdout,
            stderr=stderr
        )

    def start_recording(self):
        if self.recording:
            return
        self.model_loaded.wait()
        if self.model_error or self.model is None:
            logger.error("Recording is not ready yet.")
            return

        self.recording = True
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.temp_file = os.path.join(tempfile.gettempdir(), f"recording_{timestamp}.wav")

        # Record using arecord (ALSA) - works on most Linux systems
        self.record_process = self.start_record_process(self.temp_file)
        logger.info("[record] Recording...")
        self.notify("Recording...", f"Release {self.get_hotkey_name().upper()} when done", "audio-input-microphone", 2000)

    def stop_recording(self):
        if not self.recording:
            return

        self.recording = False

        if self.record_process:
            self.record_process.terminate()
            self.record_process.wait()
            self.record_process = None

        logger.info("Recording stopped, transcribing...")
        self.notify("Transcribing...", "Processing your speech", "emblem-synchronizing", 1500)

        # Transcribe.
        temp_file_name = self.temp_file if self.temp_file else None
        if temp_file_name and not isinstance(temp_file_name, str):
            if hasattr(temp_file_name, 'name'):
                temp_file_name = temp_file_name.name
            else:
                temp_file_name = str(temp_file_name) if temp_file_name else None
        if temp_file_name and not isinstance(temp_file_name, str):
            temp_file_name = None
        try:
            if not temp_file_name or not isinstance(temp_file_name, str) or not os.path.exists(temp_file_name):
                logger.error(f"Cannot transcribe: audio file not found: {temp_file_name}")
                return
            # Transcribe audio.
            text, duration = self._transcribe_audio_file(temp_file_name)
            if text:
                logger.info(f"Transcribed {duration:.2f}s: {text}")
                if self.config["clipboard"]:
                    process = subprocess.Popen(
                        ["xclip", "-selection", "clipboard"],
                        stdin=subprocess.PIPE
                    )
                    process.communicate(input=text.encode())
                    logger.info(f"Pasted to clipboard: {text}")
                if self.config["auto_type"] and self.typer:
                    self.typer.type_rewrite(text, 0)
                if self.config["notifications"]:
                    self.notify(
                        f"Transcribed {duration:.2f}s speech:",
                        text[:100] + ("..." if len(text) > 100 else ""),
                        "emblem-ok-symbolic",
                        3000
                    )
            else:
                logger.info("No speech detected")
                self.notify("No speech detected", "Check your microphone or try speaking louder", "dialog-warning", 2000)
        except Exception as e:
            logger.error(f"Error transcribing: {e}", exc_info=True)
            self.notify("Error", str(e)[:50], "dialog-error", 3000)
        finally:
            # Cleanup temp file unless save_recordings is enabled.
            if temp_file_name and isinstance(temp_file_name, str) and not self.config["save_recordings"]:
                if os.path.exists(temp_file_name):
                    os.unlink(temp_file_name)

    def transcribe_file(self, wav_file_path: str) -> str:
        """
        Transcribe a WAV file using non-streaming transcription.
        Expects WAV files (16-bit, 16kHz, mono).

        Args:
            wav_file_path: Path to the WAV file to transcribe

        Returns:
            The transcribed text
        """
        logger.info(f"[file] Transcribing WAV file: {wav_file_path}")
        try:
            text, duration = self._transcribe_audio_file(wav_file_path)
            if text:
                logger.info(f"[file] Transcribed {duration:.2f}s: {text}")
            else:
                logger.info("[file] No speech detected")
            return text
        except Exception as e:
            logger.error(f"[file] Error transcribing: {e}", exc_info=True)
            raise


class StreamingDictation(Dictation):
    """Streaming dictation mode.
    """

    def __init__(self, config: dict):
        # Initialize base class (sets up config, hotkey, model loading, etc.)
        super().__init__(config)
        assert self.config["auto_type"], "auto_type must be True for streaming mode"
        self.vad_silence_threshold_seconds = config["vad_silence_threshold_seconds"]
        self.vad_sample_rate = config["vad_sample_rate"]
        self.vad_chunk_size_ms = config["vad_chunk_size_ms"]
        self.vad_threshold = config["vad_threshold"]
        self.streaming_match_words_threshold_seconds = 0.1

        # Validate vad_chunk_size_ms - webrtcvad only supports 10ms, 20ms, or 30ms
        if self.vad_chunk_size_ms not in [10, 20, 30]:
            raise ValueError(f"vad_chunk_size_ms must be 10, 20, or 30 (got {self.vad_chunk_size_ms}). webrtcvad only supports these frame sizes.")

        self.transcription_queue: queue.Queue[np.ndarray | None] = queue.Queue()
        self.typing_queue: queue.Queue[tuple[str, int] | None] = queue.Queue()
        self.file_saving_queue: queue.Queue[tuple[np.ndarray, int] | None] = queue.Queue()

        self.stopping = False
        self.audio_interface: Optional[pyaudio.PyAudio] = None
        self.audio_stream = None
        self.audio_thread: Optional[threading.Thread] = None
        self.transcription_thread: Optional[threading.Thread] = None
        self.typing_thread: Optional[threading.Thread] = None
        self.file_saving_thread: Optional[threading.Thread] = None
        self.vad = webrtcvad.Vad(int(self.vad_threshold))
        self.vad_frame_size_ms = self.vad_chunk_size_ms
        self.vad_frame_size = int(self.vad_sample_rate * self.vad_frame_size_ms / 1000)
        self.in_speech = False
        self.silence_duration = 0.0
        self.current_segment_chunks: list[np.ndarray] = []
        self.accumulated_text = ""
        self._device_help_shown = False
        self.speech_start_time: Optional[float] = None
        self.total_processed_time = 0.0
        self.file_mode = False
        self.consecutive_speech_chunks = 0
        self.min_speech_chunks = 5  # TODO move to config
        self.speech_start_logged = False
        self.last_speech_time: Optional[float] = None

        # Create typer once in constructor
        self.typer: Optional[Typer] = None
        if self.config["auto_type"]:
            self.typer = Typer(
                delay_ms=int(self.config["typing_delay"] * 1000),
                start_delay_ms=100
            )

    def _finish_model_loading(self):
        logger.info(f"Press [{self.get_hotkey_name().upper()}] to start transcribing, press one more time to stop. Press Ctrl+C to quit.")

    def _get_input_device_index(self) -> Optional[int]:
        """Get the input device index from config, or None to use default."""
        audio_input_device = self.config.get("audio_input_device")
        if not audio_input_device or not isinstance(audio_input_device, str):
            return None
        audio_input_device = audio_input_device.strip()
        if not audio_input_device:
            return None
        try:
            device_index = int(audio_input_device)
            return device_index
        except ValueError:
            if self.audio_interface is None:
                self.audio_interface = pyaudio.PyAudio()
            for i in range(self.audio_interface.get_device_count()):
                try:
                    info = self.audio_interface.get_device_info_by_index(i)
                    max_inputs = int(info.get('maxInputChannels', 0))
                    device_name = str(info.get('name', ''))
                    if max_inputs > 0 and audio_input_device.lower() in device_name.lower():
                        logger.info(f"[record] Found audio device matching '{audio_input_device}': {i} - {device_name}")
                        return i
                except Exception:
                    pass
            logger.warning(f"[record] Audio device '{audio_input_device}' not found, using default")
            return None

    def _show_device_help(self):
        """Show available audio devices and instructions when no audio is detected."""
        if self._device_help_shown:
            return
        self._device_help_shown = True
        if self.audio_interface is None:
            self.audio_interface = pyaudio.PyAudio()
        logger.error("[record] No audio detected. Available input devices:")
        devices_list = []
        for i in range(self.audio_interface.get_device_count()):
            try:
                info = self.audio_interface.get_device_info_by_index(i)
                max_inputs = int(info.get('maxInputChannels', 0))
                if max_inputs > 0:
                    device_name = str(info.get('name', ''))
                    devices_list.append(f"  {i}: {device_name}")
                    logger.error(f"  {i}: {device_name}")
            except Exception:
                pass
        devices_text = "\n".join(devices_list[:5])
        if len(devices_list) > 5:
            devices_text += f"\n  ... and {len(devices_list) - 5} more"
        config_path = CONFIG_PATH
        message = f"Set 'audio_input_device' in {config_path}\n\nExample devices:\n{devices_text}"
        self.notify("No audio detected - check device", message, "dialog-warning", 5000)

    def on_press(self, key):
        if key == self.hotkey:
            if not self.recording:
                self.start_recording()
            else:
                self.stop_recording()

    def run(self):
        with keyboard.Listener(
            on_press=self.on_press
        ) as listener:
            listener.join()

    def start_recording(self):
        if self.recording:
            logger.error("[record] Recording is already started")
            return
        if self.stopping:
            logger.error("[record] Cannot start recording: previous recording is still shutting down")
            self.notify("Error", "Previous recording is still shutting down. Please wait a moment.", "dialog-error", 3000)
            return
        self.model_loaded.wait()
        if self.model_error or self.model is None:
            logger.error("[record] Cannot start recording yet")
            self.notify("Error", "Model is not loaded yet", "dialog-error", 3000)
            return
        # Reset transcription state.
        self.recording = True
        self.accumulated_text = ""
        self.in_speech = False
        self.silence_duration = 0.0
        self.current_segment_chunks = []
        self._device_help_shown = False
        self.total_processed_time = 0.0
        self.speech_start_time = None
        self.last_speech_time = None
        # Clear queues to remove any leftover sentinels from previous recording
        while not self.transcription_queue.empty():
            try:
                self.transcription_queue.get_nowait()
            except queue.Empty:
                break
        while not self.typing_queue.empty():
            try:
                self.typing_queue.get_nowait()
            except queue.Empty:
                break
        if self.audio_interface is None:
            self.audio_interface = pyaudio.PyAudio()
        input_device_index = self._get_input_device_index()
        if input_device_index is not None:
            device_info = self.audio_interface.get_device_info_by_index(input_device_index)
            logger.info(f"[record] Using audio input device {input_device_index}: {device_info['name']}")
        else:
            default_device = self.audio_interface.get_default_input_device_info()
            logger.info(f"[record] Using default audio input device {default_device['index']}: {default_device['name']}")
            input_device_index = int(default_device['index'])
        frames_per_buffer = int(self.vad_sample_rate * self.vad_chunk_size_ms / 1000.0)
        try:
            self.audio_stream = self.audio_interface.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.vad_sample_rate,
                input=True,
                input_device_index=input_device_index,
                frames_per_buffer=frames_per_buffer
            )
        except OSError as e:
            logger.error(f"[record] Failed to open audio stream: {e}")
            logger.error(f"[record] Available input devices:")
            for i in range(self.audio_interface.get_device_count()):
                try:
                    info = self.audio_interface.get_device_info_by_index(i)
                    max_inputs = int(info.get('maxInputChannels', 0))
                    if max_inputs > 0:
                        device_name = str(info.get('name', ''))
                        logger.error(f"  {i}: {device_name} (inputs: {max_inputs})")
                except Exception:
                    pass
            self.recording = False
            self.notify("Error", f"Failed to open audio device: {str(e)[:50]}...", "dialog-error", 10000)
            return
        self.audio_thread = threading.Thread(
            target=self._continuous_audio_stream_worker,
            args=(frames_per_buffer,),
            daemon=True,
            name="audio_stream_worker"
        )
        self.audio_thread.start()
        self.transcription_thread = threading.Thread(target=self._transcription_worker, daemon=True)
        self.transcription_thread.start()
        if self.config["auto_type"]:
            self.typer = Typer(
                delay_ms=int(self.config["typing_delay"] * 1000),
                start_delay_ms=100
            )
            self.typing_thread = threading.Thread(target=self._typing_worker, daemon=True)
            self.typing_thread.start()
        if self.config.get("save_recordings", False):
            self.file_saving_thread = threading.Thread(target=self._file_saving_worker, daemon=True)
            self.file_saving_thread.start()
        # Notify about recording start.
        logger.info("[record] Recording (streaming mode)...")
        self.notify("Recording...", f"Press {self.get_hotkey_name().upper()} when done", "audio-input-microphone", 1500)

    def _finalize_segment(self):
        """Finalize any remaining speech segment after processing is complete."""
        if self.in_speech and self.current_segment_chunks:
            segment = np.concatenate(self.current_segment_chunks)
            segment_audio_duration = len(segment) / float(self.vad_sample_rate)
            speech_end_time = self.last_speech_time if self.last_speech_time is not None else self.total_processed_time
            if self.speech_start_time is not None:
                logger.info(f"[SAD] {speech_end_time:.3f} speech finished, handling chunk of {segment_audio_duration:.3f} seconds audio")
            self.transcription_queue.put(segment)
            self.in_speech = False
            self.current_segment_chunks = []
            self.speech_start_time = None
            self.last_speech_time = None

    def _continuous_audio_stream_worker(self, frames_per_buffer: int):
        chunk_duration = frames_per_buffer / float(self.vad_sample_rate)
        try:
            while self.recording:
                if not self.audio_stream:
                    break
                data = self.audio_stream.read(frames_per_buffer, exception_on_overflow=False)
                if not data:
                    continue
                samples = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
                self._process_audio_chunk(samples, chunk_duration)
        except Exception as e:
            logger.error(f"[record] Error in audio stream worker: {e}", exc_info=True)
        finally:
            self._finalize_segment()

    def _process_audio_chunk(self, frame: np.ndarray, chunk_duration: float):
        """
        Process a single audio chunk using VAD to determine segment boundaries.
        All chunks (both speech and non-speech) are included in segments between
        "speech started" and "speech finished" events.
        """
        chunk_start_time = self.total_processed_time
        self.total_processed_time += chunk_duration
        # Use VAD to detect if this chunk contains speech.
        has_speech = False
        try:
            frame_int16 = (frame * 32768.0).astype(np.int16)
            frame_bytes = frame_int16.tobytes()
            if len(frame_bytes) == 2 * self.vad_frame_size:
                has_speech = self.vad.is_speech(frame_bytes, self.vad_sample_rate)
            else:
                logger.error(f"[vad] Invalid frame size: expected {2 * self.vad_frame_size} bytes, got {len(frame_bytes)}")
        except Exception as e:
            logger.error(f"[vad] VAD processing failed: {e}", exc_info=True)
            has_speech = False
        # Handle frame data.
        if has_speech:
            # Branch: Chunk contains speech
            self.consecutive_speech_chunks += 1
            if not self.in_speech:
                # Sub-branch: Not currently in a speech segment
                # Check if we have enough consecutive speech chunks to start a segment
                if self.consecutive_speech_chunks >= self.min_speech_chunks:
                    # Start a new speech segment
                    self.in_speech = True
                    self.current_segment_chunks = []
                    self.silence_duration = 0.0
                    self.speech_start_logged = False
                    self.last_speech_time = None
                    self.speech_start_time = chunk_start_time
                    # Include current chunk in the segment
                    self.current_segment_chunks.append(frame)
            else:
                # Sub-branch: Already in a speech segment
                # Add chunk to current segment (includes all chunks, not just speech)
                self.current_segment_chunks.append(frame)
                self.silence_duration = 0.0
                self.last_speech_time = chunk_start_time
                # Log "speech started" message after 0.3s delay to avoid logging on every chunk
                if not self.speech_start_logged and self.speech_start_time is not None:
                    time_since_speech_start = chunk_start_time - self.speech_start_time
                    if time_since_speech_start >= 0.3:
                        logger.info(f"[SAD] {self.speech_start_time:.3f} speech started")
                        self.speech_start_logged = True
        else:
            # Branch: Chunk does not contain speech
            self.consecutive_speech_chunks = 0
            if self.in_speech:
                # Sub-branch: Currently in a speech segment
                # Add chunk to segment (include non-speech chunks too)
                self.current_segment_chunks.append(frame)
                self.silence_duration += chunk_duration
                # Early abort: If segment just started (< 1.0s) and we get 0.2s of silence,
                # abort the segment to avoid false positives
                if self.speech_start_time is not None:
                    time_since_speech_start = chunk_start_time - self.speech_start_time
                    if time_since_speech_start < 1.0 and self.silence_duration >= 0.2:
                        self.in_speech = False
                        self.current_segment_chunks = []
                        self.silence_duration = 0.0
                        self.speech_start_time = None
                        self.speech_start_logged = False
                        self.last_speech_time = None
                        return
                # Check if silence duration exceeds threshold - time to finalize segment
                if self.silence_duration >= self.vad_silence_threshold_seconds:
                    if self.current_segment_chunks:
                        # Concatenate all chunks (speech + non-speech) into a single segment
                        segment: np.ndarray = np.concatenate(self.current_segment_chunks)
                        segment_audio_duration = len(segment) / float(self.vad_sample_rate)
                        speech_end_time = self.last_speech_time if self.last_speech_time is not None else chunk_start_time
                        segment_time_span = speech_end_time - self.speech_start_time if self.speech_start_time is not None else segment_audio_duration
                        min_segment_duration = 0.5
                        # Only send segments that meet minimum duration requirement
                        if segment_audio_duration >= min_segment_duration:
                            if self.speech_start_time is not None:
                                logger.info(f"[SAD] {speech_end_time:.3f} speech finished, handling chunk of {segment_audio_duration:.3f} seconds audio (span: {segment_time_span:.3f}s)")
                            # Send complete segment (all chunks) to transcription queue
                            self.transcription_queue.put(segment)
                            if not self.file_mode and self.config.get("save_recordings", False):
                                self.file_saving_queue.put((segment, self.vad_sample_rate))
                        else:
                            logger.debug(f"[SAD] Dropping segment too short ({segment_audio_duration:.3f}s < {min_segment_duration}s)")
                    # Reset segment state
                    self.in_speech = False
                    self.current_segment_chunks = []
                    self.silence_duration = 0.0
                    self.speech_start_time = None
                    self.last_speech_time = None

    def _transcription_worker(self):
        """Transcription worker thread - processes chunks in order."""
        chunk_idx = 0
        while self.recording or not self.transcription_queue.empty():
            try:
                segment: np.ndarray | None = self.transcription_queue.get(timeout=0.1)
                if segment is None:
                    break
                if self.model is None:
                    logger.error("[transcriber] Model not loaded")
                    continue
                trans_start = time.time()
                segments, info = self.model.transcribe(
                    segment,
                    beam_size=5,
                    vad_filter=False,
                )
                trans_duration = time.time() - trans_start
                text = self._segments_to_text(segments, False)
                if not text:
                    logger.info(f"[transcriber] Empty transcription (transcribed in {trans_duration:.2f}s)")
                    if not self._device_help_shown:
                        self._show_device_help()
                    continue
                else:
                    logger.info(f"[transcriber] Transcribed {len(segment) / float(self.vad_sample_rate):.2f}s in {trans_duration:.2f}s: {text}")
                self.accumulated_text += text
                chunk_idx += 1
                if not self.file_mode and self.typing_queue:
                    self.typing_queue.put((text, chunk_idx))
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"[transcriber] {e}", exc_info=True)

    def _typing_worker(self):
        """Typing worker thread - types text in order of chunks."""
        while self.recording or not self.typing_queue.empty():
            if not self.typer:
                logger.error("[typer] Typer is not initialized")
                self.notify("Error", "Typer is not initialized", "dialog-error", 3000)
                return
            try:
                typing_task = self.typing_queue.get(timeout=0.1)
                # Check for signal to stop typing.
                if typing_task is None:
                    break
                # Get typing task and log it.
                text_to_type, chunk_idx = typing_task
                logger.info(f"[typer] Chunk {chunk_idx} typing: {text_to_type}")
                # Run typing.
                try:
                    self.typer.type_rewrite(text_to_type, 0)
                except Exception as e:
                    logger.error(f"[typer] Typing failed: {e}", exc_info=True)
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"[typer] {e}", exc_info=True)

    def _file_saving_worker(self):
        """File saving worker thread - saves audio segments to files asynchronously."""
        while self.recording or not self.file_saving_queue.empty():
            try:
                task = self.file_saving_queue.get(timeout=0.1)
                if task is None:
                    break
                segment, sample_rate = task
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                file_path = os.path.join(tempfile.gettempdir(), f"stream_chunk_{timestamp}.wav")
                try:
                    with wave.open(file_path, "wb") as wf:
                        wf.setnchannels(1)
                        wf.setsampwidth(2)
                        wf.setframerate(sample_rate)
                        wf.writeframes((segment * 32768.0).astype(np.int16).tobytes())
                except Exception as e:
                    logger.error(f"[record] Failed to save streaming chunk: {e}", exc_info=True)
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"[file_saver] {e}", exc_info=True)

    def stop_recording(self):
        if not self.recording:
            return
        self.recording = False
        self.stopping = True
        # Notify about stopping.
        logger.info("[idle] Recording stopped, transcribing remaining chunks...")
        self.notify("Transcribing stopped", "Processing remaining chunks", "emblem-synchronizing", 1500)
        if self.audio_thread:
            self.audio_thread.join(timeout=2.0)
        if self.audio_stream:
            try:
                self.audio_stream.stop_stream()
                self.audio_stream.close()
            except Exception:
                pass
            self.audio_stream = None
        if self.audio_interface:
            try:
                self.audio_interface.terminate()
            except Exception:
                pass
            self.audio_interface = None
        # Signal workers to stop (they will process all remaining items before exiting).
        self.transcription_queue.put(None)
        if not self.file_mode and self.config["auto_type"]:
            self.typing_queue.put(None)
        if self.config.get("save_recordings", False):
            self.file_saving_queue.put(None)
        # Join worker threads.
        if self.transcription_thread:
            self.transcription_thread.join(timeout=5.0)
        if self.typing_thread:
            self.typing_thread.join(timeout=1.0)
        if self.file_saving_thread:
            self.file_saving_thread.join(timeout=1.0)
        self.stopping = False
        # Finalize any remaining chunks
        final_text = self.accumulated_text.strip()
        if final_text:
            if self.config["clipboard"]:
                process = subprocess.Popen(
                    ["xclip", "-selection", "clipboard"],
                    stdin=subprocess.PIPE
                )
                process.communicate(input=final_text.encode())
                logger.info(f"[idle] Pasted to clipboard: {final_text}")
            logger.info(f"[idle] Final text: {final_text}")
            self.notify("Got:", final_text[:100] + ("..." if len(final_text) > 100 else ""), "emblem-ok-symbolic", 3000)
        else:
            logger.info("[idle] No speech detected")
            if not self._device_help_shown:
                self._show_device_help()
            else:
                self.notify("No speech detected", "Try speaking louder or check audio device", "dialog-warning", 2000)

    def transcribe_file(self, wav_file_path: str) -> str:
        """
        Transcribe a WAV file using the streaming transcription pipeline.
        Expects WAV files created by Dictation (16-bit, 16kHz, mono).

        Args:
            wav_file_path: Path to the WAV file to transcribe

        Returns:
            The transcribed text
        """
        logger.info(f"[file] Transcribing WAV file: {wav_file_path}")
        # Set file mode to skip typing. No other preparations needed.
        self.file_mode = True
        # Read WAV file (expects 16-bit, 16kHz, mono).
        try:
            with wave.open(wav_file_path, "rb") as wf:
                n_frames = wf.getnframes()
                audio_data = wf.readframes(n_frames)
                audio_buffer = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        except Exception as e:
            raise RuntimeError(f"Failed to read WAV file: {e}")
        # Process audio in chunks similar to streaming mode.
        frames_per_buffer = int(self.vad_sample_rate * self.vad_chunk_size_ms / 1000.0)
        chunk_duration = frames_per_buffer / float(self.vad_sample_rate)
        # Start transcription worker thread (no typing worker for file mode).
        self.recording = True
        transcription_thread = threading.Thread(target=self._transcription_worker, daemon=True)
        transcription_thread.start()
        try:
            # Process audio samples in chunks.
            for i in range(0, len(audio_buffer), frames_per_buffer):
                if not self.recording:
                    break
                chunk = audio_buffer[i:i + frames_per_buffer]
                if len(chunk) < frames_per_buffer:
                    chunk = np.pad(chunk, (0, frames_per_buffer - len(chunk)), mode='constant')
                self._process_audio_chunk(chunk, chunk_duration)
            # Finalize any remaining speech segment.
            self._finalize_segment()
            # Signal transcription worker to stop.
            self.recording = False
            self.transcription_queue.put(None)
            # Wait for transcription to complete.
            transcription_thread.join(timeout=30.0)
            if transcription_thread.is_alive():
                logger.warning("[file] Transcription thread did not finish in time")
            final_text = self.accumulated_text.strip()
            logger.info(f"[file] Transcription complete: {final_text}")
            return final_text
        except Exception as e:
            logger.error(f"[file] Error during transcription: {e}", exc_info=True)
            raise
        finally:
            self.file_mode = False

    def stop(self):
        logger.info("\nExiting...")
        self.running = False
        if self.recording:
            self.stop_recording()


def check_dependencies(config: dict):
    """Check that required system commands are available."""
    missing = []

    required_cmds = ["arecord"]
    if config["clipboard"]:
        required_cmds.append("xclip")

    for cmd in required_cmds:
        if subprocess.run(["which", cmd], capture_output=True).returncode != 0:
            pkg = "alsa-utils" if cmd == "arecord" else cmd
            missing.append((cmd, pkg))

    if config["auto_type"]:
        if subprocess.run(["which", "xdotool"], capture_output=True).returncode != 0:
            missing.append(("xdotool", "xdotool"))

    # Python module dependencies
    try:
        import webrtcvad  # type: ignore
    except ImportError:
        logger.error("Python dependency missing: webrtcvad (install with: pip install webrtcvad)")
        sys.exit(1)
    if config.get("default_streaming", False):
        try:
            import pyaudio  # type: ignore
        except ImportError:
            logger.error("Python dependency missing: pyaudio (install with: pip install pyaudio)")
            sys.exit(1)

    if missing:
        logger.error("Missing dependencies:")
        for cmd, pkg in missing:
            logger.error(f"  {cmd} - install with: sudo apt install {pkg}")
        sys.exit(1)


def get_model_cache_path():
    """Get the path where faster-whisper models are cached."""
    cache_home = os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache"))
    return os.path.join(cache_home, "huggingface", "hub")


def main():
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s.%(msecs)d [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Load and print configuration
    config = load_config()
    logger.info("Configuration:")
    for key, value in config.items():
        logger.info(f"  {key}: {value}")

    # Prepare arguments parser.
    cache_path = get_model_cache_path()
    description = f"""SoupaWhisper - voice dictation tool.

Works in both streaming and non-streaming modes.
- Non-streaming mode: push-to-talk, text is available only at the end of recording, good quality transcription.
- Streaming mode: press to toggle transcribing, text is appearing incrementally as you speak (by small chunks), quality is lower.

Version: {__version__}
Config file: {CONFIG_PATH}
Model cache: {cache_path}

Available models: tiny, tiny.en, base, base.en, small, small.en, medium, medium.en, large-v1, large-v2, large-v3
"""
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "-v", "--version",
        action="version",
        version=f"SoupaWhisper {__version__}"
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Enable streaming transcription mode (default: from config)"
    )
    parser.add_argument(
        "--no-streaming",
        action="store_true",
        help="Disable streaming transcription mode (force non-streaming)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output"
    )
    parser.add_argument(
        "--file",
        type=str,
        metavar="WAV_FILE",
        help="Transcribe a WAV file and exit (uses streaming or non-streaming mode based on settings)"
    )
    args = parser.parse_args()
    check_dependencies(config)

    # Apply arguments.
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Determine streaming mode
    use_streaming = config["default_streaming"]
    if args.streaming:
        use_streaming = True
    elif args.no_streaming:
        use_streaming = False

    # Create dictation object.
    if use_streaming:
        dictation = StreamingDictation(config)
    else:
        dictation = Dictation(config)

    # Handle file transcription mode.
    if args.file:
        if not os.path.exists(args.file):
            raise FileNotFoundError(f"WAV file not found: {args.file}")
        dictation.model_loaded.wait()
        if dictation.model_error or dictation.model is None:
            logger.error("Cannot transcribe file: model failed to load")
            sys.exit(1)
        try:
            dictation.transcribe_file(args.file)
            sys.exit(0)
        except Exception as e:
            logger.error(f"Failed to transcribe file: {e}", exc_info=True)
            sys.exit(1)

    # Handle Ctrl+C gracefully
    def handle_sigint(sig, frame):
        dictation.stop()
        os._exit(0)

    signal.signal(signal.SIGINT, handle_sigint)

    # Start dictation.
    dictation.run()


if __name__ == "__main__":
    main()
