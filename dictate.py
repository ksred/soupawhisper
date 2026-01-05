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
from typing import Optional, Iterable
from concurrent.futures import ThreadPoolExecutor
from faster_whisper.transcribe import Segment

import numpy as np

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

    # Defaults
    defaults = {
        "model": "base.en",
        "device": "cpu",
        "compute_type": "int8",
        "key": DEFAULT_HOTKEY,
        "default_streaming": "true",
        "notifications": "true",
        "clipboard": "true",
        "auto_type": "true",
        "auto_sentence": "true",
        "typing_delay": "0.01",
        "streaming_chunk_seconds": "3.0",
        "streaming_overlap_seconds": "1.5",
        "streaming_match_words_threshold_seconds": "0.1",
    }

    if CONFIG_PATH.exists():
        config.read(CONFIG_PATH)

    return {
        # Whisper
        "model": config.get("whisper", "model", fallback=defaults["model"]),
        "device": config.get("whisper", "device", fallback=defaults["device"]),
        "compute_type": config.get("whisper", "compute_type", fallback=defaults["compute_type"]),
        # Hotkey
        "key": config.get("hotkey", "key", fallback=defaults["key"]),
        # Behavior
        "default_streaming": config.getboolean("behavior", "default_streaming", fallback=defaults["default_streaming"] == "true"),
        "notifications": config.getboolean("behavior", "notifications", fallback=True),
        "clipboard": config.getboolean("behavior", "clipboard", fallback=True),
        "auto_type": config.getboolean("behavior", "auto_type", fallback=True),
        "auto_sentence": config.getboolean("behavior", "auto_sentence", fallback=defaults["auto_sentence"] == "true"),
        "typing_delay": config.getfloat("behavior", "typing_delay", fallback=float(defaults["typing_delay"])),
        # Streaming
        "streaming_chunk_seconds": config.getfloat("streaming", "streaming_chunk_seconds", fallback=float(defaults["streaming_chunk_seconds"])),
        "streaming_overlap_seconds": config.getfloat("streaming", "streaming_overlap_seconds", fallback=float(defaults["streaming_overlap_seconds"])),
        "streaming_match_words_threshold_seconds": config.getfloat("streaming", "streaming_match_words_threshold_seconds", fallback=float(defaults["streaming_match_words_threshold_seconds"])),
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
        # Clear any modifier keys that might be stuck from hotkey press
        # subprocess.run(
        #     ["xdotool", "keyup", "ctrl", "alt", "shift", "super"],
        #     stdout=subprocess.DEVNULL,
        #     stderr=subprocess.DEVNULL,
        #     timeout=2
        # )
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

        # Load model in background
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

    def on_press(self, key):
        if key == self.hotkey:
            self.start_recording()

    def on_release(self, key):
        if key == self.hotkey:
            self.stop_recording()

    def stop(self):
        logger.info("\nExiting...")
        self.running = False
        os._exit(0)

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
        self.temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        self.temp_file.close()

        # Record using arecord (ALSA) - works on most Linux systems
        self.record_process = self.start_record_process(self.temp_file.name)
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

        # Wait for model loading to finish.
        self.model_loaded.wait()
        if self.model_error:
            logger.error("Cannot transcribe: model failed to load")
            self.notify("Error", "Model failed to load", "dialog-error", 3000)
            return
        if self.model is None:
            logger.error("Cannot transcribe: model not loaded")
            return

        # Transcribe.
        temp_file_name = self.temp_file.name if self.temp_file else None
        try:
            if not temp_file_name or not os.path.exists(temp_file_name):
                logger.error(f"Cannot transcribe: audio file not found: {temp_file_name}")
                return

            trans_start = time.time()
            segments, info = self.model.transcribe(
                temp_file_name,
                beam_size=5,
                vad_filter=True,
            )
            trans_duration = time.time() - trans_start

            # Handle transcription.
            text = self._segments_to_text(segments, self.config["auto_sentence"])
            if text:
                logger.info(f"Transcribed {info.duration:.2f}s in {trans_duration:.2f}s: {text}")
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
                        f"Transcribed {info.duration:.2f}s speech:",
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
            # Cleanup temp file
            if temp_file_name and os.path.exists(temp_file_name):
                os.unlink(temp_file_name)


class StreamingDictation(Dictation):
    """Streaming dictation mode with incremental transcription.

    Inside there is thread pool of 4 threads:
    - 2 for recording
    - 1 for transcribing
    - 1 for typing

    Thread of transcribing should take tasks from queue in order of chunks.
    Thread of typing should work in the same order (order of chunks).

    Strategy of recording and transcription:
    - Create temp file for chunk 1, start recording for streaming_chunk_seconds in a thread pool
    - Wait streaming_overlap_seconds, then start chunk 2 (while chunk 1 is still recording) in a thread pool. Repeat this for next chunks

    Strategy of transcription:
    - After getting transcription for chunk remove temporal file and check it took less than streaming_overlap_seconds seconds
    - If transcription took streaming_overlap_seconds seconds or more than produce log and desktop notification about too slow transcription but continue working (transcription would be lagging)
    - If it is first chunk then just print transcription as is
    - If not first chunk then process new words with overlaps handling

    Overlaps handling strategy:
    - Chunks will overlap by streaming_overlap_seconds
    - In overlapped interval need to match words between previous chunk and current chunk by timestamps with streaming_match_words_threshold_seconds threshold
    - Words from the current chunk have priority over words from the previous chunk except the first word if it is timestamped withing streaming_match_words_threshold_seconds of the chunk start (first word in chunk could be just a part of the word so could be transcribed wrongly)
    - On any difference need to remove old text including first wrong word and type new text starting from this difference, inlcuding transcription of not overlapped part of chunk
    - If no difference found just print words from not overlapped interval of current chunk
    """

    def __init__(self, config: dict):
        # Initialize base class (sets up config, hotkey, model loading, etc.)
        super().__init__(config)
        assert self.config["auto_type"], "auto_type must be True for streaming mode"
        self.streaming_chunk_seconds = config["streaming_chunk_seconds"]
        self.streaming_overlap_seconds = config["streaming_overlap_seconds"]
        self.streaming_match_words_threshold_seconds = config["streaming_match_words_threshold_seconds"]

        # Thread pool: 2 for recording, 1 for transcribing, 1 for typing
        self.threads_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="streaming")

        # Queues for ordered processing
        self.transcription_queue = queue.Queue()  # FIFO queue for transcription tasks
        self.typing_queue = queue.Queue()  # FIFO queue for typing tasks

        # Streaming state
        self.active_recordings: dict[int, tuple[str, subprocess.Popen, float]] = {}  # chunk_index -> (file_path, process, start_time)
        self.record_thread: Optional[threading.Thread] = None
        self.accumulated_segments: list[Segment] = []

        # Create typer once in constructor
        self.typer: Optional[Typer] = None
        if self.config["auto_type"]:
            self.typer = Typer(
                delay_ms=int(self.config["typing_delay"] * 1000),
                start_delay_ms=100
            )

    def _finish_model_loading(self):
        logger.info(f"Press [{self.get_hotkey_name().upper()}] to start transcribing, press one more time to stop. Press Ctrl+C to quit.")

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
        self.model_loaded.wait()
        if self.model_error or self.model is None:
            logger.error("[record] Cannot start recording yet")
            self.notify("Error", "Model is not loaded yet", "dialog-error", 3000)
            return

        # Reset transcription state.
        self.recording = True
        self.active_recordings = {}
        self.accumulated_segments = []

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

        # Recreate thread pool (it may have been shut down from previous recording)
        try:
            self.threads_pool.shutdown(wait=False)
        except RuntimeError:
            # Already shut down, ignore
            pass
        self.threads_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="streaming")

        # Start transcription worker (1 thread in pool)
        self.threads_pool.submit(self._transcription_worker)
        # Start typing worker (1 thread in pool)
        if self.config["auto_type"]:
            self.typer = Typer(
                delay_ms=int(self.config["typing_delay"] * 1000),
                start_delay_ms=100
            )
            self.threads_pool.submit(self._typing_worker)
        # Start recording coordinator thread
        self.record_thread = threading.Thread(target=self._record_chunks_worker, daemon=True)
        self.record_thread.start()

        logger.info("[record] Recording (streaming mode)...")
        self.notify("Recording...", f"Press {self.get_hotkey_name().upper()} when done", "audio-input-microphone", 1500)

    def _record_single_chunk(self, chunk_idx: int, chunk_start_time: float) -> tuple[str, float, int]:
        """Record a single chunk in thread pool. Returns (file_path, actual_start_time)."""
        chunk_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        chunk_file.close()

        # Start recording this chunk (with duration limit)
        chunk_process = self.start_record_process(chunk_file.name, duration=self.streaming_chunk_seconds)
        self.active_recordings[chunk_idx] = (chunk_file.name, chunk_process, chunk_start_time)
        logger.info(f"[record] Chunk {chunk_idx} started recording to {chunk_file.name}")
        # Wait for recording to finish
        chunk_process.wait()
        logger.info(f"[record] Chunk {chunk_idx} recording finished")
        # Remove from active recordings
        if chunk_idx in self.active_recordings:
            del self.active_recordings[chunk_idx]
        return (chunk_file.name, chunk_start_time, chunk_idx)

    def _record_chunks_worker(self):
        """Thread worker for recording chunks."""
        try:
            chunk_index = 0
            recording_start_time = time.time()
            while self.recording:
                current_time = time.time()

                # Start new chunk
                chunk_index += 1
                chunk_start_time = current_time if recording_start_time is None else recording_start_time + (chunk_index - 1) * self.streaming_overlap_seconds
                # Submit recording task to thread pool (2 recording threads)
                future = self.threads_pool.submit(self._record_single_chunk, chunk_index, chunk_start_time)

                # When recording finishes, queue for transcription
                def on_recording_done(fut):
                    try:
                        file_path, start_time, chunk_idx = fut.result()
                        self.transcription_queue.put((file_path, start_time, chunk_idx))
                    except Exception as e:
                        logger.error(f"[record] Error in recording chunk {chunk_idx}: {e}", exc_info=True)

                future.add_done_callback(on_recording_done)

                # Calculate exact time when next chunk should start
                next_chunk_start_time = recording_start_time + chunk_index * self.streaming_overlap_seconds
                # Measure time after task submission to account for execution time
                time_after_submission = time.time()
                sleep_duration = next_chunk_start_time - time_after_submission
                # Sleep until exact next chunk start time (or continue immediately if already past)
                if sleep_duration > 0:
                    time.sleep(sleep_duration)
        except Exception as e:
            logger.error(f"[record] Error in chunk coordinator: {e}")

    def _transcription_worker(self):
        """Transcription worker thread - processes chunks in order."""
        while self.recording or not self.transcription_queue.empty():
            try:
                chunk_data = self.transcription_queue.get(timeout=0.1)
                if chunk_data is None:
                    break
                chunk_file_path, chunk_absolute_start_time, chunk_idx = chunk_data
                # Transcribe chunk
                trans_start = time.time()
                new_segments = self._transcribe_chunk(chunk_file_path, chunk_absolute_start_time)
                trans_duration = time.time() - trans_start
                # Remove temp file
                try:
                    logger.info(f"[transcriber] Chunk {chunk_idx} file {chunk_file_path} deleted")
                    os.unlink(chunk_file_path)
                except Exception as e:
                    logger.warning(f"[transcriber] Chunk {chunk_idx} file {chunk_file_path} failed to delete: {e}")

                # Check if transcription took too long
                if trans_duration >= self.streaming_overlap_seconds:
                    logger.warning(f"[transcriber] Chunk {chunk_idx} transcription took {trans_duration:.2f}s (>= {self.streaming_overlap_seconds}s) - transcription is lagging")
                    if self.config["notifications"]:
                        self.notify("Transcription too slow", f"Chunk {chunk_idx} took {trans_duration:.2f}s. Transcription is lagging.", "dialog-warning", 3000)

                # Check if there are any segments transcribed.
                if not new_segments:
                    logger.info(f"[transcriber] Chunk {chunk_idx} no words transcribed")
                    continue

                # If there are new segments then process them.
                chars_to_remove = 0
                text_to_type = ""
                if chunk_idx == 1:
                    # First chunk - just print transcription as is
                    text_to_type = self._segments_to_text(new_segments, False)
                    self.accumulated_segments = new_segments
                else:
                    # Not first chunk - process with overlap handling
                    text_to_remove, text_to_type = self._process_words_with_overlap(new_segments, chunk_absolute_start_time, chunk_idx)
                if self.typing_queue and text_to_type:
                    self.typing_queue.put((text_to_remove, text_to_type, chunk_idx))
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
                text_to_remove, text_to_type, chunk_idx = typing_task
                chars_to_remove = len(text_to_remove)
                if chars_to_remove > 0:
                    logger.info(f"[typer] Chunk {chunk_idx} removing '{text_to_remove}' and typing: {text_to_type}")
                else:
                    logger.info(f"[typer] Chunk {chunk_idx} typing: {text_to_type}")
                # Run typing.
                try:
                    self.typer.type_rewrite(text_to_type, chars_to_remove)
                except Exception as e:
                    logger.error(f"[typer] Typing failed: {e}", exc_info=True)
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"[typer] {e}", exc_info=True)

    def _transcribe_chunk(self, chunk_file_path: str, chunk_absolute_start_time: float) -> list[Segment]:
        """
        Transcribe a single audio chunk file using faster-whisper with word timestamps.
        Returns list of Segment objects with absolute timestamps.
        """
        try:
            if self.model is None:
                logger.error("[transcriber] Model not loaded")
                return []
            # Pass file path to faster-whisper for transcription
            segments, info = self.model.transcribe(
                chunk_file_path,
                beam_size=5,
                vad_filter=True,
                word_timestamps=True,
            )
            for segment in segments:
                # Convert relative timestamps to absolute by adding chunk start time
                segment.start += chunk_absolute_start_time
                segment.end += chunk_absolute_start_time
                # Adjust word timestamps to be absolute as well
                if segment.words:
                    for word in segment.words:
                        word.start += chunk_absolute_start_time
                        word.end += chunk_absolute_start_time
            return list(segments)
        except Exception as e:
            logger.error(f"[transcriber] Failed to transcribe chunk: {e}", exc_info=True)
            return []

    def _process_words_with_overlap(self, new_segments: list[Segment], chunk_absolute_start_time: float, chunk_idx: int) -> tuple[str, str]:
        """
        Process new words with overlap handling according to strategy:

        - Calculate overlap region. Words after it would be printed as is.
        - Match words between previous chunk and current chunk by timestamps with threshold.
        - If first word of current chunk is timestamped within threshold of chunk start then it should be skipped from matching.
        - Other words from current chunk have priority over words from previous chunk.
        - On the first difference found: update accumulated text with removing old words from the difference and typing rest of the new words. Send similar task to typing worker.
        - If no difference found: just print words from not overlapped interval of current chunk.

        Args:
            new_segments: List of Segment objects for new chunk.
            chunk_absolute_start_time: Absolute start time of current chunk.
            chunk_idx: Index of current chunk.
        Returns:
            tuple of (text to remove, text to type)
        """
        if not new_segments:
            return "", ""

        # If no accumulated segments, just add all new segments and return
        if not self.accumulated_segments:
            text_to_type = self._segments_to_text(new_segments, False)
            self.accumulated_segments.extend(new_segments)
            return "", text_to_type

        # Calculate overlap end to find non-overlapped segments.
        chunk_start = chunk_absolute_start_time
        threshold = self.streaming_match_words_threshold_seconds

        # Find first overlapping segment and last overlapping word end time
        first_overlapping_seg_idx = len(self.accumulated_segments) - 1
        last_overlapping_word_end = None
        for seg_idx, segment in enumerate(self.accumulated_segments):
            if segment.start > chunk_start:
                first_overlapping_seg_idx = seg_idx
                break
        last_overlapping_word_end = self.accumulated_segments[-1].end

        # Log overlap to process.
        overlapping_words_text = self._segments_to_text(self.accumulated_segments[first_overlapping_seg_idx:], False)
        new_words_text = self._segments_to_text(new_segments, False)
        logger.info(f"[transcriber] Chunk {chunk_idx} over overlapped '{overlapping_words_text}' processing words: {new_words_text}")

        # Check if first segment is near chunk start
        first_segment_start = new_segments[0].start
        first_segment_near_start = (first_segment_start - chunk_start) <= threshold

        # New words always have high priority.
        priority_segments = new_segments
        # Cut out first segment if it's near start (it has lower priority).
        if first_segment_near_start:
            priority_segments = new_segments[1:]

        # Loop over higher_priority_segments forward, search backwards in accumulated_segments
        last_matched_accumulated_idx = len(self.accumulated_segments) - 1
        first_mismatch_idx = None  # Index in priority_segments.
        for i, segment in enumerate(priority_segments):
            # Check if current segment is after last overlapping segment.
            if segment.end > last_overlapping_word_end:
                break
            # Reset last matched accumulated index to the end of accumulated segments.
            last_matched_accumulated_idx = len(self.accumulated_segments) - 1
            # Search backwards in accumulated_segments.
            # I.e. from end to first_overlapping_seg_idx.
            inner_idx = len(self.accumulated_segments) - 1
            for j in range(len(self.accumulated_segments) - 1, last_matched_accumulated_idx, -1):
                accumulated_segment = self.accumulated_segments[j]
                # Check if timestamps match within threshold.
                if abs(segment.start - accumulated_segment.start) <= threshold or abs(segment.end - accumulated_segment.end) <= threshold:
                    # Timestamps match - check if text is different (mismatch).
                    if segment.text != accumulated_segment.text:
                        last_matched_accumulated_idx = inner_idx
                        first_mismatch_idx = i
                        break
                    # Timestamps match and text is same - continue matching.
                    inner_idx = j
            # Check first mismatch is found.
            if first_mismatch_idx is None:
                first_mismatch_idx = i
                break

        # Check if mismatch found.
        if first_mismatch_idx is not None:
            # Calculate segments to remove.
            segments_to_remove = self.accumulated_segments[:last_matched_accumulated_idx]
            segments_to_add = priority_segments[first_mismatch_idx:]

            # Replace tail of accumulated_segments with higher priority segments
            self.accumulated_segments = self.accumulated_segments[:last_matched_accumulated_idx + 1] + segments_to_add

            # Calculate chars to remove and text to type
            old_text_to_remove = self._segments_to_text(segments_to_remove, False)
            new_text_to_type = self._segments_to_text(segments_to_add, False)
            logger.info(f"[transcriber] Chunk {chunk_idx} mismatch found, removing '{old_text_to_remove}' and typing: {new_text_to_type}")
            return old_text_to_remove, new_text_to_type
        else:
            # No mismatches found - append remaining unmatched segments from priority_segments.
            segments_to_add = priority_segments[last_matched_accumulated_idx:]
            new_text_to_type = self._segments_to_text(segments_to_add, False)
            logger.info(f"[transcriber] Chunk {chunk_idx} typing: {new_text_to_type}")
            return "", new_text_to_type

    def _finalize_transcription(self) -> str:
        """Finalize transcription and return complete text."""
        # Wait for all queues to empty (with timeout to prevent infinite loops)
        max_wait_time = self.streaming_chunk_seconds * 2
        wait_time = 0.0
        while (not self.transcription_queue.empty() or not self.typing_queue.empty()) and wait_time < max_wait_time:
            time.sleep(0.1)
            wait_time += 0.1
        return self._segments_to_text(self.accumulated_segments, False)

    def stop_recording(self):
        if not self.recording:
            return
        self.recording = False
        # Wait for all active recordings to finish and queue them
        for chunk_idx, (file_path, process, start_time) in list(self.active_recordings.items()):
            if process.poll() is None:
                process.wait()
            if os.path.exists(file_path):
                self.transcription_queue.put((file_path, start_time, chunk_idx))
        self.active_recordings.clear()
        # Signal workers to stop
        self.transcription_queue.put(None)
        if self.config["auto_type"]:
            self.typing_queue.put(None)
        # Wait for threads
        if self.record_thread:
            self.record_thread.join(timeout=2.0)
        # Shutdown thread pool (if not already shut down)
        try:
            self.threads_pool.shutdown(wait=True)
        except RuntimeError:
            # Already shut down, ignore
            pass

        logger.info("[idle] Recording stopped, transcribing remaining chunks...")
        self.notify("Transcribing stopped", "Processing remaining chunks", "emblem-synchronizing", 1500)
        # Finalize any remaining chunks
        final_text = self._finalize_transcription()
        if final_text:
            if self.config["clipboard"]:
                process = subprocess.Popen(
                    ["xclip", "-selection", "clipboard"],
                    stdin=subprocess.PIPE
                )
                process.communicate(input=final_text.encode())
                logger.info(f"[idle] Pasted to clipboard: {final_text}")
            self.notify("Got:", final_text[:100] + ("..." if len(final_text) > 100 else ""), "emblem-ok-symbolic", 3000)
        else:
            logger.info("[idle] No speech detected")
            self.notify("No speech detected", "Try speaking louder", "dialog-warning", 2000)

    def stop(self):
        """Override to stop recording before exiting."""
        logger.info("\nExiting...")
        self.running = False
        if self.recording:
            self.stop_recording()
        os._exit(0)


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
    args = parser.parse_args()
    check_dependencies(config)

    # Apply arguments.
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    use_streaming = config["default_streaming"]
    if args.streaming:
        use_streaming = True
    elif args.no_streaming:
        use_streaming = False
    if use_streaming:
        dictation = StreamingDictation(config)
    else:
        dictation = Dictation(config)

    # Handle Ctrl+C gracefully
    def handle_sigint(sig, frame):
        dictation.stop()

    signal.signal(signal.SIGINT, handle_sigint)

    # Start dictation.
    dictation.run()


if __name__ == "__main__":
    main()
