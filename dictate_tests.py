#!/usr/bin/env python3
"""
Tests for SoupaWhisper dictate.py
"""

import pytest
import tempfile
import threading
import time
import wave
import numpy as np
import shutil
import queue
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch, mock_open
from io import BytesIO
from typing import Any

# Add 2 second timeout to all tests to prevent infinite loops
pytestmark = pytest.mark.timeout(2)

# Import the modules to test
import sys
import os

# Add the directory containing dictate.py to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Mock pynput and streaming audio deps before importing dictate
sys.modules['pynput'] = MagicMock()
sys.modules['pynput.keyboard'] = MagicMock()
sys.modules['pyaudio'] = MagicMock()
webrtcvad_mock = MagicMock()
vad_instance = MagicMock()
vad_instance.is_speech = MagicMock(return_value=False)
webrtcvad_mock.Vad = MagicMock(return_value=vad_instance)
sys.modules['webrtcvad'] = webrtcvad_mock
streamsad_mock = MagicMock()
streamsad_mock.SAD = MagicMock()
sys.modules['streamsad'] = streamsad_mock

# Now import dictate
import dictate

# Import real Segment and Word from faster_whisper
from faster_whisper.transcribe import Segment, Word


class MockWhisperModel:
    """Mock WhisperModel for testing."""
    def __init__(self, model_name="base.en", device="cpu", compute_type="int8"):
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type
        self.transcribe_calls = []
        self.transcribe = MagicMock(side_effect=self._transcribe_impl)
    
    def _transcribe_impl(self, audio_path, **kwargs):
        """Mock transcribe that records calls and returns test data."""
        self.transcribe_calls.append((audio_path, kwargs))
        
        # Return real segments based on audio path or kwargs
        if "word_timestamps" in kwargs and kwargs["word_timestamps"]:
            # Return segments with word timestamps for streaming tests
            words = [
                Word(word="hello", start=0.0, end=0.5, probability=0.9),
                Word(word="world", start=0.6, end=1.0, probability=0.9),
            ]
            segment = Segment(
                id=0, seek=0, start=0.0, end=1.0, text="hello world",
                tokens=[], avg_logprob=0.0, compression_ratio=0.0,
                no_speech_prob=0.0, words=words, temperature=None
            )
            return [segment], {"language": "en"}
        else:
            # Return simple segments for non-streaming tests
            segment = Segment(
                id=0, seek=0, start=0.0, end=1.0, text="test transcription",
                tokens=[], avg_logprob=0.0, compression_ratio=0.0,
                no_speech_prob=0.0, words=None, temperature=None
            )
            return [segment], {"language": "en"}


@pytest.fixture
def mock_config(tmp_path, monkeypatch):
    """Create a temporary config file and mock CONFIG_PATH."""
    config_dir = tmp_path / ".config" / "soupawhisper"
    config_dir.mkdir(parents=True)
    config_file = config_dir / "config.ini"
    
    # Create a test config
    config_content = """[whisper]
model = base.en
device = cpu
compute_type = int8

[hotkey]
key = f10

[behavior]
auto_type = true
notifications = false
default_streaming = false
clipboard = true

[streaming]
vad_silence_threshold_seconds = 1.0
vad_sample_rate = 16000
vad_chunk_size_ms = 30
vad_threshold = 0.5
"""
    config_file.write_text(config_content)
    
    # Mock the CONFIG_PATH
    monkeypatch.setattr(dictate, "CONFIG_PATH", config_file)
    return config_file


@pytest.fixture
def mock_whisper_model(monkeypatch):
    """Mock WhisperModel."""
    model = MockWhisperModel()
    
    def mock_init(model_name, device="cpu", compute_type="int8"):
        model.model_name = model_name
        model.device = device
        model.compute_type = compute_type
        return model
    
    monkeypatch.setattr(dictate.WhisperModel, "__new__", lambda cls, *args, **kwargs: mock_init(*args, **kwargs))
    
    return model


@pytest.fixture
def mock_arecord(monkeypatch):
    """Mock arecord subprocess."""
    mock_process = MagicMock()
    mock_process.poll.return_value = None
    mock_process.stdout = BytesIO()
    mock_process.wait.return_value = 0
    mock_process.terminate = MagicMock()
    
    def mock_popen(cmd, **kwargs):
        if "arecord" in cmd:
            return mock_process
        return MagicMock()
    
    monkeypatch.setattr(dictate.subprocess, "Popen", mock_popen)
    monkeypatch.setattr(dictate.subprocess, "run", MagicMock(return_value=MagicMock(returncode=0)))
    
    return mock_process


@pytest.fixture
def mock_xdotool(monkeypatch):
    """Mock xdotool."""
    mock_run = MagicMock()
    mock_run.return_value = MagicMock(returncode=0)
    
    # Mock subprocess.run for "which" command and other calls
    def mock_run_with_which(cmd, **kwargs):
        if isinstance(cmd, list) and len(cmd) > 0 and cmd[0] == "which":
            result = MagicMock()
            if len(cmd) > 1 and cmd[1] in ["xdotool", "arecord", "xclip"]:
                result.returncode = 0
            else:
                result.returncode = 1
            return result
        # For other commands (xdotool type, etc.), use the mock
        return mock_run(cmd, **kwargs)
    
    monkeypatch.setattr(dictate.subprocess, "run", mock_run_with_which)
    
    # Store the mock so tests can access it
    mock_run_with_which._mock_run = mock_run
    return mock_run_with_which


class TestTyper:
    """Tests for Typer class."""
    
    def test_typer_init(self, mock_xdotool):
        """Test Typer initialization."""
        typer = dictate.Typer(delay_ms=20, start_delay_ms=100)
        assert typer.delay_ms == 20
        assert typer.start_delay_ms == 100
        assert typer.enabled is True
    
    def test_typer_type_rewrite_append(self, mock_xdotool):
        """Test typing text (append mode with previous_length=0)."""
        typer = dictate.Typer()
        typer.type_rewrite("hello world", 0)
        assert mock_xdotool._mock_run.called
    
    def test_typer_type_rewrite_incremental(self, mock_xdotool):
        """Test incremental typing using type_rewrite with previous_length=0."""
        typer = dictate.Typer()
        # Simulate incremental: calculate suffix and type with previous_length=0
        previous_text = "hello"
        new_text = "hello world"
        suffix = new_text[len(previous_text):]
        typer.type_rewrite(suffix, 0)
        assert mock_xdotool._mock_run.called
    
    def test_typer_type_rewrite_correction(self, mock_xdotool):
        """Test rewrite typing with character removal."""
        typer = dictate.Typer()
        typer.type_rewrite("new text", 5)
        assert mock_xdotool._mock_run.called


class TestStreamingDictation:
    """Tests for StreamingDictation class."""

    def test_streaming_initializes(self, mock_config, mock_whisper_model, mock_xdotool):
        """Basic sanity check that StreamingDictation can be created."""
        config = dictate.load_config()
        config["default_streaming"] = True
        config["auto_type"] = True  # Required for streaming mode
        dictation = dictate.StreamingDictation(config)
        dictation.model_loaded.wait(timeout=1.0)
        assert isinstance(dictation, dictate.StreamingDictation)


class TestDictation:
    """Tests for non-streaming Dictation class (backward compatibility)."""
    
    @patch('dictate.subprocess.run')
    @patch('dictate.subprocess.Popen')
    def test_non_streaming_mode(self, mock_popen, mock_run, mock_config, mock_whisper_model):
        """Test non-streaming mode (backward compatibility)."""
        # Ensure non-streaming mode
        config = dictate.load_config()
        config["default_streaming"] = False
        
        dictation = dictate.Dictation(config)
        dictation.model_loaded.wait(timeout=1.0)
        
        # Start recording
        dictation.start_recording()
        assert dictation.recording is True
        
        # Stop recording
        dictation.stop_recording()
        assert dictation.recording is False
    
    def test_config_loading(self, mock_config):
        """Test configuration loading."""
        # Reload config to test loading
        config = dictate.load_config()
        assert "model" in config
        assert "default_streaming" in config
        assert "clipboard" in config
        assert config["model"] == "base.en"
        assert isinstance(config["default_streaming"], bool)
        assert config["clipboard"] is True

    def test_config_clipboard_disabled(self, mock_config):
        """Test that clipboard=false is correctly loaded."""
        # Replace existing clipboard = true with clipboard = false
        content = mock_config.read_text()
        new_content = content.replace("clipboard = true", "clipboard = false")
        mock_config.write_text(new_content)
        
        config = dictate.load_config()
        assert config["clipboard"] is False

class TestClipboardIntegration:
    """Tests specifically for the clipboard parameter integration."""

    @patch('dictate.subprocess.run')
    @patch('dictate.subprocess.Popen')
    def test_dictation_no_clipboard_call(self, mock_popen, mock_run, mock_config, mock_whisper_model):
        """Test that Dictation doesn't call xclip when clipboard is disabled."""
        # Mock subprocess.run for Typer initialization
        def mock_run_side_effect(cmd, **kwargs):
            result = MagicMock()
            if isinstance(cmd, list) and len(cmd) > 0 and cmd[0] == "which":
                if len(cmd) > 1 and cmd[1] in ["xdotool", "arecord", "xclip"]:
                    result.returncode = 0
                else:
                    result.returncode = 1
            else:
                result.returncode = 0
            return result
        mock_run.side_effect = mock_run_side_effect
        
        content = mock_config.read_text()
        new_content = content.replace("clipboard = true", "clipboard = false")
        mock_config.write_text(new_content)
        config = dictate.load_config()
        
        dictation = dictate.Dictation(config)
        dictation.model_loaded.wait(timeout=1.0)
        
        # Mock a recorded file
        dictation.temp_file = MagicMock()
        dictation.temp_file.name = "test.wav"
        dictation.recording = True
        
        # Mock model return
        segment = Segment(
            id=0, seek=0, start=0.0, end=1.0, text="test text",
            tokens=[], avg_logprob=0.0, compression_ratio=0.0,
            no_speech_prob=0.0, words=None, temperature=None
        )
        model: Any = dictation.model
        model.transcribe.side_effect = None
        model.transcribe.return_value = ([segment], {})
        
        dictation.stop_recording()
        
        # Check that xclip was NOT called
        for call in mock_popen.call_args_list:
            args = call[0][0]
            assert "xclip" not in args

    @patch('dictate.subprocess.run')
    @patch('dictate.subprocess.Popen')
    @patch('dictate.pyaudio.PyAudio')
    def test_streaming_dictation_no_clipboard_call(self, mock_pyaudio, mock_popen, mock_run, mock_config, mock_whisper_model, mock_arecord):
        """Test that StreamingDictation doesn't call xclip when clipboard is disabled."""
        # Mock subprocess.run for Typer initialization
        def mock_run_side_effect(cmd, **kwargs):
            result = MagicMock()
            if isinstance(cmd, list) and len(cmd) > 0 and cmd[0] == "which":
                if len(cmd) > 1 and cmd[1] in ["xdotool", "arecord", "xclip"]:
                    result.returncode = 0
                else:
                    result.returncode = 1
            else:
                result.returncode = 0
            return result
        mock_run.side_effect = mock_run_side_effect
        
        # Mock PyAudio
        mock_audio_instance = MagicMock()
        mock_audio_stream = MagicMock()
        mock_audio_instance.open.return_value = mock_audio_stream
        mock_audio_instance.get_device_count.return_value = 1
        mock_audio_instance.get_default_input_device_info.return_value = {'index': 0, 'name': 'test device'}
        mock_pyaudio.return_value = mock_audio_instance
        
        content = mock_config.read_text()
        new_content = content.replace("clipboard = true", "clipboard = false")
        mock_config.write_text(new_content)
        config = dictate.load_config()
        config["auto_type"] = True  # Required for streaming mode
        
        dictation = dictate.StreamingDictation(config)
        dictation.model_loaded.wait(timeout=1.0)
        
        # Clear queues to prevent timeout
        while not dictation.transcription_queue.empty():
            try:
                dictation.transcription_queue.get_nowait()
            except queue.Empty:
                break
        while not dictation.typing_queue.empty():
            try:
                dictation.typing_queue.get_nowait()
            except queue.Empty:
                break
        
        # Set up state for stop_recording
        dictation.recording = True
        dictation.accumulated_text = "final text"
        dictation.audio_stream = mock_audio_stream
        dictation.audio_interface = mock_audio_instance
        dictation.audio_thread = None  # No thread to join
        dictation.transcription_thread = None
        dictation.typing_thread = None
        dictation.file_saving_thread = None
        
        dictation.stop_recording()
        
        # Check that xclip was NOT called
        for call in mock_popen.call_args_list:
            args = call[0][0]
            assert "xclip" not in args

    def test_check_dependencies_clipboard_optional(self, monkeypatch):
        """Test that xclip is optional in check_dependencies if clipboard is disabled."""
        mock_run = MagicMock()
        
        # Mock 'which' to return 0 for arecord, but 1 for xclip
        def side_effect(cmd, **kwargs):
            res = MagicMock()
            if isinstance(cmd, list) and len(cmd) > 1 and cmd[1] == "xclip":
                res.returncode = 1
            else:
                res.returncode = 0
            return res
        
        mock_run.side_effect = side_effect
        monkeypatch.setattr(dictate.subprocess, "run", mock_run)
        
        # Should NOT exit if clipboard is False (webrtcvad is already imported, so import check passes)
        dictate.check_dependencies({"clipboard": False, "auto_type": False, "default_streaming": False})
        
        # Should exit if clipboard is True (since xclip is missing)
        with pytest.raises(SystemExit):
            dictate.check_dependencies({"clipboard": True, "auto_type": False, "default_streaming": False})




if __name__ == "__main__":
    pytest.main([__file__, "-v"])
