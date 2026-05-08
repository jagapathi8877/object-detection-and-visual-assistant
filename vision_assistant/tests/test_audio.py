"""
Tests for AudioFeedback -- pyttsx3/gTTS non-blocking TTS.
"""

import pytest
import sys
import time
from unittest.mock import MagicMock, patch

from modules.audio import AudioFeedback


@pytest.fixture
def audio_config():
    return {
        "engine": "pyttsx3",
        "rate": 160,
        "volume": 0.9,
        "cooldown_seconds": 1.0,
        "clear_path_interval": 2.0,
    }


class TestAudioFeedback:
    """Test the AudioFeedback class."""

    def test_init_success(self, audio_config):
        """Test successful initialization with pyttsx3."""
        mock_pyttsx3 = MagicMock()
        mock_engine = MagicMock()
        mock_pyttsx3.init.return_value = mock_engine
        
        with patch.dict("sys.modules", {"pyttsx3": mock_pyttsx3}):
            audio = AudioFeedback(audio_config)
            assert audio._engine is mock_engine
            assert audio._use_gtts is False
            audio.stop()

    def test_fallback_to_gtts(self, audio_config):
        """Test fallback to gTTS when pyttsx3 fails."""
        mock_pyttsx3 = MagicMock()
        mock_pyttsx3.init.side_effect = Exception("Init failed")
        
        with patch.dict("sys.modules", {"pyttsx3": mock_pyttsx3}):
            audio = AudioFeedback(audio_config)
            assert audio._engine is None
            assert audio._use_gtts is True
            audio.stop()

    def test_speak_queues_item(self, audio_config):
        """Test speak() adds items to the queue."""
        mock_pyttsx3 = MagicMock()
        with patch.dict("sys.modules", {"pyttsx3": mock_pyttsx3}):
            audio = AudioFeedback(audio_config)
            audio.speak("Hello", priority=5)
            time.sleep(0.1)
            audio.stop()

    def test_flush_clears_queue(self, audio_config):
        """Test flush() clears pending items."""
        mock_pyttsx3 = MagicMock()
        with patch.dict("sys.modules", {"pyttsx3": mock_pyttsx3}):
            audio = AudioFeedback(audio_config)
            
            # Stop worker to safely fill queue without race conditions
            audio.stop()
            
            audio.speak("1", priority=5)
            audio.speak("2", priority=5)
            assert audio.queue_size == 2
            
            audio.flush()
            assert audio.queue_size == 0
            
            # Restore state so stop() cleans up cleanly
            audio._running.set()
            audio.stop()

    def test_stop_cleans_up(self, audio_config):
        """Test stop() terminates the worker thread."""
        mock_pyttsx3 = MagicMock()
        with patch.dict("sys.modules", {"pyttsx3": mock_pyttsx3}):
            audio = AudioFeedback(audio_config)
            assert audio.is_running() is True
            
            audio.stop()
            assert audio.is_running() is False
            
            # Calling stop again should be safe
            audio.stop()
