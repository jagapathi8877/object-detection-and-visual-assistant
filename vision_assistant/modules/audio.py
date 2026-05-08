"""
Audio Feedback Module -- pyttsx3 Offline TTS with gTTS Online Fallback.

Per master prompt:
  Audio: pyttsx3 (offline TTS), gTTS (online fallback)

Non-blocking audio: speak() returns immediately. The TTS worker runs
in a background thread using queue.PriorityQueue for urgency ordering.
"""

import os
import queue
import tempfile
import threading
import time
from typing import Dict, Optional

from utils.logger import get_logger

logger = get_logger(__name__)


class AudioFeedback:
    """Non-blocking TTS with pyttsx3 + gTTS fallback.

    speak() enqueues text and returns immediately. A background worker
    thread processes the queue using pyttsx3 (offline) with gTTS as
    online fallback.

    Attributes:
        _engine: pyttsx3 TTS engine instance.
        _rate: Speech rate (words per minute).
        _volume: Volume level [0.0, 1.0].
        _cooldown_seconds: Cooldown between same announcement.
        _clear_path_interval: Seconds of silence before "Path is clear".
        _audio_queue: Thread-safe PriorityQueue.
        _worker_thread: Background TTS worker thread.
        _running: Threading event for lifecycle control.
        _cooldown_tracker: Dict of (label_zone) -> last_announced_time.
    """

    def __init__(self, config: dict) -> None:
        """Initialise the audio feedback system.

        Args:
            config: Audio section from config.yaml.
        """
        self._rate: int = int(config.get("rate", 160))
        self._volume: float = float(config.get("volume", 0.9))
        self._cooldown_seconds: float = float(
            config.get("cooldown_seconds", 3.0)
        )
        self._clear_path_interval: float = float(
            config.get("clear_path_interval", 5.0)
        )
        self._use_gtts: bool = False
        self._cooldown_tracker: Dict[str, float] = {}

        # Try to initialise pyttsx3
        self._engine = None
        try:
            import pyttsx3
            self._engine = pyttsx3.init()
            self._engine.setProperty("rate", self._rate)
            self._engine.setProperty("volume", self._volume)
            logger.info(
                "pyttsx3 initialised: rate=%d, volume=%.1f",
                self._rate, self._volume,
            )
        except Exception as exc:
            logger.warning(
                "pyttsx3 init failed (%s) -- will try gTTS fallback.", exc
            )
            self._use_gtts = True

        # Thread-safe priority queue: items are (priority, timestamp, text)
        self._audio_queue: queue.PriorityQueue = queue.PriorityQueue()
        self._running = threading.Event()
        self._running.set()

        # Start background worker thread
        self._worker_thread = threading.Thread(
            target=self._audio_worker,
            name="AudioWorker",
            daemon=True,
        )
        self._worker_thread.start()
        logger.info("AudioFeedback worker thread started.")

    def _audio_worker(self) -> None:
        """Background worker that processes the audio queue.

        Blocks on queue.get(), speaks the text via pyttsx3 or gTTS,
        and logs each announcement.
        """
        while self._running.is_set():
            try:
                # Block with timeout so we can check _running periodically
                try:
                    priority, ts, text = self._audio_queue.get(timeout=0.5)
                except queue.Empty:
                    continue

                if text == "__STOP__":
                    break

                logger.info("Speaking: '%s' (priority=%d)", text, priority)
                self._speak_text(text)
                self._audio_queue.task_done()

            except Exception as exc:
                logger.error("Audio worker error: %s", exc)

        logger.info("Audio worker stopped.")

    def _speak_text(self, text: str) -> None:
        """Speak text using pyttsx3, falling back to gTTS if needed.

        Args:
            text: The text string to speak aloud.
        """
        if not self._use_gtts and self._engine is not None:
            try:
                self._engine.say(text)
                self._engine.runAndWait()
                return
            except Exception as exc:
                logger.warning("pyttsx3 speak failed (%s), trying gTTS.", exc)

        # gTTS fallback (requires internet)
        try:
            from gtts import gTTS
            tts = gTTS(text=text, lang="en")
            temp_path = os.path.join(
                tempfile.gettempdir(),
                f"vision_tts_{int(time.time() * 1000)}.mp3",
            )
            tts.save(temp_path)

            # Play with pygame or playsound
            try:
                import pygame
                if not pygame.mixer.get_init():
                    pygame.mixer.init()
                pygame.mixer.music.load(temp_path)
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    time.sleep(0.05)
            except Exception:
                # Last resort: try os-level playback
                import subprocess
                subprocess.run(
                    ["start", "", temp_path],
                    shell=True, check=False,
                )
                time.sleep(2)

            # Clean up
            try:
                os.remove(temp_path)
            except OSError:
                pass

        except Exception as exc:
            logger.error("gTTS fallback also failed: %s", exc)

    def speak(self, text: str, priority: int = 5) -> None:
        """Queue a text announcement for non-blocking playback.

        Returns immediately. The actual TTS runs in the background.

        Args:
            text: The announcement string to speak.
            priority: Lower = spoken sooner (1=urgent, 5=normal, 10=low).
        """
        if not self._running.is_set():
            return
        self._audio_queue.put((priority, time.time(), text))

    def flush(self) -> None:
        """Clear all pending items from the audio queue."""
        cleared = 0
        while not self._audio_queue.empty():
            try:
                self._audio_queue.get_nowait()
                self._audio_queue.task_done()
                cleared += 1
            except queue.Empty:
                break
        if cleared > 0:
            logger.debug("Flushed %d pending announcements.", cleared)

    def stop(self) -> None:
        """Stop the audio worker thread and clean up."""
        if not self._running.is_set():
            return

        logger.info("Stopping audio feedback...")
        self._running.clear()
        self._audio_queue.put((0, 0, "__STOP__"))

        if self._worker_thread.is_alive():
            self._worker_thread.join(timeout=2.0)

        if self._engine is not None:
            try:
                self._engine.stop()
            except Exception:
                pass

        logger.info("Audio feedback stopped.")

    def is_running(self) -> bool:
        """Check if the audio worker is active."""
        return self._running.is_set() and self._worker_thread.is_alive()

    @property
    def queue_size(self) -> int:
        """Number of pending items in the audio queue."""
        return self._audio_queue.qsize()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False
