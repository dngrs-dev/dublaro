"""Text-to-speech adapter interfaces."""

from dublaro.adapters.tts.base import SpeechSynthesisOptions, TtsAdapter
from dublaro.adapters.tts.fake import FakeTtsAdapter
from dublaro.adapters.tts.piper import PiperTtsAdapter

__all__ = [
    "FakeTtsAdapter",
    "PiperTtsAdapter",
    "SpeechSynthesisOptions",
    "TtsAdapter",
]
