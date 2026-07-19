"""Text-to-speech adapter interfaces."""

from dublaro.adapters.tts.base import SpeechSynthesisOptions, TtsAdapter
from dublaro.adapters.tts.fake import FakeTtsAdapter

__all__ = [
    "FakeTtsAdapter",
    "SpeechSynthesisOptions",
    "TtsAdapter",
]
