"""Speech-to-Text (ASR) adapter interfaces."""

from dublaro.adapters.asr.base import AsrAdapter, TranscriptionOptions
from dublaro.adapters.asr.fake import FakeAsrAdapter

__all__ = ["AsrAdapter", "FakeAsrAdapter", "TranscriptionOptions"]
