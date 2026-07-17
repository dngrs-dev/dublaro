from pathlib import Path
from typing import Any

from dublaro.adapters.asr.base import TranscriptionOptions
from dublaro.schemas import Segment, Transcript, WordTiming


class FastWhisperAsrAdapter:
    name = "faster-whisper"

    def __init__(
        self,
        model_size: str = "small",
        device: str = "cpu",
        compute_type: str = "int8",
    ) -> None:
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self._model: Any | None = None

    def _load_model(self) -> Any:
        if self._model is None:
            try:
                from faster_whisper import WhisperModel
            except ImportError as error:
                raise RuntimeError(
                    "faster-whisper is not installed. "
                    'Install it with: pip install -e ".[asr]"'
                ) from error

            self._model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
            )

        return self._model

    def transcribe(
        self,
        audio_path: Path,
        options: TranscriptionOptions,
    ) -> Transcript:
        model = self._load_model()

        transcribe_kwargs: dict[str, Any] = {
            "beam_size": options.beam_size,
            "word_timestamps": options.word_timestamps,
        }

        if options.source_language is not None:
            transcribe_kwargs["language"] = options.source_language

        raw_segments, info = model.transcribe(str(audio_path), **transcribe_kwargs)

        language = (
            getattr(info, "language", None) or options.source_language or "unknown"
        )
        duration = getattr(info, "duration", None)

        segments: list[Segment] = []

        for index, raw_segment in enumerate(raw_segments, start=1):
            words = [
                WordTiming(
                    text=word.word.strip(),
                    start=float(word.start),
                    end=float(word.end),
                    confidence=getattr(word, "probability", None),
                )
                for word in (raw_segment.words or [])
            ]

            segments.append(
                Segment(
                    id=f"seg-{index:04d}",
                    start=float(raw_segment.start),
                    end=float(raw_segment.end),
                    source_text=raw_segment.text.strip(),
                    source_language=language,
                    words=words,
                )
            )

        return Transcript(
            id=audio_path.stem,
            source_language=language,
            duration=duration,
            segments=segments,
            metadata={
                "adapter": self.name,
                "model_size": self.model_size,
                "device": self.device,
                "compute_type": self.compute_type,
            },
        )
