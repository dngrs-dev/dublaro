import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from dublaro.adapters.diarization.base import DiarizationOptions, DiarizationTurn
from dublaro.audio.wav import read_mono_pcm16_wav

DEFAULT_PYANNOTE_MODEL_ID = "pyannote/speaker-diarization-community-1"


class PyannoteDiarizationAdapter:
    name = "pyannote"

    def __init__(
        self,
        *,
        model_id: str = DEFAULT_PYANNOTE_MODEL_ID,
        device: str | None = None,
        token_env_var: str | None = None,
        pipeline: Any | None = None,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.token_env_var = token_env_var
        self._pipeline = pipeline or self._load_pipeline()

    def diarize(
        self,
        audio_path: Path,
        options: DiarizationOptions,
    ) -> list[DiarizationTurn]:
        kwargs: dict[str, int] = {}
        if options.min_speakers is not None:
            kwargs["min_speakers"] = options.min_speakers
        if options.max_speakers is not None:
            kwargs["max_speakers"] = options.max_speakers

        output = self._pipeline(_load_waveform(audio_path), **kwargs)
        diarization = getattr(output, "exclusive_speaker_diarization", None) or getattr(
            output, "speaker_diarization", output
        )

        turns = [
            DiarizationTurn(start=start, end=end, speaker=speaker)
            for start, end, speaker in _iter_turns(diarization)
        ]
        return sorted(turns, key=lambda turn: (turn.start, turn.end, turn.speaker))

    def _load_pipeline(self) -> Any:
        try:
            from pyannote.audio import Pipeline
        except ImportError as error:
            raise RuntimeError(
                'pyannote.audio is not installed. Install it with: pip install -e ".[diarization]"'
            ) from error

        token = _read_token(self.token_env_var)
        pipeline = Pipeline.from_pretrained(self.model_id, token=token)

        if pipeline is None:
            raise RuntimeError(
                "Could not load pyannote diarization pipeline. Accept the model terms "
                "on Hugging Face and set HF_TOKEN, or use a local model path."
            )

        if self.device is not None:
            try:
                import torch
            except ImportError as error:
                raise RuntimeError(
                    "torch is required to set pyannote device."
                ) from error

            pipeline.to(torch.device(self.device))

        return pipeline


def _read_token(token_env_var: str | None) -> str | None:
    if token_env_var is not None:
        return os.getenv(token_env_var)

    return os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")


def _iter_turns(diarization: Any) -> Iterable[tuple[float, float, str]]:
    if hasattr(diarization, "itertracks"):
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            yield float(turn.start), float(turn.end), str(speaker)
        return

    for item in diarization:
        if len(item) == 2:
            turn, speaker = item
        elif len(item) == 3:
            turn, _, speaker = item
        else:
            raise RuntimeError(f"Unexpected pyannote diarization item: {item!r}")

        yield float(turn.start), float(turn.end), str(speaker)


def _load_waveform(audio_path: Path) -> dict[str, Any]:
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("torch is required for pyannote diarization.") from error

    sample_rate, samples = read_mono_pcm16_wav(audio_path)

    waveform = torch.tensor(
        samples.tolist(),
        dtype=torch.float32,
    ).unsqueeze(0)

    waveform = waveform / 32768.0

    return {
        "waveform": waveform,
        "sample_rate": sample_rate,
    }
