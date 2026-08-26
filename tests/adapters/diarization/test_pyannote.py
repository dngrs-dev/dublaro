from dataclasses import dataclass
from pathlib import Path

import dublaro.adapters.diarization.pyannote as pyannote_module
import pytest
from dublaro.adapters.diarization import DiarizationOptions
from dublaro.adapters.diarization.pyannote import PyannoteDiarizationAdapter


@dataclass(frozen=True)
class FakeTurn:
    start: float
    end: float


@dataclass(frozen=True)
class FakeWaveform:
    shape: tuple[int, int]


class FakeOutput:
    def __init__(self) -> None:
        self.exclusive_speaker_diarization = [
            (FakeTurn(0.0, 1.5), "speaker-1"),
            (FakeTurn(1.5, 3.0), "speaker-2"),
        ]


class FakePipeline:
    def __init__(self) -> None:
        self.calls: list[tuple[dict[str, object], dict[str, int]]] = []

    def __call__(self, audio: dict[str, object], **kwargs: int) -> FakeOutput:
        self.calls.append((audio, kwargs))
        return FakeOutput()


def test_pyannote_diarization_adapter_converts_turns(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"fake wav")

    fake_waveform = {
        "waveform": FakeWaveform(shape=(1, 4)),
        "sample_rate": 16_000,
    }

    def fake_load_waveform(path: Path) -> dict[str, object]:
        assert path == audio_path
        return fake_waveform

    monkeypatch.setattr(pyannote_module, "_load_waveform", fake_load_waveform)

    pipeline = FakePipeline()
    adapter = PyannoteDiarizationAdapter(pipeline=pipeline)

    turns = adapter.diarize(
        audio_path,
        DiarizationOptions(min_speakers=1, max_speakers=2),
    )

    audio, kwargs = pipeline.calls[0]

    assert kwargs == {"min_speakers": 1, "max_speakers": 2}
    assert audio is fake_waveform
    assert audio["sample_rate"] == 16_000

    waveform = audio["waveform"]
    assert isinstance(waveform, FakeWaveform)
    assert waveform.shape == (1, 4)

    assert turns[0].speaker == "speaker-1"
    assert turns[0].start == 0.0
    assert turns[0].end == 1.5
    assert turns[1].speaker == "speaker-2"
