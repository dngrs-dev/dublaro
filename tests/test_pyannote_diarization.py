from array import array
from dataclasses import dataclass
from pathlib import Path

from dublaro.adapters.diarization import DiarizationOptions
from dublaro.adapters.diarization.pyannote import PyannoteDiarizationAdapter
from dublaro.audio.wav import write_mono_pcm16_wav


@dataclass(frozen=True)
class FakeTurn:
    start: float
    end: float


class FakeOutput:
    def __init__(self) -> None:
        self.exclusive_speaker_diarization = [
            (FakeTurn(0.0, 1.5), "speaker-1"),
            (FakeTurn(1.5, 3.0), "speaker-2"),
        ]


class FakePipeline:
    def __init__(self) -> None:
        self.calls: list[tuple[object, dict[str, int]]] = []

    def __call__(self, audio: object, **kwargs: int) -> FakeOutput:
        self.calls.append((audio, kwargs))
        return FakeOutput()


def test_pyannote_diarization_adapter_converts_turns(tmp_path: Path) -> None:
    audio_path = tmp_path / "audio.wav"
    write_mono_pcm16_wav(
        audio_path,
        array("h", [0, 1000, -1000, 0]),
        sample_rate=16_000,
    )

    pipeline = FakePipeline()
    adapter = PyannoteDiarizationAdapter(pipeline=pipeline)

    turns = adapter.diarize(
        audio_path,
        DiarizationOptions(min_speakers=1, max_speakers=2),
    )

    audio, kwargs = pipeline.calls[0]

    assert kwargs == {"min_speakers": 1, "max_speakers": 2}
    assert isinstance(audio, dict)
    assert audio["sample_rate"] == 16_000
    assert audio["waveform"].shape == (1, 4)
    assert turns[0].speaker == "speaker-1"
    assert turns[0].start == 0.0
    assert turns[0].end == 1.5
    assert turns[1].speaker == "speaker-2"
