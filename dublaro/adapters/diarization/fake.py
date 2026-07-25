from pathlib import Path

from dublaro.adapters.diarization.base import DiarizationOptions, DiarizationTurn


class FakeDiarizationAdapter:
    name = "fake-diarization"

    def diarize(
        self,
        audio_path: Path,
        options: DiarizationOptions,
    ) -> list[DiarizationTurn]:
        return [
            DiarizationTurn(
                start=0.0,
                end=1_000_000.0,
                speaker="speaker-1",
            )
        ]
