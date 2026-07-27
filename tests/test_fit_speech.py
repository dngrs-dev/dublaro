from array import array
from dataclasses import dataclass
from pathlib import Path

import dublaro.pipeline.fit_speech as fit_speech_module
import pytest
from dublaro.audio.wav import write_mono_pcm16_wav
from dublaro.schemas import Segment, Transcript


@dataclass
class TempoCall:
    input_path: Path
    output_path: Path
    tempo_factor: float
    sample_rate: int | None
    overwrite: bool
    executable: str


def test_fit_generated_speech_to_segments_fits_overlong_clip(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clip_path = tmp_path / "seg-0001.wav"
    output_dir = tmp_path / "fitted"

    write_mono_pcm16_wav(clip_path, samples=array("h", [0] * 12), sample_rate=10)

    calls: list[TempoCall] = []

    def fake_change_audio_tempo(
        input_path: str | Path,
        output_path: str | Path,
        *,
        tempo_factor: float,
        sample_rate: int | None = None,
        overwrite: bool = False,
        executable: str = "ffmpeg",
    ) -> Path:
        output_file = Path(output_path)
        calls.append(
            TempoCall(
                input_path=Path(input_path),
                output_path=output_file,
                tempo_factor=tempo_factor,
                sample_rate=sample_rate,
                overwrite=overwrite,
                executable=executable,
            )
        )
        write_mono_pcm16_wav(output_file, samples=array("h", [0] * 10), sample_rate=10)
        return output_file

    monkeypatch.setattr(
        fit_speech_module,
        "change_audio_tempo",
        fake_change_audio_tempo,
    )

    transcript = Transcript(
        id="lesson-1",
        source_language="en",
        target_language="pl",
        segments=[
            Segment(
                id="seg-0001",
                start=0.0,
                end=1.0,
                generated_audio_path=str(clip_path),
            )
        ],
    )

    fitted = fit_speech_module.fit_generated_speech_to_segments(
        transcript,
        output_dir=output_dir,
        max_speedup=1.5,
        min_overrun_seconds=0.05,
        overwrite=True,
        executable="ffmpeg-test",
    )

    assert calls[0].input_path == clip_path
    assert calls[0].tempo_factor == pytest.approx(1.2)
    assert calls[0].sample_rate == 10
    assert calls[0].overwrite is True
    assert calls[0].executable == "ffmpeg-test"
    assert fitted.segments[0].generated_audio_path == str(
        output_dir / "seg-0001.fit.wav"
    )
    assert fitted.metadata["speech_fitting_fitted_segments"] == "1"
    assert transcript.segments[0].generated_audio_path == str(clip_path)


def test_fit_generated_speech_to_segments_skips_clip_inside_tolerance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clip_path = tmp_path / "seg-0001.wav"

    write_mono_pcm16_wav(clip_path, samples=array("h", [0] * 10), sample_rate=10)

    calls: list[object] = []

    monkeypatch.setattr(
        fit_speech_module,
        "change_audio_tempo",
        lambda *args, **kwargs: calls.append(args),
    )

    transcript = Transcript(
        id="lesson-1",
        source_language="en",
        segments=[
            Segment(
                id="seg-0001",
                start=0.0,
                end=1.0,
                generated_audio_path=str(clip_path),
            )
        ],
    )

    fitted = fit_speech_module.fit_generated_speech_to_segments(
        transcript,
        output_dir=tmp_path / "fitted",
    )

    assert calls == []
    assert fitted.segments[0].generated_audio_path == str(clip_path)
    assert fitted.metadata["speech_fitting_fitted_segments"] == "0"


def test_fit_generated_speech_to_segments_rejects_excessive_speedup(
    tmp_path: Path,
) -> None:
    clip_path = tmp_path / "seg-0001.wav"

    write_mono_pcm16_wav(clip_path, samples=array("h", [0] * 20), sample_rate=10)

    transcript = Transcript(
        id="lesson-1",
        source_language="en",
        segments=[
            Segment(
                id="seg-0001",
                start=0.0,
                end=1.0,
                generated_audio_path=str(clip_path),
            )
        ],
    )

    with pytest.raises(ValueError, match="above max_speedup"):
        fit_speech_module.fit_generated_speech_to_segments(
            transcript,
            output_dir=tmp_path / "fitted",
            max_speedup=1.35,
        )


def test_fit_generated_speech_to_segments_caps_speedup_when_unfit_overruns_allowed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clip_path = tmp_path / "seg-0001.wav"
    output_dir = tmp_path / "fitted"

    write_mono_pcm16_wav(clip_path, samples=array("h", [0] * 20), sample_rate=10)

    calls: list[TempoCall] = []

    def fake_change_audio_tempo(
        input_path: str | Path,
        output_path: str | Path,
        *,
        tempo_factor: float,
        sample_rate: int | None = None,
        overwrite: bool = False,
        executable: str = "ffmpeg",
    ) -> Path:
        output_file = Path(output_path)
        calls.append(
            TempoCall(
                input_path=Path(input_path),
                output_path=output_file,
                tempo_factor=tempo_factor,
                sample_rate=sample_rate,
                overwrite=overwrite,
                executable=executable,
            )
        )
        write_mono_pcm16_wav(output_file, samples=array("h", [0] * 16), sample_rate=10)
        return output_file

    monkeypatch.setattr(
        fit_speech_module,
        "change_audio_tempo",
        fake_change_audio_tempo,
    )

    transcript = Transcript(
        id="lesson-1",
        source_language="en",
        segments=[
            Segment(
                id="seg-0001",
                start=0.0,
                end=1.0,
                generated_audio_path=str(clip_path),
            )
        ],
    )

    fitted = fit_speech_module.fit_generated_speech_to_segments(
        transcript,
        output_dir=output_dir,
        max_speedup=1.25,
        allow_unfit_overruns=True,
    )

    assert calls[0].tempo_factor == pytest.approx(1.25)
    assert fitted.metadata["speech_fitting_fitted_segments"] == "1"
    assert fitted.metadata["speech_fitting_unresolved_segments"] == "1"
