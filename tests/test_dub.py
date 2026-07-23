from array import array
from pathlib import Path

import pytest
from dublaro.adapters.asr import FakeAsrAdapter
from dublaro.adapters.text_adapter import FakeTextAdapter
from dublaro.adapters.translation import FakeTranslationAdapter
from dublaro.adapters.tts import FakeTtsAdapter
from dublaro.audio.wav import write_mono_pcm16_wav
from dublaro.pipeline import dub as dub_module
from dublaro.pipeline.dub import dub_video
from dublaro.pipeline.transcribe import load_transcript
from dublaro.schemas import Transcript


def test_dub_video_runs_full_pipeline(
    tmp_path: Path,
    monkeypatch,
) -> None:
    video_path = tmp_path / "lesson.mp4"
    output_path = tmp_path / "lesson.pl.dubbed.mp4"
    workspace_dir = tmp_path / "workspace"

    video_path.write_bytes(b"fake video")

    def fake_extract_audio_from_video(
        input_path: str | Path,
        output_path: str | Path | None = None,
        *,
        sample_rate: int = 16_000,
        channels: int = 1,
        overwrite: bool = False,
        executable: str = "ffmpeg",
    ) -> Path:
        input_file = Path(input_path)
        output_file = (
            Path(output_path)
            if output_path is not None
            else input_file.with_suffix(".wav")
        )

        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_bytes(b"fake audio")
        return output_file

    def fake_export_dubbed_video(
        video_path,
        speech_track_path,
        output_path,
        *,
        overwrite=False,
        executable: str = "ffmpeg",
    ) -> Path:
        assert Path(video_path).exists()
        assert Path(speech_track_path).exists()

        output_file = Path(output_path)
        output_file.write_bytes(b"fake dubbed video")
        return output_file

    monkeypatch.setattr(
        dub_module,
        "extract_audio_from_video",
        fake_extract_audio_from_video,
    )
    monkeypatch.setattr(
        dub_module,
        "export_dubbed_video",
        fake_export_dubbed_video,
    )

    artifacts = dub_video(
        video_path,
        output_path,
        source_language="en",
        target_language="pl",
        workspace_dir=workspace_dir,
        asr_adapter=FakeAsrAdapter(),
        translation_adapter=FakeTranslationAdapter(),
        text_adapter=FakeTextAdapter(),
        tts_adapter=FakeTtsAdapter(),
        overwrite=True,
    )

    assert artifacts.dubbed_video_path == output_path
    assert output_path.read_bytes() == b"fake dubbed video"

    assert artifacts.extracted_audio_path.exists()
    assert artifacts.source_transcript_path.exists()
    assert artifacts.translated_transcript_path.exists()
    assert artifacts.adapted_transcript_path.exists()
    assert artifacts.synthesized_transcript_path.exists()
    assert artifacts.speech_track_path.exists()
    assert artifacts.fitted_transcript_path is None
    assert artifacts.fitted_speech_dir is None
    assert artifacts.mix_original_audio_path is None
    assert artifacts.mixed_audio_path is None
    assert artifacts.srt_path is None

    translated = load_transcript(artifacts.translated_transcript_path)
    adapted = load_transcript(artifacts.adapted_transcript_path)
    synthesized = load_transcript(artifacts.synthesized_transcript_path)

    assert translated.target_language == "pl"
    assert translated.segments[0].translated_text.startswith("[pl]")
    assert adapted.segments[0].adapted_text.startswith("[pl]")
    assert synthesized.metadata["tts_adapter"] == "fake-tts"
    assert synthesized.segments[0].generated_audio_path is not None


def test_dub_video_can_fit_speech_before_alignment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video_path = tmp_path / "lesson.mp4"
    output_path = tmp_path / "lesson.pl.dubbed.mp4"
    workspace_dir = tmp_path / "workspace"

    video_path.write_bytes(b"fake video")

    calls: list[dict[str, object]] = []

    def fake_extract_audio_from_video(
        input_path: str | Path,
        output_path: str | Path | None = None,
        *,
        sample_rate: int = 16_000,
        channels: int = 1,
        overwrite: bool = False,
        executable: str = "ffmpeg",
    ) -> Path:
        output_file = Path(output_path or Path(input_path).with_suffix(".wav"))
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_bytes(b"fake audio")
        calls.append({"step": "extract", "executable": executable})
        return output_file

    def fake_fit_generated_speech_to_segments(
        transcript: Transcript,
        *,
        output_dir: str | Path,
        max_speedup: float = 1.35,
        min_overrun_seconds: float = 0.05,
        overwrite: bool = False,
        executable: str = "ffmpeg",
    ) -> Transcript:
        fitted = transcript.model_copy(deep=True)
        fitted_dir = Path(output_dir)
        fitted_audio_path = fitted_dir / "seg-0001.fit.wav"

        write_mono_pcm16_wav(
            fitted_audio_path,
            samples=array("h", [0] * 8_000),
            sample_rate=8_000,
        )

        fitted.segments[0].generated_audio_path = str(fitted_audio_path)

        calls.append(
            {
                "step": "fit",
                "output_dir": fitted_dir,
                "max_speedup": max_speedup,
                "min_overrun_seconds": min_overrun_seconds,
                "overwrite": overwrite,
                "executable": executable,
            }
        )

        return fitted

    def fake_export_dubbed_video(
        video_path: str | Path,
        speech_track_path: str | Path,
        output_path: str | Path,
        *,
        overwrite: bool = False,
        executable: str = "ffmpeg",
    ) -> Path:
        output_file = Path(output_path)
        output_file.write_bytes(b"fake dubbed video")
        calls.append({"step": "export", "executable": executable})
        return output_file

    monkeypatch.setattr(
        dub_module, "extract_audio_from_video", fake_extract_audio_from_video
    )
    monkeypatch.setattr(
        dub_module,
        "fit_generated_speech_to_segments",
        fake_fit_generated_speech_to_segments,
    )
    monkeypatch.setattr(dub_module, "export_dubbed_video", fake_export_dubbed_video)

    artifacts = dub_video(
        video_path,
        output_path,
        source_language="en",
        target_language="pl",
        workspace_dir=workspace_dir,
        asr_adapter=FakeAsrAdapter(),
        translation_adapter=FakeTranslationAdapter(),
        text_adapter=FakeTextAdapter(),
        tts_adapter=FakeTtsAdapter(),
        speech_sample_rate=8_000,
        fit_speech=True,
        max_speech_speedup=1.25,
        min_speech_overrun_seconds=0.1,
        ffmpeg_executable="ffmpeg-test",
        overwrite=True,
    )

    assert artifacts.dubbed_video_path == output_path
    fitted_transcript_path = artifacts.fitted_transcript_path
    fitted_speech_dir = artifacts.fitted_speech_dir

    assert fitted_transcript_path is not None
    assert fitted_speech_dir is not None

    assert fitted_transcript_path == workspace_dir / "lesson.pl.fitted.json"
    assert fitted_speech_dir == workspace_dir / "lesson.pl.fitted-speech"
    assert fitted_transcript_path.exists()

    fitted = load_transcript(fitted_transcript_path)
    assert fitted.segments[0].generated_audio_path == str(
        workspace_dir / "lesson.pl.fitted-speech" / "seg-0001.fit.wav"
    )

    assert calls == [
        {"step": "extract", "executable": "ffmpeg-test"},
        {
            "step": "fit",
            "output_dir": workspace_dir / "lesson.pl.fitted-speech",
            "max_speedup": 1.25,
            "min_overrun_seconds": 0.1,
            "overwrite": True,
            "executable": "ffmpeg-test",
        },
        {"step": "export", "executable": "ffmpeg-test"},
    ]


def test_dub_video_can_mix_original_audio_before_export(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video_path = tmp_path / "lesson.mp4"
    output_path = tmp_path / "lesson.pl.dubbed.mp4"
    workspace_dir = tmp_path / "workspace"

    video_path.write_bytes(b"fake video")

    calls: list[dict[str, object]] = []

    def fake_extract_audio_from_video(
        input_path: str | Path,
        output_path: str | Path | None = None,
        *,
        sample_rate: int = 16_000,
        channels: int = 1,
        overwrite: bool = False,
        executable: str = "ffmpeg",
    ) -> Path:
        output_file = Path(output_path or Path(input_path).with_suffix(".wav"))
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_bytes(b"fake audio")

        calls.append(
            {
                "step": "extract",
                "output_path": output_file,
                "sample_rate": sample_rate,
                "channels": channels,
                "overwrite": overwrite,
                "executable": executable,
            }
        )

        return output_file

    def fake_mix_original_audio_with_dubbed_speech(
        transcript: Transcript,
        *,
        original_audio_path: str | Path,
        speech_track_path: str | Path,
        output_path: str | Path,
        original_gain: float = 1.0,
        ducking_gain: float = 0.25,
        speech_gain: float = 1.0,
        ducking_margin_seconds: float = 0.05,
        ducking_fade_seconds: float = 0.05,
    ) -> Path:
        output_file = Path(output_path)
        output_file.write_bytes(b"fake mixed audio")

        calls.append(
            {
                "step": "mix",
                "original_audio_path": Path(original_audio_path),
                "speech_track_path": Path(speech_track_path),
                "output_path": output_file,
                "original_gain": original_gain,
                "ducking_gain": ducking_gain,
                "speech_gain": speech_gain,
                "ducking_margin_seconds": ducking_margin_seconds,
                "ducking_fade_seconds": ducking_fade_seconds,
            }
        )

        return output_file

    def fake_export_dubbed_video(
        video_path: str | Path,
        speech_track_path: str | Path,
        output_path: str | Path,
        *,
        overwrite: bool = False,
        executable: str = "ffmpeg",
    ) -> Path:
        output_file = Path(output_path)
        output_file.write_bytes(b"fake dubbed video")

        calls.append(
            {
                "step": "export",
                "speech_track_path": Path(speech_track_path),
                "overwrite": overwrite,
                "executable": executable,
            }
        )

        return output_file

    monkeypatch.setattr(
        dub_module,
        "extract_audio_from_video",
        fake_extract_audio_from_video,
    )
    monkeypatch.setattr(
        dub_module,
        "mix_original_audio_with_dubbed_speech",
        fake_mix_original_audio_with_dubbed_speech,
    )
    monkeypatch.setattr(
        dub_module,
        "export_dubbed_video",
        fake_export_dubbed_video,
    )

    artifacts = dub_video(
        video_path,
        output_path,
        source_language="en",
        target_language="pl",
        workspace_dir=workspace_dir,
        asr_adapter=FakeAsrAdapter(),
        translation_adapter=FakeTranslationAdapter(),
        text_adapter=FakeTextAdapter(),
        tts_adapter=FakeTtsAdapter(),
        speech_sample_rate=8_000,
        mix_original_audio=True,
        original_audio_gain=0.8,
        ducking_gain=0.2,
        speech_gain=1.1,
        ducking_margin_seconds=0.2,
        ducking_fade_seconds=0.03,
        ffmpeg_executable="ffmpeg-test",
        overwrite=True,
    )

    assert (
        artifacts.mix_original_audio_path == workspace_dir / "lesson.original-mix.wav"
    )
    assert artifacts.mixed_audio_path == workspace_dir / "lesson.pl.mixed.wav"
    assert artifacts.dubbed_video_path == output_path

    assert calls == [
        {
            "step": "extract",
            "output_path": workspace_dir / "lesson.audio.wav",
            "sample_rate": 16_000,
            "channels": 1,
            "overwrite": True,
            "executable": "ffmpeg-test",
        },
        {
            "step": "extract",
            "output_path": workspace_dir / "lesson.original-mix.wav",
            "sample_rate": 8_000,
            "channels": 1,
            "overwrite": True,
            "executable": "ffmpeg-test",
        },
        {
            "step": "mix",
            "original_audio_path": workspace_dir / "lesson.original-mix.wav",
            "speech_track_path": workspace_dir / "lesson.pl.speech-track.wav",
            "output_path": workspace_dir / "lesson.pl.mixed.wav",
            "original_gain": 0.8,
            "ducking_gain": 0.2,
            "speech_gain": 1.1,
            "ducking_margin_seconds": 0.2,
            "ducking_fade_seconds": 0.03,
        },
        {
            "step": "export",
            "speech_track_path": workspace_dir / "lesson.pl.mixed.wav",
            "overwrite": True,
            "executable": "ffmpeg-test",
        },
    ]


def test_dub_video_can_export_srt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video_path = tmp_path / "lesson.mp4"
    output_path = tmp_path / "lesson.pl.dubbed.mp4"
    workspace_dir = tmp_path / "workspace"

    video_path.write_bytes(b"fake video")

    def fake_extract_audio_from_video(
        input_path: str | Path,
        output_path: str | Path | None = None,
        *,
        sample_rate: int = 16_000,
        channels: int = 1,
        overwrite: bool = False,
        executable: str = "ffmpeg",
    ) -> Path:
        output_file = Path(output_path or Path(input_path).with_suffix(".wav"))
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_bytes(b"fake audio")
        return output_file

    def fake_export_dubbed_video(
        video_path: str | Path,
        speech_track_path: str | Path,
        output_path: str | Path,
        *,
        overwrite: bool = False,
        executable: str = "ffmpeg",
    ) -> Path:
        output_file = Path(output_path)
        output_file.write_bytes(b"fake dubbed video")
        return output_file

    monkeypatch.setattr(
        dub_module,
        "extract_audio_from_video",
        fake_extract_audio_from_video,
    )
    monkeypatch.setattr(
        dub_module,
        "export_dubbed_video",
        fake_export_dubbed_video,
    )

    artifacts = dub_video(
        video_path,
        output_path,
        source_language="en",
        target_language="pl",
        workspace_dir=workspace_dir,
        asr_adapter=FakeAsrAdapter(),
        translation_adapter=FakeTranslationAdapter(),
        text_adapter=FakeTextAdapter(),
        tts_adapter=FakeTtsAdapter(),
        export_srt=True,
        overwrite=True,
    )

    srt_path = artifacts.srt_path

    assert srt_path is not None
    assert srt_path == tmp_path / "lesson.pl.dubbed.srt"
    assert srt_path.exists()
    assert srt_path.read_text(encoding="utf-8") == (
        "1\n00:00:00,000 --> 00:00:01,000\n[pl] This is a placeholder transcript.\n"
    )
