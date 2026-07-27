import json
from array import array
from pathlib import Path

import pytest
from dublaro.adapters.asr import FakeAsrAdapter
from dublaro.adapters.text_adapter import FakeTextAdapter
from dublaro.adapters.translation import FakeTranslationAdapter
from dublaro.adapters.tts import FakeTtsAdapter
from dublaro.audio.wav import write_mono_pcm16_wav
from dublaro.pipeline import dub_stages
from dublaro.pipeline.dub import dub_video
from dublaro.pipeline.transcribe import load_transcript, save_transcript
from dublaro.pipeline.voices import SpeakerVoice
from dublaro.schemas import Segment, Transcript, VoiceProfile


class ExplodingAsrAdapter:
    name = "exploding-asr"

    def transcribe(self, *args: object, **kwargs: object) -> Transcript:
        raise AssertionError("ASR should not run during resume")


class ExplodingTranslationAdapter:
    name = "exploding-translation"

    def translate_text(self, *args: object, **kwargs: object) -> str:
        raise AssertionError("Translation should not run during resume")


class ExplodingTextAdapter:
    name = "exploding-text-adapter"

    def adapt_segment(self, *args: object, **kwargs: object) -> str:
        raise AssertionError("Text adapter should not run during resume")


class ExplodingTtsAdapter:
    name = "exploding-tts"

    def synthesize_segment(self, *args: object, **kwargs: object) -> Path:
        raise AssertionError("TTS should not run during resume")


class ManifestTtsAdapter(FakeTtsAdapter):
    name = "piper"

    def __init__(self, model_path: str) -> None:
        self.model_path = Path(model_path)
        self.config_path = Path(f"{model_path}.json")
        self.executable = "piper"
        self.speaker = None
        self.model_sample_rate = 16_000


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
        subtitle_path: Path | None = None,
        subtitle_embed: str = "none",
        subtitle_language: str | None = None,
        overwrite=False,
        executable: str = "ffmpeg",
    ) -> Path:
        assert Path(video_path).exists()
        assert Path(speech_track_path).exists()

        output_file = Path(output_path)
        output_file.write_bytes(b"fake dubbed video")
        return output_file

    monkeypatch.setattr(
        dub_stages,
        "extract_audio_from_video",
        fake_extract_audio_from_video,
    )
    monkeypatch.setattr(
        dub_stages,
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
        subtitle_path: Path | None = None,
        subtitle_embed: str = "none",
        subtitle_language: str | None = None,
        overwrite: bool = False,
        executable: str = "ffmpeg",
    ) -> Path:
        output_file = Path(output_path)
        output_file.write_bytes(b"fake dubbed video")
        calls.append({"step": "export", "executable": executable})
        return output_file

    monkeypatch.setattr(
        dub_stages, "extract_audio_from_video", fake_extract_audio_from_video
    )
    monkeypatch.setattr(
        dub_stages,
        "fit_generated_speech_to_segments",
        fake_fit_generated_speech_to_segments,
    )
    monkeypatch.setattr(dub_stages, "export_dubbed_video", fake_export_dubbed_video)

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
        subtitle_path: Path | None = None,
        subtitle_embed: str = "none",
        subtitle_language: str | None = None,
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
        dub_stages,
        "extract_audio_from_video",
        fake_extract_audio_from_video,
    )
    monkeypatch.setattr(
        dub_stages,
        "mix_original_audio_with_dubbed_speech",
        fake_mix_original_audio_with_dubbed_speech,
    )
    monkeypatch.setattr(
        dub_stages,
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
        subtitle_path: Path | None = None,
        subtitle_embed: str = "none",
        subtitle_language: str | None = None,
        overwrite: bool = False,
        executable: str = "ffmpeg",
    ) -> Path:
        output_file = Path(output_path)
        output_file.write_bytes(b"fake dubbed video")
        return output_file

    monkeypatch.setattr(
        dub_stages,
        "extract_audio_from_video",
        fake_extract_audio_from_video,
    )
    monkeypatch.setattr(
        dub_stages,
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


def test_dub_video_can_soft_embed_subtitles_without_sidecar_srt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video_path = tmp_path / "lesson.mp4"
    output_path = tmp_path / "lesson.pl.dubbed.mp4"
    workspace_dir = tmp_path / "workspace"

    video_path.write_bytes(b"fake video")

    export_calls: list[dict[str, object]] = []

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
        subtitle_path: Path | None = None,
        subtitle_embed: str = "none",
        subtitle_language: str | None = None,
        overwrite: bool = False,
        executable: str = "ffmpeg",
    ) -> Path:
        export_calls.append(
            {
                "subtitle_path": subtitle_path,
                "subtitle_embed": subtitle_embed,
                "subtitle_language": subtitle_language,
            }
        )
        output_file = Path(output_path)
        output_file.write_bytes(b"fake dubbed video")
        return output_file

    monkeypatch.setattr(
        dub_stages,
        "extract_audio_from_video",
        fake_extract_audio_from_video,
    )
    monkeypatch.setattr(dub_stages, "export_dubbed_video", fake_export_dubbed_video)

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
        subtitle_embed="soft",
        overwrite=True,
    )

    assert artifacts.srt_path is None
    assert artifacts.embedded_srt_path == workspace_dir / "lesson.pl.embed.srt"
    assert artifacts.embedded_srt_path is not None
    assert artifacts.embedded_srt_path.exists()
    assert export_calls == [
        {
            "subtitle_path": workspace_dir / "lesson.pl.embed.srt",
            "subtitle_embed": "soft",
            "subtitle_language": "pl",
        }
    ]


def test_dub_video_reports_progress(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video_path = tmp_path / "lesson.mp4"
    output_path = tmp_path / "lesson.pl.dubbed.mp4"
    workspace_dir = tmp_path / "workspace"

    video_path.write_bytes(b"fake video")

    events: list[tuple[str, str, str]] = []

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
        subtitle_path: Path | None = None,
        subtitle_embed: str = "none",
        subtitle_language: str | None = None,
        overwrite: bool = False,
        executable: str = "ffmpeg",
    ) -> Path:
        output_file = Path(output_path)
        output_file.write_bytes(b"fake dubbed video")
        return output_file

    def record_progress(step: str, status: str, message: str) -> None:
        events.append((step, status, message))

    monkeypatch.setattr(
        dub_stages,
        "extract_audio_from_video",
        fake_extract_audio_from_video,
    )
    monkeypatch.setattr(
        dub_stages,
        "export_dubbed_video",
        fake_export_dubbed_video,
    )

    dub_video(
        video_path,
        output_path,
        source_language="en",
        target_language="pl",
        workspace_dir=workspace_dir,
        asr_adapter=FakeAsrAdapter(),
        translation_adapter=FakeTranslationAdapter(),
        text_adapter=FakeTextAdapter(),
        tts_adapter=FakeTtsAdapter(),
        progress_callback=record_progress,
        overwrite=True,
    )

    assert [event[0] for event in events] == [
        "extract_audio",
        "extract_audio",
        "transcribe",
        "transcribe",
        "translate",
        "translate",
        "adapt_text",
        "adapt_text",
        "synthesize",
        "synthesize",
        "align_speech",
        "align_speech",
        "export_video",
        "export_video",
        "write_manifest",
        "write_manifest",
    ]

    assert [event[1] for event in events] == [
        "started",
        "finished",
        "started",
        "finished",
        "started",
        "finished",
        "started",
        "finished",
        "started",
        "finished",
        "started",
        "finished",
        "started",
        "finished",
        "started",
        "finished",
    ]


def test_dub_video_writes_manifest_by_default(
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
        subtitle_path: Path | None = None,
        subtitle_embed: str = "none",
        subtitle_language: str | None = None,
        overwrite: bool = False,
        executable: str = "ffmpeg",
    ) -> Path:
        output_file = Path(output_path)
        output_file.write_bytes(b"fake dubbed video")
        return output_file

    monkeypatch.setattr(
        dub_stages, "extract_audio_from_video", fake_extract_audio_from_video
    )
    monkeypatch.setattr(dub_stages, "export_dubbed_video", fake_export_dubbed_video)

    speaker_voices = {
        "speaker-1": SpeakerVoice(
            profile=VoiceProfile(
                speaker_id="speaker-1",
                display_name="Speaker 1",
                language="pl",
                tts_backend="piper",
                metadata={"role": "host"},
            ),
            adapter=ManifestTtsAdapter("models/piper/speaker-1.onnx"),
        )
    }

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
        speaker_voices=speaker_voices,
        overwrite=True,
    )

    manifest_path = artifacts.manifest_path

    assert manifest_path is not None
    assert manifest_path == workspace_dir / "lesson.pl.manifest.json"
    assert manifest_path.exists()

    data = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert data["schema_version"] == 1
    assert data["language"] == {"source": "en", "target": "pl"}
    assert data["adapters"]["asr"]["name"] == "fake-asr"
    assert data["adapters"]["translation"]["name"] == "fake-translation"
    assert data["adapters"]["text_adapter"]["name"] == "fake-text-adapter"
    assert data["adapters"]["tts"]["name"] == "fake-tts"
    speaker_voice = data["adapters"]["speaker_voices"]["speaker-1"]

    assert speaker_voice["profile"] == {
        "speaker_id": "speaker-1",
        "display_name": "Speaker 1",
        "language": "pl",
        "tts_backend": "piper",
        "metadata": {"role": "host"},
    }
    assert speaker_voice["adapter"]["name"] == "piper"
    assert speaker_voice["adapter"]["settings"]["model_path"] == str(
        Path("models/piper/speaker-1.onnx")
    )
    assert speaker_voice["adapter"]["settings"]["model_sample_rate"] == 16_000
    assert data["metadata"]["configured_speaker_voice_count"] == "1"
    assert data["options"]["translation_group_segments"] is True
    assert data["artifacts"]["dubbed_video_path"] == str(output_path)
    assert data["artifacts"]["manifest_path"] == str(manifest_path)
    assert data["metadata"]["source_segment_count"] == "1"


def test_dub_video_resumes_intermediate_artifacts_but_regenerates_final_video(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video_path = tmp_path / "lesson.mp4"
    output_path = tmp_path / "lesson.pl.dubbed.mp4"
    workspace_dir = tmp_path / "workspace"

    video_path.write_bytes(b"fake video")

    extracted_audio_path = workspace_dir / "lesson.audio.wav"
    source_transcript_path = workspace_dir / "lesson.en.json"
    translated_transcript_path = workspace_dir / "lesson.pl.json"
    adapted_transcript_path = workspace_dir / "lesson.pl.adapted.json"
    synthesized_transcript_path = workspace_dir / "lesson.pl.synthesized.json"
    speech_dir = workspace_dir / "lesson.pl.speech"
    speech_path = speech_dir / "seg-0001.wav"
    speech_track_path = workspace_dir / "lesson.pl.speech-track.wav"

    speech_dir.mkdir(parents=True, exist_ok=True)
    extracted_audio_path.write_bytes(b"existing audio")
    speech_path.write_bytes(b"existing speech")
    speech_track_path.write_bytes(b"existing speech track")

    save_transcript(
        Transcript(
            id="lesson.audio",
            source_language="en",
            segments=[
                Segment(
                    id="seg-0001",
                    start=0.0,
                    end=1.0,
                    source_text="Hello",
                    source_language="en",
                )
            ],
        ),
        source_transcript_path,
    )

    save_transcript(
        Transcript(
            id="lesson.audio",
            source_language="en",
            target_language="pl",
            segments=[
                Segment(
                    id="seg-0001",
                    start=0.0,
                    end=1.0,
                    source_text="Hello",
                    translated_text="Czesc",
                    target_language="pl",
                )
            ],
        ),
        translated_transcript_path,
    )

    save_transcript(
        Transcript(
            id="lesson.audio",
            source_language="en",
            target_language="pl",
            segments=[
                Segment(
                    id="seg-0001",
                    start=0.0,
                    end=1.0,
                    source_text="Hello",
                    translated_text="Czesc",
                    adapted_text="Czesc",
                    target_language="pl",
                )
            ],
        ),
        adapted_transcript_path,
    )

    save_transcript(
        Transcript(
            id="lesson.audio",
            source_language="en",
            target_language="pl",
            segments=[
                Segment(
                    id="seg-0001",
                    start=0.0,
                    end=1.0,
                    source_text="Hello",
                    translated_text="Czesc",
                    adapted_text="Czesc",
                    target_language="pl",
                    generated_audio_path=str(speech_path),
                )
            ],
        ),
        synthesized_transcript_path,
    )

    def fail_extract_audio_from_video(*args: object, **kwargs: object) -> Path:
        raise AssertionError("extract_audio_from_video should not run during resume")

    def fake_export_dubbed_video(
        video_path: str | Path,
        speech_track_path: str | Path,
        output_path: str | Path,
        *,
        subtitle_path: Path | None = None,
        subtitle_embed: str = "none",
        subtitle_language: str | None = None,
        overwrite: bool = False,
        executable: str = "ffmpeg",
    ) -> Path:
        output_file = Path(output_path)
        output_file.write_bytes(b"new dubbed video")
        return output_file

    monkeypatch.setattr(
        dub_stages,
        "extract_audio_from_video",
        fail_extract_audio_from_video,
    )
    monkeypatch.setattr(
        dub_stages,
        "export_dubbed_video",
        fake_export_dubbed_video,
    )

    artifacts = dub_video(
        video_path,
        output_path,
        source_language="en",
        target_language="pl",
        workspace_dir=workspace_dir,
        asr_adapter=ExplodingAsrAdapter(),
        translation_adapter=ExplodingTranslationAdapter(),
        text_adapter=ExplodingTextAdapter(),
        tts_adapter=ExplodingTtsAdapter(),
        resume=True,
        write_manifest=False,
    )

    assert artifacts.extracted_audio_path == extracted_audio_path
    assert artifacts.source_transcript_path == source_transcript_path
    assert artifacts.translated_transcript_path == translated_transcript_path
    assert artifacts.adapted_transcript_path == adapted_transcript_path
    assert artifacts.synthesized_transcript_path == synthesized_transcript_path
    assert artifacts.speech_track_path == speech_track_path
    assert artifacts.dubbed_video_path == output_path
    assert output_path.read_bytes() == b"new dubbed video"


def test_dub_video_rejects_resume_with_overwrite(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="resume"):
        dub_video(
            tmp_path / "lesson.mp4",
            tmp_path / "lesson.pl.dubbed.mp4",
            source_language="en",
            target_language="pl",
            workspace_dir=tmp_path / "workspace",
            asr_adapter=FakeAsrAdapter(),
            translation_adapter=FakeTranslationAdapter(),
            text_adapter=FakeTextAdapter(),
            tts_adapter=FakeTtsAdapter(),
            resume=True,
            overwrite=True,
        )
