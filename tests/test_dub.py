from pathlib import Path

from dublaro.adapters.asr import FakeAsrAdapter
from dublaro.adapters.text_adapter import FakeTextAdapter
from dublaro.adapters.translation import FakeTranslationAdapter
from dublaro.adapters.tts import FakeTtsAdapter
from dublaro.pipeline import dub as dub_module
from dublaro.pipeline.dub import dub_video
from dublaro.pipeline.transcribe import load_transcript


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

    translated = load_transcript(artifacts.translated_transcript_path)
    adapted = load_transcript(artifacts.adapted_transcript_path)
    synthesized = load_transcript(artifacts.synthesized_transcript_path)

    assert translated.target_language == "pl"
    assert translated.segments[0].translated_text.startswith("[pl]")
    assert adapted.segments[0].adapted_text.startswith("[pl]")
    assert synthesized.metadata["tts_adapter"] == "fake-tts"
    assert synthesized.segments[0].generated_audio_path is not None
