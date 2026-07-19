from pathlib import Path

import pytest
from dublaro import cli
from dublaro.adapters.translation import FakeTranslationAdapter
from dublaro.pipeline.transcribe import load_transcript, save_transcript
from dublaro.schemas import Segment, Transcript
from typer.testing import CliRunner

runner = CliRunner()


def test_cli_shows_version() -> None:
    result = runner.invoke(cli.app, ["--version"])

    assert result.exit_code == 0
    assert "dublaro" in result.output


def test_extract_audio_command_calls_extractor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_video = tmp_path / "video.mp4"
    output_audio = tmp_path / "voice.wav"
    input_video.write_bytes(b"fake video")

    calls: list[dict[str, object]] = []

    def fake_extract_audio_from_video(
        input_path: Path,
        output_path: Path | None = None,
        *,
        sample_rate: int = 16_000,
        channels: int = 1,
        overwrite: bool = False,
    ) -> Path:
        calls.append(
            {
                "input_path": input_path,
                "output_path": output_path,
                "sample_rate": sample_rate,
                "channels": channels,
                "overwrite": overwrite,
            }
        )
        return output_path or input_path.with_suffix(".wav")

    monkeypatch.setattr(
        cli,
        "extract_audio_from_video",
        fake_extract_audio_from_video,
    )

    result = runner.invoke(
        cli.app,
        [
            "extract-audio",
            str(input_video),
            "--output",
            str(output_audio),
            "--sample-rate",
            "24000",
            "--channels",
            "2",
            "--overwrite",
        ],
    )

    assert result.exit_code == 0
    assert "Audio extracted" in result.output
    assert calls == [
        {
            "input_path": input_video,
            "output_path": output_audio,
            "sample_rate": 24_000,
            "channels": 2,
            "overwrite": True,
        }
    ]


def test_extract_audio_command_reports_extractor_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_video = tmp_path / "video.mp4"
    input_video.write_bytes(b"fake video")

    def fake_extract_audio_from_video(*args, **kwargs) -> Path:
        raise FileExistsError("already exists")

    monkeypatch.setattr(
        cli,
        "extract_audio_from_video",
        fake_extract_audio_from_video,
    )

    result = runner.invoke(cli.app, ["extract-audio", str(input_video)])

    assert result.exit_code == 1
    assert "already exists" in result.output


def test_transcribe_command_writes_transcript(tmp_path: Path) -> None:
    audio_path = tmp_path / "audio.wav"
    output_path = tmp_path / "transcribe.json"
    audio_path.write_bytes(b"fake audio")

    result = runner.invoke(
        cli.app,
        [
            "transcribe",
            str(audio_path),
            "--output",
            str(output_path),
            "--language",
            "en",
        ],
    )

    assert result.exit_code == 0
    assert output_path.exists()

    transcript = load_transcript(output_path)

    assert transcript.id == "audio"
    assert transcript.source_language == "en"
    assert transcript.segments[0].source_text == "This is a placeholder transcript."
    assert transcript.metadata["adapter"] == "fake-asr"


def test_translate_command_writes_translated_transcript(tmp_path: Path) -> None:
    transcript_path = tmp_path / "audio.en.json"
    output_path = tmp_path / "audio.pl.json"
    save_transcript(
        Transcript(
            id="audio",
            source_language="en",
            segments=[
                Segment(
                    id="seg-0001",
                    start=0.0,
                    end=1.0,
                    source_text="Hello world",
                )
            ],
        ),
        transcript_path,
    )

    result = runner.invoke(
        cli.app,
        [
            "translate",
            str(transcript_path),
            "--to",
            "pl",
            "--output",
            str(output_path),
        ],
    )

    assert result.exit_code == 0
    assert output_path.exists()

    transcript = load_transcript(output_path)

    assert transcript.target_language == "pl"
    assert transcript.segments[0].translated_text == "[pl] Hello world"
    assert transcript.metadata["translation_adapter"] == "fake-translation"


def test_translate_command_passes_translator_options(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transcript_path = tmp_path / "audio.en.json"
    output_path = tmp_path / "audio.pl.json"

    save_transcript(
        Transcript(
            id="audio",
            source_language="en",
            segments=[
                Segment(
                    id="seg-0001",
                    start=0.0,
                    end=1.0,
                    source_text="Hello world",
                )
            ],
        ),
        transcript_path,
    )

    calls: list[dict[str, object]] = []

    def fake_create_translation_adapter(
        backend: str,
        *,
        auto_install: bool = False,
    ) -> FakeTranslationAdapter:
        calls.append({"backend": backend, "auto_install": auto_install})
        return FakeTranslationAdapter()

    monkeypatch.setattr(
        cli,
        "create_translation_adapter",
        fake_create_translation_adapter,
    )

    result = runner.invoke(
        cli.app,
        [
            "translate",
            str(transcript_path),
            "--to",
            "es",
            "--output",
            str(output_path),
            "--translator",
            "argos",
            "--install-package",
        ],
    )

    assert result.exit_code == 0
    assert calls == [{"backend": "argos", "auto_install": True}]


def test_adapt_text_command_writes_adapted_transcript(tmp_path: Path) -> None:
    transcript_path = tmp_path / "audio.pl.json"
    output_path = tmp_path / "audio.pl.adapted.json"

    save_transcript(
        Transcript(
            id="audio",
            source_language="en",
            target_language="pl",
            segments=[
                Segment(
                    id="seg-0001",
                    start=0.0,
                    end=1.0,
                    source_text="Hello world",
                    translated_text="  Cześć    świecie  ",
                    target_language="pl",
                )
            ],
        ),
        transcript_path,
    )

    result = runner.invoke(
        cli.app,
        [
            "adapt-text",
            str(transcript_path),
            "--output",
            str(output_path),
            "--max-chars-per-second",
            "14",
        ],
    )

    assert result.exit_code == 0
    assert output_path.exists()

    transcript = load_transcript(output_path)

    assert transcript.segments[0].adapted_text == "Cześć świecie"
    assert transcript.metadata["text_adapter"] == "fake-text-adapter"
    assert transcript.metadata["text_adapter_max_chars_per_second"] == "14.0"


def test_synthesize_command_writes_audio_and_transcript(tmp_path: Path) -> None:
    transcript_path = tmp_path / "audio.pl.adapted.json"
    output_dir = tmp_path / "speech"
    output_path = tmp_path / "audio.pl.synthesized.json"

    save_transcript(
        Transcript(
            id="audio",
            source_language="en",
            target_language="pl",
            segments=[
                Segment(
                    id="seg-0001",
                    start=0.0,
                    end=1.0,
                    source_text="Hello world",
                    translated_text="Cześć świecie",
                    adapted_text="Cześć świecie",
                )
            ],
        ),
        transcript_path,
    )

    result = runner.invoke(
        cli.app,
        [
            "synthesize",
            str(transcript_path),
            "--output-dir",
            str(output_dir),
            "--output",
            str(output_path),
            "--sample-rate",
            "16000",
        ],
    )

    assert result.exit_code == 0
    assert output_path.exists()
    assert (output_dir / "seg-0001.wav").exists()

    transcript = load_transcript(output_path)

    assert transcript.metadata["tts_adapter"] == "fake-tts"
    assert transcript.metadata["tts_sample_rate"] == "16000"
    assert transcript.segments[0].generated_audio_path == str(
        output_dir / "seg-0001.wav"
    )
