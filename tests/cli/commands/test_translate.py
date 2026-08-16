from importlib import import_module
from pathlib import Path

import pytest
from dublaro.adapters.translation import FakeTranslationAdapter
from dublaro.pipeline.transcribe import load_transcript, save_transcript
from dublaro.schemas import Segment, Transcript
from typer.testing import CliRunner

cli = import_module("dublaro.cli.app")
cli_command_translate = import_module("dublaro.cli.commands.translate")


runner = CliRunner()


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
        ollama_model: str | None = None,
        ollama_url: str | None = None,
        ollama_timeout_seconds: float | None = None,
        ollama_temperature: float | None = None,
    ) -> FakeTranslationAdapter:
        calls.append(
            {
                "backend": backend,
                "auto_install": auto_install,
                "ollama_model": ollama_model,
                "ollama_url": ollama_url,
                "ollama_timeout_seconds": ollama_timeout_seconds,
                "ollama_temperature": ollama_temperature,
            }
        )
        return FakeTranslationAdapter()

    monkeypatch.setattr(
        cli_command_translate,
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
    assert calls == [
        {
            "backend": "argos",
            "auto_install": True,
            "ollama_model": None,
            "ollama_url": None,
            "ollama_timeout_seconds": None,
            "ollama_temperature": None,
        }
    ]


def test_translate_command_passes_ollama_options(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transcript_path = tmp_path / "audio.en.json"
    output_path = tmp_path / "audio.pl.json"

    save_transcript(
        Transcript(
            id="audio",
            source_language="en",
            segments=[Segment(id="seg-0001", start=0.0, end=1.0, source_text="Hi")],
        ),
        transcript_path,
    )

    calls: list[dict[str, object]] = []

    def fake_create_translation_adapter(
        backend: str,
        *,
        auto_install: bool = False,
        ollama_model: str | None = None,
        ollama_url: str | None = None,
        ollama_timeout_seconds: float | None = None,
        ollama_temperature: float | None = None,
    ) -> FakeTranslationAdapter:
        calls.append(
            {
                "backend": backend,
                "auto_install": auto_install,
                "ollama_model": ollama_model,
                "ollama_url": ollama_url,
                "ollama_timeout_seconds": ollama_timeout_seconds,
                "ollama_temperature": ollama_temperature,
            }
        )
        return FakeTranslationAdapter()

    monkeypatch.setattr(
        cli_command_translate,
        "create_translation_adapter",
        fake_create_translation_adapter,
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
            "--translator",
            "ollama",
            "--translation-ollama-model",
            "mistral",
            "--translation-ollama-url",
            "http://ollama.local:11434",
            "--translation-ollama-timeout",
            "90",
            "--translation-ollama-temperature",
            "0.2",
        ],
    )

    assert result.exit_code == 0
    assert calls == [
        {
            "backend": "ollama",
            "auto_install": False,
            "ollama_model": "mistral",
            "ollama_url": "http://ollama.local:11434",
            "ollama_timeout_seconds": 90.0,
            "ollama_temperature": 0.2,
        }
    ]
