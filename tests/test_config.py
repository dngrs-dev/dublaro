from pathlib import Path

import pytest
from dublaro.config import DublaroConfigError, load_config, resolve_config_path


def test_load_config_reads_dub_settings(tmp_path: Path) -> None:
    config_path = tmp_path / "dublaro.toml"
    config_path.write_text(
        """
[dub]
source_language = "en"
target_language = "pl"
output_path = "out/video.pl.mp4"
workspace_dir = ".dublaro/video"
resume = true
preflight = false

[dub.asr]
backend = "faster-whisper"
model_size = "small"

[dub.translation]
backend = "argos"
install_package = true
group_segments = true

[dub.srt]
export = true
text_mode = "adapted"
""",
        encoding="utf-8",
    )

    loaded = load_config(config_path)

    assert loaded.base_dir == tmp_path
    assert loaded.config.dub.source_language == "en"
    assert loaded.config.dub.target_language == "pl"
    assert loaded.config.dub.asr.backend == "faster-whisper"
    assert loaded.config.dub.translation.install_package is True
    assert loaded.config.dub.srt.export is True


def test_load_config_rejects_unknown_keys(tmp_path: Path) -> None:
    config_path = tmp_path / "dublaro.toml"
    config_path.write_text("[dub]\nunknown = true\n", encoding="utf-8")

    with pytest.raises(DublaroConfigError, match="unknown"):
        load_config(config_path)


def test_resolve_config_path_uses_config_directory(tmp_path: Path) -> None:
    assert resolve_config_path(Path("out/video.mp4"), tmp_path) == (
        tmp_path / "out" / "video.mp4"
    )


def test_resolve_config_path_keeps_absolute_path(tmp_path: Path) -> None:
    absolute_path = tmp_path / "video.mp4"

    assert resolve_config_path(absolute_path, tmp_path / "other") == absolute_path


def test_load_config_rejects_output_path_with_output_dir(tmp_path: Path) -> None:
    config_path = tmp_path / "dublaro.toml"
    config_path.write_text(
        """
[dub]
target_language = "pl"
output_path = "exact.mp4"
output_dir = "out"
""",
        encoding="utf-8",
    )

    with pytest.raises(DublaroConfigError, match="output_path"):
        load_config(config_path)


def test_load_config_reads_ollama_text_adapter_settings(tmp_path: Path) -> None:
    config_path = tmp_path / "dublaro.toml"
    config_path.write_text(
        """
[dub]
target_language = "pl"

[dub.text_adapter]
backend = "ollama"
ollama_model = "llama3.1"
ollama_url = "http://localhost:11434"
ollama_timeout_seconds = 12.0
ollama_temperature = 0.1
""",
        encoding="utf-8",
    )

    loaded = load_config(config_path)

    assert loaded.config.dub.text_adapter.backend == "ollama"
    assert loaded.config.dub.text_adapter.ollama_model == "llama3.1"
    assert loaded.config.dub.text_adapter.ollama_url == "http://localhost:11434"
    assert loaded.config.dub.text_adapter.ollama_timeout_seconds == 12.0
    assert loaded.config.dub.text_adapter.ollama_temperature == 0.1
