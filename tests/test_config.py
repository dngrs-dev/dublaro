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


def test_load_config_reads_ollama_translation_settings(tmp_path: Path) -> None:
    config_path = tmp_path / "dublaro.toml"
    config_path.write_text(
        """
[dub]
target_language = "pl"

[dub.translation]
backend = "ollama"
ollama_model = "mistral"
ollama_url = "http://ollama.local:11434"
ollama_timeout_seconds = 120.0
ollama_temperature = 0.1
""",
        encoding="utf-8",
    )

    loaded = load_config(config_path)

    assert loaded.config.dub.translation.backend == "ollama"
    assert loaded.config.dub.translation.ollama_model == "mistral"
    assert loaded.config.dub.translation.ollama_url == "http://ollama.local:11434"
    assert loaded.config.dub.translation.ollama_timeout_seconds == 120.0
    assert loaded.config.dub.translation.ollama_temperature == 0.1


def test_load_config_reads_source_separation_settings(tmp_path: Path) -> None:
    config_path = tmp_path / "dublaro.toml"
    config_path.write_text(
        """
[dub]
target_language = "pl"
background_mode = "separated"

[dub.source_separation]
backend = "demucs"
demucs_executable = "demucs-custom"
demucs_model = "mdx-extra"
demucs_device = "cpu"
""",
        encoding="utf-8",
    )

    loaded = load_config(config_path)

    assert loaded.config.dub.background_mode == "separated"
    assert loaded.config.dub.source_separation.backend == "demucs"
    assert loaded.config.dub.source_separation.demucs_executable == "demucs-custom"
    assert loaded.config.dub.source_separation.demucs_model == "mdx-extra"
    assert loaded.config.dub.source_separation.demucs_device == "cpu"


def test_load_config_reads_audio_normalization_settings(tmp_path: Path) -> None:
    config_path = tmp_path / "dublaro.toml"
    config_path.write_text(
        """
[dub]
target_language = "pl"

[dub.audio_normalization]
normalize_final_audio = true
target_final_lufs = -18.0
final_true_peak = -2.0
final_loudness_range = 9.0
""",
        encoding="utf-8",
    )

    loaded = load_config(config_path)

    assert loaded.config.dub.audio_normalization.normalize_final_audio is True
    assert loaded.config.dub.audio_normalization.target_final_lufs == -18.0
    assert loaded.config.dub.audio_normalization.final_true_peak == -2.0
    assert loaded.config.dub.audio_normalization.final_loudness_range == 9.0
