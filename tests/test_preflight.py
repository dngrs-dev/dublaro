import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest
from dublaro.pipeline import preflight as preflight_module
from dublaro.pipeline.preflight import (
    SpeakerVoicePreflightSettings,
    validate_dub_preflight,
)


def stub_ffmpeg_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        preflight_module.shutil,
        "which",
        lambda executable: f"C:/tools/{executable}.exe",
    )

    def fake_run(
        args: list[str],
        *,
        check: bool,
        capture_output: bool,
        text: bool,
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args, 0, stdout="ok", stderr="")

    monkeypatch.setattr(preflight_module.subprocess, "run", fake_run)


def test_dub_preflight_passes_for_fake_backends(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stub_ffmpeg_ok(monkeypatch)

    video_path = tmp_path / "input.mp4"
    output_path = tmp_path / "output.mp4"
    video_path.write_bytes(b"fake video")

    report = validate_dub_preflight(
        video_path=video_path,
        output_path=output_path,
        workspace_dir=tmp_path / "workspace",
        overwrite=False,
        ffmpeg_executable="ffmpeg",
        asr_backend="fake",
        translation_backend="fake",
        source_language="en",
        target_language="pl",
        install_translation_package=False,
        tts_backend="fake",
    )

    assert report.passed
    assert report.issues == ()


def test_dub_preflight_reports_existing_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stub_ffmpeg_ok(monkeypatch)

    video_path = tmp_path / "input.mp4"
    output_path = tmp_path / "output.mp4"
    video_path.write_bytes(b"fake video")
    output_path.write_bytes(b"existing")

    report = validate_dub_preflight(
        video_path=video_path,
        output_path=output_path,
        workspace_dir=tmp_path / "workspace",
        overwrite=False,
        ffmpeg_executable="ffmpeg",
        asr_backend="fake",
        translation_backend="fake",
        source_language="en",
        target_language="pl",
        install_translation_package=False,
        tts_backend="fake",
    )

    assert not report.passed
    assert {issue.code for issue in report.errors} == {"output_exists"}


def test_dub_preflight_checks_piper_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_which(executable: str) -> str | None:
        if executable == "ffmpeg":
            return "C:/tools/ffmpeg.exe"

        return None

    monkeypatch.setattr(preflight_module.shutil, "which", fake_which)

    def fake_run(
        args: list[str],
        *,
        check: bool,
        capture_output: bool,
        text: bool,
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args, 0, stdout="ok", stderr="")

    monkeypatch.setattr(preflight_module.subprocess, "run", fake_run)

    video_path = tmp_path / "input.mp4"
    output_path = tmp_path / "output.mp4"
    video_path.write_bytes(b"fake video")

    report = validate_dub_preflight(
        video_path=video_path,
        output_path=output_path,
        workspace_dir=tmp_path / "workspace",
        overwrite=False,
        ffmpeg_executable="ffmpeg",
        asr_backend="fake",
        translation_backend="fake",
        source_language="en",
        target_language="pl",
        install_translation_package=False,
        tts_backend="piper",
        piper_model_path=None,
        piper_executable="piper",
    )

    assert {issue.code for issue in report.errors} == {
        "piper_model_missing",
        "piper_executable_missing",
    }


def test_dub_preflight_checks_argos_package(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stub_ffmpeg_ok(monkeypatch)

    package_module = ModuleType("argostranslate.package")
    translate_module = ModuleType("argostranslate.translate")
    argos_module = ModuleType("argostranslate")

    def get_translation_from_codes(source_language: str, target_language: str) -> None:
        raise RuntimeError("missing package")

    translate_module.__dict__.update(
        {"get_translation_from_codes": get_translation_from_codes}
    )
    argos_module.__dict__.update(
        {"package": package_module, "translate": translate_module}
    )

    monkeypatch.setitem(sys.modules, "argostranslate", argos_module)
    monkeypatch.setitem(sys.modules, "argostranslate.package", package_module)
    monkeypatch.setitem(sys.modules, "argostranslate.translate", translate_module)

    video_path = tmp_path / "input.mp4"
    output_path = tmp_path / "output.mp4"
    video_path.write_bytes(b"fake video")

    report = validate_dub_preflight(
        video_path=video_path,
        output_path=output_path,
        workspace_dir=tmp_path / "workspace",
        overwrite=False,
        ffmpeg_executable="ffmpeg",
        asr_backend="fake",
        translation_backend="argos",
        source_language="en",
        target_language="pl",
        install_translation_package=False,
        tts_backend="fake",
    )

    assert {issue.code for issue in report.errors} == {"argos_package_missing"}


def test_dub_preflight_allows_existing_srt_and_manifest_when_resuming(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stub_ffmpeg_ok(monkeypatch)

    video_path = tmp_path / "input.mp4"
    output_path = tmp_path / "output.mp4"
    srt_path = tmp_path / "output.srt"
    manifest_path = tmp_path / "manifest.json"

    video_path.write_bytes(b"fake video")
    srt_path.write_text("existing subtitles", encoding="utf-8")
    manifest_path.write_text("{}", encoding="utf-8")

    report = validate_dub_preflight(
        video_path=video_path,
        output_path=output_path,
        workspace_dir=tmp_path / "workspace",
        overwrite=False,
        ffmpeg_executable="ffmpeg",
        asr_backend="fake",
        translation_backend="fake",
        source_language="en",
        target_language="pl",
        install_translation_package=False,
        tts_backend="fake",
        export_srt=True,
        srt_output_path=srt_path,
        write_manifest=True,
        manifest_output_path=manifest_path,
        resume=True,
    )

    assert report.passed
    assert report.issues == ()


def test_dub_preflight_checks_speaker_voice_piper_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stub_ffmpeg_ok(monkeypatch)

    video_path = tmp_path / "input.mp4"
    output_path = tmp_path / "output.mp4"
    video_path.write_bytes(b"fake video")

    report = validate_dub_preflight(
        video_path=video_path,
        output_path=output_path,
        workspace_dir=tmp_path / "workspace",
        overwrite=False,
        ffmpeg_executable="ffmpeg",
        asr_backend="fake",
        translation_backend="fake",
        source_language="en",
        target_language="pl",
        install_translation_package=False,
        tts_backend="fake",
        speaker_voices={
            "SPEAKER_00": SpeakerVoicePreflightSettings(
                tts_backend="piper",
                piper_model_path=tmp_path / "missing.onnx",
                piper_config_path=tmp_path / "missing.onnx.json",
                piper_executable="piper",
            )
        },
    )

    messages = [issue.message for issue in report.errors]

    assert not report.passed
    assert any(
        'Speaker voice "SPEAKER_00" Piper model does not exist' in message
        for message in messages
    )
    assert any(
        'Speaker voice "SPEAKER_00" Piper config does not exist' in message
        for message in messages
    )
