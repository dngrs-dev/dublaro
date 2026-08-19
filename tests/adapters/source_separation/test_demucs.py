import subprocess
from pathlib import Path

import dublaro.adapters.source_separation.demucs as demucs_module
import pytest
from dublaro.adapters.source_separation import (
    DemucsSourceSeparationAdapter,
    SourceSeparationOptions,
)


def test_demucs_source_separation_runs_demucs_and_converts_stems(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audio_path = tmp_path / "source.wav"
    background_path = tmp_path / "background.wav"
    voice_path = tmp_path / "voice.wav"

    audio_path.write_bytes(b"audio")

    commands: list[list[str]] = []

    def fake_run(
        args: list[str],
        *,
        check: bool,
        capture_output: bool,
        text: bool,
    ) -> subprocess.CompletedProcess[str]:
        assert check is True
        assert capture_output is True
        assert text is True

        commands.append(args)

        if args[0] == "demucs-test":
            output_root = Path(args[args.index("--out") + 1])
            model = args[args.index("-n") + 1]
            stem_dir = output_root / model / "source"
            stem_dir.mkdir(parents=True)

            (stem_dir / "no_vocals.wav").write_bytes(b"background stem")
            (stem_dir / "vocals.wav").write_bytes(b"voice stem")

        if args[0] == "ffmpeg-test":
            input_file = Path(args[args.index("-i") + 1])
            output_file = Path(args[-1])
            output_file.parent.mkdir(parents=True, exist_ok=True)
            output_file.write_bytes(f"converted {input_file.name}".encode())

        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    monkeypatch.setattr(demucs_module.subprocess, "run", fake_run)

    result = DemucsSourceSeparationAdapter(
        executable="demucs-test",
        model="htdemucs-test",
        ffmpeg_executable="ffmpeg-test",
    ).separate_sources(
        audio_path,
        background_output_path=background_path,
        voice_output_path=voice_path,
        options=SourceSeparationOptions(sample_rate=16_000),
        overwrite=True,
    )

    assert result.background_audio_path == background_path
    assert result.voice_audio_path == voice_path
    assert background_path.read_bytes() == b"converted no_vocals.wav"
    assert voice_path.read_bytes() == b"converted vocals.wav"

    assert commands[0][:5] == [
        "demucs-test",
        "--two-stems",
        "vocals",
        "-n",
        "htdemucs-test",
    ]
    assert commands[0][-1] == str(audio_path)

    assert commands[1][0] == "ffmpeg-test"
    assert commands[1][commands[1].index("-ar") + 1] == "16000"
    assert commands[1][-1] == str(background_path)

    assert commands[2][0] == "ffmpeg-test"
    assert commands[2][commands[2].index("-ar") + 1] == "16000"
    assert commands[2][-1] == str(voice_path)


def test_demucs_source_separation_rejects_existing_output_without_overwrite(
    tmp_path: Path,
) -> None:
    audio_path = tmp_path / "source.wav"
    background_path = tmp_path / "background.wav"
    voice_path = tmp_path / "voice.wav"

    audio_path.write_bytes(b"audio")
    background_path.write_bytes(b"existing")

    with pytest.raises(FileExistsError, match="Output already exists"):
        DemucsSourceSeparationAdapter().separate_sources(
            audio_path,
            background_output_path=background_path,
            voice_output_path=voice_path,
            options=SourceSeparationOptions(sample_rate=16_000),
            overwrite=False,
        )


def test_demucs_source_separation_reports_missing_demucs_executable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audio_path = tmp_path / "source.wav"
    background_path = tmp_path / "background.wav"
    voice_path = tmp_path / "voice.wav"

    audio_path.write_bytes(b"audio")

    def fake_run(
        args: list[str],
        *,
        check: bool,
        capture_output: bool,
        text: bool,
    ) -> subprocess.CompletedProcess[str]:
        raise FileNotFoundError(args[0])

    monkeypatch.setattr(demucs_module.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="Demucs executable was not found"):
        DemucsSourceSeparationAdapter(executable="missing-demucs").separate_sources(
            audio_path,
            background_output_path=background_path,
            voice_output_path=voice_path,
            options=SourceSeparationOptions(sample_rate=16_000),
            overwrite=True,
        )
