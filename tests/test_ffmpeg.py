import subprocess
from pathlib import Path

import pytest
from dublaro.audio import ffmpeg
from dublaro.audio.ffmpeg import FFmpegError, extract_audio_from_video, run_ffmpeg


def test_extract_audio_uses_expected_ffmpeg_arguments(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_video = tmp_path / "video.mp4"
    output_audio = tmp_path / "audio" / "video.wav"
    input_video.write_bytes(b"fake video")

    calls: list[list[str | Path]] = []

    def fake_run_ffmpeg(
        args: list[str | Path],
        *,
        executable: str = "ffmpeg",
    ) -> subprocess.CompletedProcess[str]:
        calls.append(args)
        return subprocess.CompletedProcess(args=[executable], returncode=0)

    monkeypatch.setattr(ffmpeg, "run_ffmpeg", fake_run_ffmpeg)

    result = extract_audio_from_video(input_video, output_audio)

    assert result == output_audio
    assert output_audio.parent.exists()

    args = calls[0]
    assert "-i" in args
    assert input_video in args
    assert "-vn" in args
    assert "-ac" in args
    assert "1" in args
    assert "-ar" in args
    assert "16000" in args
    assert "pcm_s16le" in args
    assert output_audio in args


def test_extract_audio_uses_default_output_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_video = tmp_path / "lesson.mp4"
    input_video.write_bytes(b"fake video")

    monkeypatch.setattr(
        ffmpeg,
        "run_ffmpeg",
        lambda args, *, executable="ffmpeg": subprocess.CompletedProcess(
            args=[executable],
            returncode=0,
        ),
    )

    result = extract_audio_from_video(input_video)

    assert result == tmp_path / "lesson.wav"


def test_extract_audio_rejects_existing_output_without_overwrite(
    tmp_path: Path,
) -> None:
    input_video = tmp_path / "video.mp4"
    output_audio = tmp_path / "video.wav"

    input_video.write_bytes(b"fake video")
    output_audio.write_bytes(b"existing audio")

    with pytest.raises(FileExistsError):
        extract_audio_from_video(input_video, output_audio)


def test_extract_audio_rejects_missing_input(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        extract_audio_from_video(tmp_path / "missing.mp4")


def test_run_ffmpeg_raises_when_process_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ffmpeg, "find_ffmpeg", lambda executable="ffmpeg": "ffmpeg")

    def fake_subprocess_run(*args, **kwargs) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args=["ffmpeg"],
            returncode=1,
            stderr="bad input",
        )

    monkeypatch.setattr(subprocess, "run", fake_subprocess_run)

    with pytest.raises(FFmpegError, match="bad input"):
        run_ffmpeg(["-version"])
