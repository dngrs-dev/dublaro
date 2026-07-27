import subprocess
from pathlib import Path

import pytest
from dublaro.audio import ffmpeg
from dublaro.audio.ffmpeg import (
    FFmpegError,
    build_atempo_filter,
    change_audio_tempo,
    extract_audio_from_video,
    replace_video_audio,
    replace_video_audio_with_hard_subtitles,
    replace_video_audio_with_soft_subtitles,
    run_ffmpeg,
    slow_video,
)


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


def test_replace_video_audio_uses_expected_ffmpeg_arguments(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video_path = tmp_path / "video.mp4"
    audio_path = tmp_path / "speech.wav"
    output_path = tmp_path / "dubbed.mp4"

    video_path.write_bytes(b"fake video")
    audio_path.write_bytes(b"fake audio")

    calls: list[list[str | Path]] = []

    def fake_run_ffmpeg(
        args: list[str | Path],
        *,
        executable: str = "ffmpeg",
    ) -> subprocess.CompletedProcess[str]:
        calls.append(args)
        return subprocess.CompletedProcess(args=[executable], returncode=0)

    monkeypatch.setattr(ffmpeg, "run_ffmpeg", fake_run_ffmpeg)

    result = replace_video_audio(video_path, audio_path, output_path)

    assert result == output_path

    args = calls[0]

    assert "-i" in args
    assert video_path in args
    assert audio_path in args
    assert "-map" in args
    assert "0:v:0" in args
    assert "1:a:0" in args
    assert "-c:v" in args
    assert "copy" in args
    assert "-c:a" in args
    assert "aac" in args
    assert "-shortest" in args
    assert output_path in args


def test_replace_video_audio_with_soft_subtitles_uses_expected_ffmpeg_arguments(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video_path = tmp_path / "video.mp4"
    audio_path = tmp_path / "speech.wav"
    subtitle_path = tmp_path / "subs.srt"
    output_path = tmp_path / "dubbed.mp4"

    video_path.write_bytes(b"fake video")
    audio_path.write_bytes(b"fake audio")
    subtitle_path.write_text(
        "1\n00:00:00,000 --> 00:00:01,000\nHello\n", encoding="utf-8"
    )

    calls: list[list[str | Path]] = []

    def fake_run_ffmpeg(
        args: list[str | Path],
        *,
        executable: str = "ffmpeg",
    ) -> subprocess.CompletedProcess[str]:
        calls.append(args)
        return subprocess.CompletedProcess(args=[executable], returncode=0)

    monkeypatch.setattr(ffmpeg, "run_ffmpeg", fake_run_ffmpeg)

    result = replace_video_audio_with_soft_subtitles(
        video_path,
        audio_path,
        subtitle_path,
        output_path,
        subtitle_language="pl",
    )

    assert result == output_path

    args = calls[0]

    assert video_path in args
    assert audio_path in args
    assert subtitle_path in args
    assert "2:0" in args
    assert "-c:s" in args
    assert "mov_text" in args
    assert "-metadata:s:s:0" in args
    assert "language=pl" in args
    assert output_path in args


def test_replace_video_audio_with_hard_subtitles_uses_expected_ffmpeg_arguments(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video_path = tmp_path / "video.mp4"
    audio_path = tmp_path / "speech.wav"
    subtitle_path = tmp_path / "subs.srt"
    output_path = tmp_path / "dubbed.mp4"

    video_path.write_bytes(b"fake video")
    audio_path.write_bytes(b"fake audio")
    subtitle_path.write_text(
        "1\n00:00:00,000 --> 00:00:01,000\nHello\n", encoding="utf-8"
    )

    calls: list[tuple[list[str | Path], str | Path | None]] = []

    def fake_run_ffmpeg(
        args: list[str | Path],
        *,
        executable: str = "ffmpeg",
        cwd: str | Path | None = None,
    ) -> subprocess.CompletedProcess[str]:
        calls.append((args, cwd))
        return subprocess.CompletedProcess(args=[executable], returncode=0)

    monkeypatch.setattr(ffmpeg, "run_ffmpeg", fake_run_ffmpeg)

    result = replace_video_audio_with_hard_subtitles(
        video_path,
        audio_path,
        subtitle_path,
        output_path,
    )

    assert result == output_path

    args, cwd = calls[0]

    assert cwd == subtitle_path.parent.resolve()
    assert "-vf" in args
    assert f"subtitles=filename='{subtitle_path.name}'" in args


def test_replace_video_audio_rejects_existing_output_without_overwrite(
    tmp_path: Path,
) -> None:
    video_path = tmp_path / "video.mp4"
    audio_path = tmp_path / "speech.wav"
    output_path = tmp_path / "dubbed.mp4"

    video_path.write_bytes(b"fake video")
    audio_path.write_bytes(b"fake audio")
    output_path.write_bytes(b"existing output")

    with pytest.raises(FileExistsError):
        replace_video_audio(video_path, audio_path, output_path)


def test_build_atempo_filter_chains_large_tempo_factor() -> None:
    assert build_atempo_filter(2.5) == "atempo=2,atempo=1.25"


def test_change_audio_tempo_uses_expected_ffmpeg_arguments(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_audio = tmp_path / "speech.wav"
    output_audio = tmp_path / "speech.fit.wav"
    input_audio.write_bytes(b"fake wav")

    calls: list[list[str | Path]] = []

    def fake_run_ffmpeg(
        args: list[str | Path],
        *,
        executable: str = "ffmpeg",
    ) -> subprocess.CompletedProcess[str]:
        calls.append(args)
        return subprocess.CompletedProcess(args=[executable], returncode=0)

    monkeypatch.setattr(ffmpeg, "run_ffmpeg", fake_run_ffmpeg)

    result = change_audio_tempo(
        input_audio,
        output_audio,
        tempo_factor=2.5,
        sample_rate=22_050,
        overwrite=True,
    )

    assert result == output_audio

    args = calls[0]
    assert "-filter:a" in args
    assert "atempo=2,atempo=1.25" in args
    assert "-ar" in args
    assert "22050" in args
    assert "pcm_s16le" in args
    assert output_audio in args


def test_slow_video_uses_expected_ffmpeg_arguments(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_video = tmp_path / "video.mp4"
    output_video = tmp_path / "video.slow.mp4"
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

    result = slow_video(
        input_video,
        output_video,
        slowdown_factor=1.5,
        overwrite=True,
        executable="ffmpeg-test",
    )

    assert result == output_video

    args = calls[0]
    assert "-filter:v" in args
    assert "setpts=1.5*PTS" in args
    assert "-an" in args
    assert "libx264" in args
    assert output_video in args


def test_change_audio_tempo_rejects_in_place_output(tmp_path: Path) -> None:
    input_audio = tmp_path / "speech.wav"
    input_audio.write_bytes(b"fake wav")

    with pytest.raises(ValueError, match="in place"):
        change_audio_tempo(input_audio, input_audio, tempo_factor=1.2)
