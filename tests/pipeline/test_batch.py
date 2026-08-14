from pathlib import Path

from dublaro.pipeline.batch import (
    default_batch_output_dir,
    default_batch_workspace_dir,
    discover_batch_videos,
    format_video_extensions,
)


def test_discover_batch_videos_from_single_supported_file(tmp_path: Path) -> None:
    video_path = tmp_path / "video.mp4"
    video_path.write_bytes(b"fake video")

    assert discover_batch_videos(video_path) == [video_path]


def test_discover_batch_videos_sorts_supported_files(tmp_path: Path) -> None:
    first = tmp_path / "a.mp4"
    second = tmp_path / "b.mov"
    ignored = tmp_path / "notes.txt"

    second.write_bytes(b"fake video")
    ignored.write_text("ignore", encoding="utf-8")
    first.write_bytes(b"fake video")

    assert discover_batch_videos(tmp_path) == [first, second]


def test_discover_batch_videos_can_search_recursively(tmp_path: Path) -> None:
    nested_dir = tmp_path / "nested"
    nested_dir.mkdir()

    nested_video = nested_dir / "video.mkv"
    nested_video.write_bytes(b"fake video")

    assert discover_batch_videos(tmp_path) == []
    assert discover_batch_videos(tmp_path, recursive=True) == [nested_video]


def test_default_batch_paths_preserve_relative_directories(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    video_path = input_dir / "course" / "lesson.mp4"

    assert (
        default_batch_workspace_dir(
            input_dir,
            video_path,
            tmp_path / "work",
        )
        == tmp_path / "work" / "course" / "lesson"
    )

    assert (
        default_batch_output_dir(
            input_dir,
            video_path,
            tmp_path / "output",
        )
        == tmp_path / "output" / "course"
    )


def test_format_video_extensions() -> None:
    assert ".mp4" in format_video_extensions()
