from pathlib import Path

from dublaro.pipeline.export import (
    default_dubbed_video_path,
    default_dubbed_video_path_in_dir,
)


def test_default_dubbed_video_path() -> None:
    assert default_dubbed_video_path("lesson.mp4", "pl") == Path("lesson.pl.dubbed.mp4")


def test_default_dubbed_video_path_in_dir() -> None:
    assert default_dubbed_video_path_in_dir("lesson.mp4", "pl", "out") == Path(
        "out/lesson.pl.dubbed.mp4"
    )
