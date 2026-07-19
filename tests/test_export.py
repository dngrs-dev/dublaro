from pathlib import Path

from dublaro.pipeline.export import default_dubbed_video_path


def test_default_dubbed_video_path() -> None:
    assert default_dubbed_video_path("lesson.mp4", "pl") == Path("lesson.pl.dubbed.mp4")
