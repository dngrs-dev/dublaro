from pathlib import Path

from dublaro.pipeline.export import (
    default_dubbed_video_path,
    default_dubbed_video_path_in_dir,
    export_dubbed_video,
)


def test_default_dubbed_video_path() -> None:
    assert default_dubbed_video_path("lesson.mp4", "pl") == Path("lesson.pl.dubbed.mp4")


def test_default_dubbed_video_path_in_dir() -> None:
    assert default_dubbed_video_path_in_dir("lesson.mp4", "pl", "out") == Path(
        "out/lesson.pl.dubbed.mp4"
    )


def test_export_dubbed_video_requires_subtitle_path_when_embedding() -> None:
    try:
        export_dubbed_video(
            "video.mp4",
            "speech.wav",
            "out.mp4",
            subtitle_embed="soft",
        )
    except ValueError as error:
        assert "subtitle_path" in str(error)
    else:
        raise AssertionError("Expected ValueError")
