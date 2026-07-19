from pathlib import Path

from dublaro.audio.ffmpeg import replace_video_audio


def export_dubbed_video(
    video_path: str | Path,
    speech_track_path: str | Path,
    output_path: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    return replace_video_audio(
        video_path,
        speech_track_path,
        output_path,
        overwrite=overwrite,
    )


def default_dubbed_video_path(video_path: str | Path, target_language: str) -> Path:
    video_file = Path(video_path)
    return video_file.with_name(
        f"{video_file.stem}.{target_language}.dubbed{video_file.suffix}"
    )
