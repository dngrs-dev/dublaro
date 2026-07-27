from pathlib import Path

from dublaro.audio.ffmpeg import (
    replace_video_audio,
    replace_video_audio_with_hard_subtitles,
    replace_video_audio_with_soft_subtitles,
)
from dublaro.pipeline.subtitles import SubtitleEmbedMode


def export_dubbed_video(
    video_path: str | Path,
    speech_track_path: str | Path,
    output_path: str | Path,
    *,
    subtitle_path: str | Path | None = None,
    subtitle_embed: SubtitleEmbedMode = "none",
    subtitle_language: str | None = None,
    overwrite: bool = False,
    executable: str = "ffmpeg",
) -> Path:
    if subtitle_embed == "none":
        return replace_video_audio(
            video_path,
            speech_track_path,
            output_path,
            overwrite=overwrite,
            executable=executable,
        )

    if subtitle_path is None:
        raise ValueError("subtitle_path is required when subtitle_embed is not none.")

    if subtitle_embed == "soft":
        return replace_video_audio_with_soft_subtitles(
            video_path,
            speech_track_path,
            subtitle_path,
            output_path,
            subtitle_language=subtitle_language,
            overwrite=overwrite,
            executable=executable,
        )

    if subtitle_embed == "hard":
        return replace_video_audio_with_hard_subtitles(
            video_path,
            speech_track_path,
            subtitle_path,
            output_path,
            overwrite=overwrite,
            executable=executable,
        )

    raise ValueError("subtitle_embed must be one of: none, soft, hard.")


def default_dubbed_video_filename(video_path: str | Path, target_language: str) -> str:
    video_file = Path(video_path)
    return f"{video_file.stem}.{target_language}.dubbed{video_file.suffix}"


def default_dubbed_video_path(video_path: str | Path, target_language: str) -> Path:
    video_file = Path(video_path)
    return video_file.with_name(
        default_dubbed_video_filename(video_file, target_language)
    )


def default_dubbed_video_path_in_dir(
    video_path: str | Path,
    target_language: str,
    output_dir: str | Path,
) -> Path:
    return Path(output_dir) / default_dubbed_video_filename(video_path, target_language)
