from pathlib import Path

DEFAULT_VIDEO_EXTENSIONS = (".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v")


def discover_batch_videos(
    input_path: str | Path,
    *,
    recursive: bool = False,
    extensions: tuple[str, ...] = DEFAULT_VIDEO_EXTENSIONS,
) -> list[Path]:
    path = Path(input_path)
    normalized_extensions = _normalize_extensions(extensions)

    if path.is_file():
        return [path] if path.suffix.lower() in normalized_extensions else []

    if not path.is_dir():
        raise ValueError(f"Batch input is not a file or directory: {path}")

    candidates = path.rglob("*") if recursive else path.iterdir()

    return sorted(
        (
            candidate
            for candidate in candidates
            if candidate.is_file() and candidate.suffix.lower() in normalized_extensions
        ),
        key=lambda candidate: str(candidate).lower(),
    )


def default_batch_workspace_dir(
    input_path: str | Path,
    video_path: str | Path,
    workspace_root: str | Path,
) -> Path:
    relative_stem = _relative_video_stem(input_path, video_path)
    return Path(workspace_root).joinpath(
        *(_safe_path_part(part) for part in relative_stem.parts)
    )


def default_batch_output_dir(
    input_path: str | Path,
    video_path: str | Path,
    output_root: str | Path,
) -> Path:
    relative_parent = _relative_video_stem(input_path, video_path).parent
    return Path(output_root).joinpath(
        *(_safe_path_part(part) for part in relative_parent.parts if part != ".")
    )


def format_video_extensions(
    extensions: tuple[str, ...] = DEFAULT_VIDEO_EXTENSIONS,
) -> str:
    return ", ".join(extensions)


def _normalize_extensions(extensions: tuple[str, ...]) -> set[str]:
    return {
        extension.lower() if extension.startswith(".") else f".{extension.lower()}"
        for extension in extensions
    }


def _relative_video_stem(input_path: str | Path, video_path: str | Path) -> Path:
    input_file_or_dir = Path(input_path)
    video_file = Path(video_path)

    if input_file_or_dir.is_file():
        return Path(video_file.stem)

    try:
        return video_file.relative_to(input_file_or_dir).with_suffix("")
    except ValueError:
        return Path(video_file.stem)


def _safe_path_part(value: str) -> str:
    safe = "".join(
        character if character.isalnum() or character in {"-", "_", "."} else "_"
        for character in value
    )
    return safe or "_"
