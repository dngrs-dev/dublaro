import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

WorkspaceArtifactStatus = Literal["present", "missing"]
WorkspaceArtifactSource = Literal["workspace", "manifest"]


@dataclass(frozen=True)
class WorkspaceArtifact:
    category: str
    label: str
    path: Path
    status: WorkspaceArtifactStatus
    source: WorkspaceArtifactSource
    size_bytes: int | None = None
    item_count: int | None = None
    details: str | None = None


@dataclass(frozen=True)
class WorkspaceInspectionReport:
    workspace_dir: Path
    artifacts: list[WorkspaceArtifact]
    manifest_paths: list[Path]

    @property
    def present_count(self) -> int:
        return sum(artifact.status == "present" for artifact in self.artifacts)

    @property
    def missing_count(self) -> int:
        return sum(artifact.status == "missing" for artifact in self.artifacts)


_MANIFEST_ARTIFACT_LABELS = {
    "extracted_audio_path": ("audio", "extracted audio"),
    "source_transcript_path": ("transcript", "source transcript"),
    "diarized_transcript_path": ("transcript", "diarized source transcript"),
    "translated_transcript_path": ("transcript", "translated transcript"),
    "adapted_transcript_path": ("transcript", "adapted transcript"),
    "synthesized_transcript_path": ("transcript", "synthesized transcript"),
    "timing_repaired_transcript_path": ("transcript", "timing-repaired transcript"),
    "timing_repaired_speech_dir": ("speech", "timing-repaired speech clips"),
    "speech_dir": ("speech", "speech clips"),
    "speech_track_path": ("audio", "speech track"),
    "dubbed_video_path": ("video", "dubbed video"),
    "fitted_transcript_path": ("transcript", "fitted transcript"),
    "fitted_speech_dir": ("speech", "fitted speech clips"),
    "video_fitted_transcript_path": ("transcript", "video-fitted transcript"),
    "fitted_video_path": ("video", "video-fitted video"),
    "video_fitted_original_audio_path": ("audio", "video-fitted original audio"),
    "mix_original_audio_path": ("audio", "original audio for mixing"),
    "mixed_audio_path": ("audio", "mixed audio"),
    "srt_path": ("subtitle", "SRT subtitles"),
    "embedded_srt_path": ("subtitle", "embedded subtitle source"),
    "manifest_path": ("manifest", "manifest"),
}

_CATEGORY_ORDER = {
    "audio": 10,
    "transcript": 20,
    "speech": 30,
    "video": 40,
    "subtitle": 50,
    "manifest": 60,
    "directory": 70,
    "other": 90,
}


def inspect_workspace(
    workspace_dir: str | Path,
    *,
    include_manifest: bool = True,
    include_unknown: bool = False,
) -> WorkspaceInspectionReport:
    workspace = Path(workspace_dir)

    if not workspace.exists():
        raise FileNotFoundError(f"Workspace does not exist: {workspace}")

    if not workspace.is_dir():
        raise ValueError(f"Workspace is not a directory: {workspace}")

    artifacts_by_path: dict[str, WorkspaceArtifact] = {}

    for path in sorted(workspace.iterdir(), key=lambda item: item.name.lower()):
        artifact = _workspace_artifact(path)
        if artifact is None and include_unknown:
            artifact = _build_artifact(
                category="directory" if path.is_dir() else "other",
                label="directory" if path.is_dir() else "other file",
                path=path,
                source="workspace",
            )

        if artifact is not None:
            _add_artifact(artifacts_by_path, artifact)

    manifest_paths = sorted(workspace.glob("*.manifest.json"))

    if include_manifest:
        for manifest_path in manifest_paths:
            for artifact in _manifest_artifacts(manifest_path):
                _add_artifact(artifacts_by_path, artifact)

    artifacts = sorted(
        artifacts_by_path.values(),
        key=lambda artifact: (
            _CATEGORY_ORDER.get(artifact.category, 90),
            artifact.label,
            str(artifact.path).lower(),
        ),
    )

    return WorkspaceInspectionReport(
        workspace_dir=workspace,
        artifacts=artifacts,
        manifest_paths=manifest_paths,
    )


def _workspace_artifact(path: Path) -> WorkspaceArtifact | None:
    name = path.name

    if path.is_dir():
        if name.endswith(".timing-repaired-speech"):
            return _build_artifact(
                category="speech",
                label="timing-repaired speech clips",
                path=path,
                source="workspace",
            )

        if name.endswith(".fitted-speech"):
            return _build_artifact(
                category="speech",
                label="fitted speech clips",
                path=path,
                source="workspace",
            )

        if name.endswith(".speech"):
            return _build_artifact(
                category="speech",
                label="speech clips",
                path=path,
                source="workspace",
            )

        return None

    if name.endswith(".audio.wav"):
        return _build_artifact("audio", "extracted audio", path, "workspace")

    if name.endswith(".original-video-fitted.wav"):
        return _build_artifact(
            "audio",
            "video-fitted original audio",
            path,
            "workspace",
        )

    if name.endswith(".original-mix.wav"):
        return _build_artifact("audio", "original audio for mixing", path, "workspace")

    if name.endswith(".mixed.wav"):
        return _build_artifact("audio", "mixed audio", path, "workspace")

    if name.endswith(".speech-track.wav"):
        return _build_artifact("audio", "speech track", path, "workspace")

    if name.endswith(".diarized.json"):
        return _build_artifact(
            "transcript",
            "diarized source transcript",
            path,
            "workspace",
        )

    if name.endswith(".adapted.json"):
        return _build_artifact("transcript", "adapted transcript", path, "workspace")

    if name.endswith(".synthesized.json"):
        return _build_artifact(
            "transcript",
            "synthesized transcript",
            path,
            "workspace",
        )

    if name.endswith(".timing-repaired.json"):
        return _build_artifact(
            "transcript",
            "timing-repaired transcript",
            path,
            "workspace",
        )

    if name.endswith(".fitted.json"):
        return _build_artifact("transcript", "fitted transcript", path, "workspace")

    if name.endswith(".video-fitted.json"):
        return _build_artifact(
            "transcript",
            "video-fitted transcript",
            path,
            "workspace",
        )

    if name.endswith(".manifest.json"):
        return _build_artifact("manifest", "manifest", path, "workspace")

    if name.endswith(".embed.srt"):
        return _build_artifact(
            "subtitle",
            "embedded subtitle source",
            path,
            "workspace",
        )

    if name.endswith(".srt"):
        return _build_artifact("subtitle", "SRT subtitles", path, "workspace")

    if name.endswith(".json"):
        return _build_artifact(
            "transcript",
            "source or translated transcript",
            path,
            "workspace",
        )

    if ".video-fitted" in name and path.suffix.lower() in {
        ".mp4",
        ".mov",
        ".mkv",
        ".webm",
        ".avi",
        ".m4v",
    }:
        return _build_artifact("video", "video-fitted video", path, "workspace")

    return None


def _manifest_artifacts(manifest_path: Path) -> list[WorkspaceArtifact]:
    try:
        data: Any = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []

    if not isinstance(data, dict):
        return []

    artifacts = data.get("artifacts")
    if not isinstance(artifacts, dict):
        return []

    manifest_artifacts: list[WorkspaceArtifact] = []

    for key, value in artifacts.items():
        if key == "workspace_dir" or value is None:
            continue

        if not isinstance(value, str):
            continue

        category, label = _MANIFEST_ARTIFACT_LABELS.get(
            key,
            ("other", key.replace("_", " ")),
        )
        manifest_artifacts.append(
            _build_artifact(
                category=category,
                label=label,
                path=Path(value),
                source="manifest",
            )
        )

    return manifest_artifacts


def _build_artifact(
    category: str,
    label: str,
    path: Path,
    source: WorkspaceArtifactSource,
) -> WorkspaceArtifact:
    if not path.exists():
        return WorkspaceArtifact(
            category=category,
            label=label,
            path=path,
            status="missing",
            source=source,
        )

    if path.is_dir():
        item_count = _count_directory_items(
            path, "*.wav" if category == "speech" else None
        )
        details = (
            f"{item_count} WAV file(s)"
            if category == "speech"
            else f"{item_count} item(s)"
        )

        return WorkspaceArtifact(
            category=category,
            label=label,
            path=path,
            status="present",
            source=source,
            item_count=item_count,
            details=details,
        )

    return WorkspaceArtifact(
        category=category,
        label=label,
        path=path,
        status="present",
        source=source,
        size_bytes=path.stat().st_size,
        details=_format_size(path.stat().st_size),
    )


def _count_directory_items(path: Path, pattern: str | None) -> int:
    if pattern is None:
        return sum(1 for _ in path.iterdir())

    return sum(1 for _ in path.glob(pattern))


def _format_size(size_bytes: int) -> str:
    if size_bytes < 1024:
        return f"{size_bytes} B"

    size_kib = size_bytes / 1024
    if size_kib < 1024:
        return f"{size_kib:.1f} KiB"

    return f"{size_kib / 1024:.1f} MiB"


def _add_artifact(
    artifacts_by_path: dict[str, WorkspaceArtifact],
    artifact: WorkspaceArtifact,
) -> None:
    key = str(artifact.path.resolve(strict=False)).casefold()
    existing = artifacts_by_path.get(key)

    if existing is None:
        artifacts_by_path[key] = artifact
        return

    if existing.source == "manifest" and artifact.source == "workspace":
        artifacts_by_path[key] = artifact
