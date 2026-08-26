import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dublaro.cli.reports.preview import (
    TimingPreviewReport,
    TimingRepairPreviewReport,
    build_timing_preview_report,
    build_timing_repair_preview_report,
)
from dublaro.cli.reports.workspace import (
    WorkspaceInspectionReport,
    inspect_workspace,
)

_TIMING_ARTIFACT_KEYS = (
    "video_fitted_transcript_path",
    "fitted_transcript_path",
    "timing_repaired_transcript_path",
    "synthesized_transcript_path",
)

_TIMING_WORKSPACE_PATTERNS = (
    "*.video-fitted.json",
    "*.fitted.json",
    "*.timing-repaired.json",
    "*.synthesized.json",
)


@dataclass(frozen=True)
class QualityManifestSummary:
    path: Path
    input_video_path: str | None
    output_video_path: str | None
    source_language: str | None
    target_language: str | None
    adapters: dict[str, str]
    speaker_voice_count: int
    options: dict[str, str]
    artifacts: dict[str, str]
    metadata: dict[str, str]


@dataclass(frozen=True)
class DubQualityReport:
    workspace: WorkspaceInspectionReport
    manifest: QualityManifestSummary | None
    timing_transcript_path: Path | None
    timing: TimingPreviewReport | None
    repair_transcript_path: Path | None
    repairs: TimingRepairPreviewReport | None
    max_speedup: float
    min_overrun_seconds: float
    warnings: list[str]
    errors: list[str]

    @property
    def has_issues(self) -> bool:
        return (
            bool(self.errors)
            or self.workspace.missing_count > 0
            or (self.timing is not None and self.timing.attention_count > 0)
            or (self.repairs is not None and self.repairs.not_improved_count > 0)
        )


def build_dub_quality_report(
    workspace_dir: str | Path,
    *,
    manifest_path: str | Path | None = None,
    max_speedup: float | None = None,
    min_overrun_seconds: float | None = None,
) -> DubQualityReport:
    workspace_path = Path(workspace_dir)
    workspace = inspect_workspace(
        workspace_path,
        include_manifest=True,
        include_unknown=False,
    )

    warnings: list[str] = []
    errors: list[str] = []

    resolved_manifest_path = (
        Path(manifest_path)
        if manifest_path is not None
        else _latest_manifest_path(workspace.manifest_paths)
    )

    manifest = None
    if resolved_manifest_path is None:
        warnings.append("No manifest file found in workspace.")
    else:
        try:
            manifest = _load_manifest_summary(resolved_manifest_path)
        except ValueError as error:
            errors.append(str(error))

    effective_max_speedup = (
        max_speedup or _manifest_float_option(manifest, "max_speech_speedup") or 1.35
    )
    effective_min_overrun = (
        min_overrun_seconds
        or _manifest_float_option(manifest, "min_speech_overrun_seconds")
        or 0.05
    )

    timing_transcript_path = _choose_timing_transcript_path(workspace_path, manifest)
    timing = None
    if timing_transcript_path is None:
        warnings.append("No synthesized/fitted transcript found for timing analysis.")
    else:
        try:
            timing = build_timing_preview_report(
                timing_transcript_path,
                max_speedup=effective_max_speedup,
                min_overrun_seconds=effective_min_overrun,
                only_issues=False,
            )
        except (OSError, ValueError, EOFError) as error:
            errors.append(f"Timing analysis failed: {error}")

    repair_transcript_path = _choose_repair_transcript_path(workspace_path, manifest)
    repairs = None
    if repair_transcript_path is not None:
        try:
            repairs = build_timing_repair_preview_report(
                repair_transcript_path,
                include_all=True,
            )
        except (OSError, ValueError) as error:
            errors.append(f"Timing repair analysis failed: {error}")

    return DubQualityReport(
        workspace=workspace,
        manifest=manifest,
        timing_transcript_path=timing_transcript_path,
        timing=timing,
        repair_transcript_path=repair_transcript_path,
        repairs=repairs,
        max_speedup=effective_max_speedup,
        min_overrun_seconds=effective_min_overrun,
        warnings=warnings,
        errors=errors,
    )


def _latest_manifest_path(paths: list[Path]) -> Path | None:
    if not paths:
        return None

    return max(paths, key=lambda path: path.stat().st_mtime)


def _load_manifest_summary(manifest_path: Path) -> QualityManifestSummary:
    try:
        data: Any = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Manifest is not valid JSON: {manifest_path}") from error
    except OSError as error:
        raise ValueError(f"Manifest could not be read: {manifest_path}") from error

    if not isinstance(data, dict):
        raise ValueError(f"Manifest root must be an object: {manifest_path}")

    language = _as_mapping(data.get("language"))
    adapters = _as_mapping(data.get("adapters"))
    speaker_voices = _as_mapping(adapters.get("speaker_voices"))

    return QualityManifestSummary(
        path=manifest_path,
        input_video_path=_optional_string(data.get("input_video_path")),
        output_video_path=_optional_string(data.get("output_video_path")),
        source_language=_optional_string(language.get("source")),
        target_language=_optional_string(language.get("target")),
        adapters=_adapter_names(adapters),
        speaker_voice_count=len(speaker_voices),
        options=_string_map(_as_mapping(data.get("options"))),
        artifacts=_string_map(_as_mapping(data.get("artifacts"))),
        metadata=_string_map(_as_mapping(data.get("metadata"))),
    )


def _adapter_names(adapters: dict[str, Any]) -> dict[str, str]:
    names: dict[str, str] = {}

    for key, value in adapters.items():
        if key == "speaker_voices":
            continue

        adapter_data = _as_mapping(value)
        name = _optional_string(adapter_data.get("name"))
        if name:
            names[str(key)] = name

    return names


def _choose_timing_transcript_path(
    workspace_dir: Path,
    manifest: QualityManifestSummary | None,
) -> Path | None:
    manifest_path = _first_manifest_artifact(manifest, _TIMING_ARTIFACT_KEYS)
    if manifest_path is not None:
        return manifest_path

    return _first_workspace_match(workspace_dir, _TIMING_WORKSPACE_PATTERNS)


def _choose_repair_transcript_path(
    workspace_dir: Path,
    manifest: QualityManifestSummary | None,
) -> Path | None:
    manifest_path = _first_manifest_artifact(
        manifest,
        ("timing_repaired_transcript_path",),
    )
    if manifest_path is not None:
        return manifest_path

    return _first_workspace_match(workspace_dir, ("*.timing-repaired.json",))


def _first_manifest_artifact(
    manifest: QualityManifestSummary | None,
    keys: tuple[str, ...],
) -> Path | None:
    if manifest is None:
        return None

    for key in keys:
        value = manifest.artifacts.get(key)
        if value:
            return Path(value)

    return None


def _first_workspace_match(
    workspace_dir: Path,
    patterns: tuple[str, ...],
) -> Path | None:
    for pattern in patterns:
        matches = sorted(workspace_dir.glob(pattern))
        if matches:
            return matches[-1]

    return None


def _manifest_float_option(
    manifest: QualityManifestSummary | None,
    name: str,
) -> float | None:
    if manifest is None:
        return None

    value = manifest.options.get(name)
    if value is None:
        return None

    try:
        return float(value)
    except ValueError:
        return None


def _as_mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _string_map(mapping: dict[str, Any]) -> dict[str, str]:
    result: dict[str, str] = {}

    for key, value in mapping.items():
        if isinstance(value, str | int | float | bool):
            result[str(key)] = str(value)

    return result


def _optional_string(value: Any) -> str | None:
    return value if isinstance(value, str) else None
