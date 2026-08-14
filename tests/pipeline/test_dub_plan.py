from pathlib import Path

import pytest
from dublaro.pipeline.dub_plan import DubOptions, DubPaths


def test_dub_paths_builds_default_artifact_paths(tmp_path: Path) -> None:
    paths = DubPaths.build(
        video_path=tmp_path / "lesson.mp4",
        output_path=tmp_path / "lesson.pl.dubbed.mp4",
        workspace_dir=tmp_path / "workspace",
    )

    artifacts = paths.artifacts(DubOptions(source_language=None, target_language="pl"))

    assert artifacts.extracted_audio_path == tmp_path / "workspace" / "lesson.audio.wav"
    assert (
        artifacts.source_transcript_path == tmp_path / "workspace" / "lesson.auto.json"
    )
    assert (
        artifacts.translated_transcript_path
        == tmp_path / "workspace" / "lesson.pl.json"
    )
    assert artifacts.adapted_transcript_path == (
        tmp_path / "workspace" / "lesson.pl.adapted.json"
    )
    assert artifacts.synthesized_transcript_path == (
        tmp_path / "workspace" / "lesson.pl.synthesized.json"
    )
    assert artifacts.speech_dir == tmp_path / "workspace" / "lesson.pl.speech"
    assert artifacts.speech_track_path == (
        tmp_path / "workspace" / "lesson.pl.speech-track.wav"
    )
    assert artifacts.srt_path == tmp_path / "lesson.pl.dubbed.srt"
    assert artifacts.manifest_path == (
        tmp_path / "workspace" / "lesson.pl.manifest.json"
    )


def test_dub_paths_uses_explicit_source_language(tmp_path: Path) -> None:
    paths = DubPaths.build(
        video_path=tmp_path / "lesson.mp4",
        output_path=tmp_path / "lesson.pl.dubbed.mp4",
        workspace_dir=tmp_path / "workspace",
    )

    artifacts = paths.artifacts(DubOptions(source_language="en", target_language="pl"))

    assert artifacts.source_transcript_path == tmp_path / "workspace" / "lesson.en.json"


def test_dub_paths_uses_custom_srt_and_manifest_paths(tmp_path: Path) -> None:
    custom_srt = tmp_path / "subs" / "lesson.srt"
    custom_manifest = tmp_path / "runs" / "manifest.json"

    paths = DubPaths.build(
        video_path=tmp_path / "lesson.mp4",
        output_path=tmp_path / "lesson.pl.dubbed.mp4",
        workspace_dir=tmp_path / "workspace",
    )

    artifacts = paths.artifacts(
        DubOptions(
            source_language="en",
            target_language="pl",
            srt_output_path=custom_srt,
            manifest_output_path=custom_manifest,
        )
    )

    assert artifacts.srt_path == custom_srt
    assert artifacts.manifest_path == custom_manifest


def test_dub_options_rejects_manifest_path_when_manifest_is_disabled(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="manifest_output_path"):
        DubOptions(
            source_language="en",
            target_language="pl",
            write_manifest=False,
            manifest_output_path=tmp_path / "manifest.json",
        )


def test_dub_options_rejects_resume_with_overwrite() -> None:
    with pytest.raises(ValueError, match="resume"):
        DubOptions(
            source_language="en",
            target_language="pl",
            resume=True,
            overwrite=True,
        )
