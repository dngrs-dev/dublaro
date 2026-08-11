import json
from pathlib import Path

from dublaro.cli.workspace import inspect_workspace


def test_inspect_workspace_reports_workspace_and_manifest_artifacts(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / ".dublaro" / "zoo"
    output_dir = tmp_path / "output"
    speech_dir = workspace / "zoo.pl.speech"
    timing_repaired_speech_dir = workspace / "zoo.pl.timing-repaired-speech"
    timing_repaired_transcript_path = workspace / "zoo.pl.timing-repaired.json"
    output_video_path = output_dir / "zoo.pl.dubbed.mp4"
    missing_srt_path = output_dir / "zoo.pl.dubbed.srt"

    workspace.mkdir(parents=True)
    output_dir.mkdir()
    speech_dir.mkdir()
    timing_repaired_speech_dir.mkdir()

    (workspace / "zoo.audio.wav").write_bytes(b"audio")
    (workspace / "zoo.en.json").write_text("{}", encoding="utf-8")
    (workspace / "zoo.pl.synthesized.json").write_text("{}", encoding="utf-8")
    (speech_dir / "seg-0001.wav").write_bytes(b"clip")
    timing_repaired_transcript_path.write_text("{}", encoding="utf-8")
    (timing_repaired_speech_dir / "seg-0001.repair-1.wav").write_bytes(b"clip")
    output_video_path.write_bytes(b"video")

    manifest_path = workspace / "zoo.pl.manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "artifacts": {
                    "workspace_dir": str(workspace),
                    "extracted_audio_path": str(workspace / "zoo.audio.wav"),
                    "source_transcript_path": str(workspace / "zoo.en.json"),
                    "synthesized_transcript_path": str(
                        workspace / "zoo.pl.synthesized.json"
                    ),
                    "timing_repaired_transcript_path": str(
                        timing_repaired_transcript_path
                    ),
                    "timing_repaired_speech_dir": str(timing_repaired_speech_dir),
                    "speech_dir": str(speech_dir),
                    "dubbed_video_path": str(output_video_path),
                    "srt_path": str(missing_srt_path),
                    "manifest_path": str(manifest_path),
                }
            }
        ),
        encoding="utf-8",
    )

    report = inspect_workspace(workspace)

    assert report.present_count == 8
    assert report.missing_count == 1
    assert report.manifest_paths == [manifest_path]

    assert any(
        artifact.label == "extracted audio"
        and artifact.path == workspace / "zoo.audio.wav"
        and artifact.status == "present"
        for artifact in report.artifacts
    )
    assert any(
        artifact.label == "timing-repaired transcript"
        and artifact.path == timing_repaired_transcript_path
        and artifact.status == "present"
        for artifact in report.artifacts
    )
    assert any(
        artifact.label == "timing-repaired speech clips"
        and artifact.path == timing_repaired_speech_dir
        and artifact.item_count == 1
        for artifact in report.artifacts
    )
    assert any(
        artifact.label == "speech clips"
        and artifact.path == speech_dir
        and artifact.item_count == 1
        for artifact in report.artifacts
    )
    assert any(
        artifact.label == "dubbed video"
        and artifact.path == output_video_path
        and artifact.source == "manifest"
        and artifact.status == "present"
        for artifact in report.artifacts
    )
    assert any(
        artifact.label == "SRT subtitles"
        and artifact.path == missing_srt_path
        and artifact.status == "missing"
        for artifact in report.artifacts
    )


def test_inspect_workspace_rejects_missing_workspace(tmp_path: Path) -> None:
    missing_workspace = tmp_path / "missing"

    try:
        inspect_workspace(missing_workspace)
    except FileNotFoundError as error:
        assert str(missing_workspace) in str(error)
    else:
        raise AssertionError("inspect_workspace should reject missing workspaces")
