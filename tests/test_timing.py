from array import array
from pathlib import Path

from dublaro.audio.wav import write_mono_pcm16_wav
from dublaro.pipeline.timing import analyze_speech_timing, preview_speech_timing
from dublaro.schemas import Segment, Transcript


def test_analyze_speech_timing_reports_overlong_clip(tmp_path: Path) -> None:
    clip_path = tmp_path / "seg-0001.wav"

    write_mono_pcm16_wav(
        clip_path,
        samples=array("h", [0] * 12),
        sample_rate=10,
    )

    transcript = Transcript(
        id="lesson-1",
        source_language="en",
        target_language="pl",
        segments=[
            Segment(
                id="seg-0001",
                start=0.0,
                end=1.0,
                generated_audio_path=str(clip_path),
            )
        ],
    )

    issues = analyze_speech_timing(
        transcript,
        max_overrun_seconds=0.05,
        max_ratio=1.05,
    )

    assert len(issues) == 1
    assert issues[0].segment_id == "seg-0001"
    assert issues[0].target_duration == 1.0
    assert issues[0].audio_duration == 1.2


def test_analyze_speech_timing_accepts_clip_inside_tolerance(tmp_path: Path) -> None:
    clip_path = tmp_path / "seg-0001.wav"

    write_mono_pcm16_wav(
        clip_path,
        samples=array("h", [0] * 10),
        sample_rate=10,
    )

    transcript = Transcript(
        id="lesson-1",
        source_language="en",
        target_language="pl",
        segments=[
            Segment(
                id="seg-0001",
                start=0.0,
                end=1.0,
                generated_audio_path=str(clip_path),
            )
        ],
    )

    issues = analyze_speech_timing(transcript)

    assert issues == []


def test_analyze_speech_timing_skips_segments_without_audio() -> None:
    transcript = Transcript(
        id="lesson-1",
        source_language="en",
        target_language="pl",
        segments=[
            Segment(
                id="seg-0001",
                start=0.0,
                end=1.0,
            )
        ],
    )

    assert analyze_speech_timing(transcript) == []


def test_preview_speech_timing_classifies_segments(tmp_path: Path) -> None:
    ok_clip = tmp_path / "ok.wav"
    speedup_clip = tmp_path / "speedup.wav"
    video_clip = tmp_path / "video.wav"

    write_mono_pcm16_wav(ok_clip, samples=array("h", [0] * 10), sample_rate=10)
    write_mono_pcm16_wav(speedup_clip, samples=array("h", [0] * 12), sample_rate=10)
    write_mono_pcm16_wav(video_clip, samples=array("h", [0] * 20), sample_rate=10)

    transcript = Transcript(
        id="lesson",
        source_language="en",
        target_language="pl",
        segments=[
            Segment(id="seg-ok", start=0.0, end=1.0, generated_audio_path=str(ok_clip)),
            Segment(
                id="seg-speedup",
                start=1.0,
                end=2.0,
                generated_audio_path=str(speedup_clip),
            ),
            Segment(
                id="seg-video",
                start=2.0,
                end=3.0,
                generated_audio_path=str(video_clip),
            ),
            Segment(id="seg-missing", start=3.0, end=4.0),
        ],
    )

    previews = preview_speech_timing(
        transcript,
        max_speedup=1.35,
        min_overrun_seconds=0.05,
    )

    assert [preview.status for preview in previews] == [
        "ok",
        "speed-up",
        "needs-video",
        "missing-audio",
    ]
    assert previews[1].required_speedup == 1.2
    assert previews[2].required_speedup == 2.0
    assert previews[2].applied_speedup == 1.35
    assert previews[2].needs_video_fit is True
