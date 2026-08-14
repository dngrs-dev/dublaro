from array import array

import pytest
from dublaro.audio.wav import write_mono_pcm16_wav
from dublaro.pipeline.fit_video import plan_video_slowdown, scale_transcript_timing
from dublaro.schemas import Segment, Transcript


def test_plan_video_slowdown_uses_largest_overrun(tmp_path):
    first_clip = tmp_path / "seg-0001.wav"
    second_clip = tmp_path / "seg-0002.wav"

    write_mono_pcm16_wav(first_clip, samples=array("h", [0] * 15), sample_rate=10)
    write_mono_pcm16_wav(second_clip, samples=array("h", [0] * 24), sample_rate=10)

    transcript = Transcript(
        id="lesson",
        source_language="en",
        target_language="pl",
        segments=[
            Segment(
                id="seg-0001", start=0.0, end=1.0, generated_audio_path=str(first_clip)
            ),
            Segment(
                id="seg-0002", start=1.0, end=3.0, generated_audio_path=str(second_clip)
            ),
        ],
    )

    plan = plan_video_slowdown(transcript, max_slowdown=1.6)

    assert plan.slowdown_factor == pytest.approx(1.5)
    assert plan.overlong_segment_count == 2
    assert plan.limiting_segment_id == "seg-0001"


def test_plan_video_slowdown_rejects_excessive_slowdown(tmp_path):
    clip = tmp_path / "seg-0001.wav"
    write_mono_pcm16_wav(clip, samples=array("h", [0] * 20), sample_rate=10)

    transcript = Transcript(
        id="lesson",
        source_language="en",
        segments=[
            Segment(id="seg-0001", start=0.0, end=1.0, generated_audio_path=str(clip)),
        ],
    )

    with pytest.raises(ValueError, match="above max_video_slowdown"):
        plan_video_slowdown(transcript, max_slowdown=1.5)


def test_scale_transcript_timing_scales_segments_and_duration():
    transcript = Transcript(
        id="lesson",
        source_language="en",
        duration=2.0,
        segments=[Segment(id="seg-0001", start=0.5, end=1.5)],
    )

    scaled = scale_transcript_timing(transcript, slowdown_factor=1.5)

    assert scaled.duration == pytest.approx(3.0)
    assert scaled.segments[0].start == pytest.approx(0.75)
    assert scaled.segments[0].end == pytest.approx(2.25)
    assert transcript.segments[0].start == pytest.approx(0.5)


def test_scale_transcript_timing_writes_video_diagnostics(tmp_path):
    clip = tmp_path / "seg-0001.wav"
    write_mono_pcm16_wav(clip, samples=array("h", [0] * 15), sample_rate=10)

    transcript = Transcript(
        id="lesson",
        source_language="en",
        duration=1.0,
        segments=[
            Segment(id="seg-0001", start=0.0, end=1.0, generated_audio_path=str(clip)),
        ],
    )

    scaled = scale_transcript_timing(transcript, slowdown_factor=1.5)

    metadata = scaled.segments[0].metadata
    assert metadata["timing_required_video_slowdown"] == "1.5"
    assert metadata["timing_applied_video_slowdown"] == "1.5"
    assert metadata["timing_video_fitted_duration_seconds"] == "1.5"
    assert metadata["timing_video_fit_status"] == "video_slowed"
