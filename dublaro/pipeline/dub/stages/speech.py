from pathlib import Path

from dublaro.audio.ffmpeg import slow_video
from dublaro.pipeline.align import build_speech_timeline
from dublaro.pipeline.dub.context import DubRunContext
from dublaro.pipeline.dub.progress import (
    progress_skipped as _progress_skipped,
)
from dublaro.pipeline.dub.progress import (
    progress_stage as _progress_stage,
)
from dublaro.pipeline.dub.results import (
    SpeechTimingResult,
    TimingRepairResult,
    VideoFitResult,
)
from dublaro.pipeline.fit_speech import fit_generated_speech_to_segments
from dublaro.pipeline.fit_video import plan_video_slowdown, scale_transcript_timing
from dublaro.pipeline.resume import load_reusable_synthesized_transcript, reusable_file
from dublaro.pipeline.synthesize import synthesize_transcript_speech
from dublaro.pipeline.timing_repair import repair_overlong_speech_segments
from dublaro.pipeline.transcribe import save_transcript
from dublaro.schemas import Transcript


def _synthesize_speech(
    context: DubRunContext,
    adapted_transcript: Transcript,
) -> Transcript:
    synthesized_transcript_path = context.artifact_paths.synthesized_transcript_path

    synthesized_transcript = (
        load_reusable_synthesized_transcript(synthesized_transcript_path)
        if context.options.resume
        else None
    )

    if synthesized_transcript is not None:
        _progress_skipped(
            context.progress_callback,
            "synthesize",
            f"Using existing synthesized speech: {synthesized_transcript_path}.",
        )
        return synthesized_transcript

    with _progress_stage(
        context.progress_callback,
        "synthesize",
        "Synthesizing speech clips.",
    ):
        synthesized_transcript = synthesize_transcript_speech(
            adapted_transcript,
            adapter=context.adapters.tts,
            output_dir=context.artifact_paths.speech_dir,
            language=context.options.target_language,
            sample_rate=context.options.speech_sample_rate,
            speaker_voices=context.adapters.speaker_voices,
        )
        save_transcript(synthesized_transcript, synthesized_transcript_path)
        return synthesized_transcript


def _repair_speech_timing(
    context: DubRunContext,
    synthesized_transcript: Transcript,
) -> TimingRepairResult:
    if not context.options.repair_timing:
        return TimingRepairResult(transcript=synthesized_transcript)

    timing_repaired_transcript_path = (
        context.artifact_paths.timing_repaired_transcript_path
    )
    timing_repaired_speech_dir = context.artifact_paths.timing_repaired_speech_dir

    reusable_repaired_transcript = (
        load_reusable_synthesized_transcript(timing_repaired_transcript_path)
        if context.options.resume
        else None
    )

    if reusable_repaired_transcript is not None:
        _progress_skipped(
            context.progress_callback,
            "repair_timing",
            f"Using existing timing-repaired transcript: "
            f"{timing_repaired_transcript_path}.",
        )
        return TimingRepairResult(
            transcript=reusable_repaired_transcript,
            timing_repaired_transcript_path=timing_repaired_transcript_path,
            timing_repaired_speech_dir=timing_repaired_speech_dir,
        )

    with _progress_stage(
        context.progress_callback,
        "repair_timing",
        "Repairing overlong speech text.",
    ):
        repaired_transcript = repair_overlong_speech_segments(
            synthesized_transcript,
            text_adapter=context.adapters.text_adapter,
            tts_adapter=context.adapters.tts,
            output_dir=timing_repaired_speech_dir,
            language=context.options.target_language,
            sample_rate=context.options.speech_sample_rate,
            speaker_voices=context.adapters.speaker_voices,
            max_attempts=context.options.max_timing_repair_attempts,
            target_speedup=context.options.timing_repair_target_speedup,
            min_overrun_seconds=context.options.min_speech_overrun_seconds,
            overwrite=context.options.overwrite,
        )
        save_transcript(repaired_transcript, timing_repaired_transcript_path)

        return TimingRepairResult(
            transcript=repaired_transcript,
            timing_repaired_transcript_path=timing_repaired_transcript_path,
            timing_repaired_speech_dir=timing_repaired_speech_dir,
        )


def _fit_speech_to_timing(
    context: DubRunContext,
    synthesized_transcript: Transcript,
) -> SpeechTimingResult:
    if not context.options.fit_speech:
        return SpeechTimingResult(transcript=synthesized_transcript)

    fitted_transcript_path = context.artifact_paths.fitted_transcript_path
    fitted_speech_dir = context.artifact_paths.fitted_speech_dir

    reusable_fitted_transcript = (
        load_reusable_synthesized_transcript(fitted_transcript_path)
        if context.options.resume
        else None
    )

    if reusable_fitted_transcript is not None:
        _progress_skipped(
            context.progress_callback,
            "fit_speech",
            f"Using existing fitted transcript: {fitted_transcript_path}.",
        )
        return SpeechTimingResult(
            transcript=reusable_fitted_transcript,
            fitted_transcript_path=fitted_transcript_path,
            fitted_speech_dir=fitted_speech_dir,
        )

    with _progress_stage(
        context.progress_callback,
        "fit_speech",
        "Fitting overlong speech clips to segment timing.",
    ):
        fitted_transcript = fit_generated_speech_to_segments(
            synthesized_transcript,
            output_dir=fitted_speech_dir,
            max_speedup=context.options.max_speech_speedup,
            min_overrun_seconds=context.options.min_speech_overrun_seconds,
            allow_unfit_overruns=context.options.fit_video,
            overwrite=context.options.overwrite,
            executable=context.options.ffmpeg_executable,
        )
        save_transcript(fitted_transcript, fitted_transcript_path)

        return SpeechTimingResult(
            transcript=fitted_transcript,
            fitted_transcript_path=fitted_transcript_path,
            fitted_speech_dir=fitted_speech_dir,
        )


def _fit_video_to_speech(
    context: DubRunContext,
    transcript: Transcript,
) -> VideoFitResult:
    if not context.options.fit_video:
        return VideoFitResult(
            transcript=transcript,
            video_path=context.paths.video_path,
        )

    plan = plan_video_slowdown(
        transcript,
        max_slowdown=context.options.max_video_slowdown,
        min_overrun_seconds=context.options.min_speech_overrun_seconds,
    )

    if plan.slowdown_factor <= 1.0:
        _progress_skipped(
            context.progress_callback,
            "fit_video",
            "Video slowdown not needed.",
        )
        return VideoFitResult(
            transcript=transcript,
            video_path=context.paths.video_path,
        )

    fitted_video_path = context.artifact_paths.fitted_video_path
    video_fitted_transcript_path = context.artifact_paths.video_fitted_transcript_path

    reusable_transcript = (
        load_reusable_synthesized_transcript(video_fitted_transcript_path)
        if context.options.resume and reusable_file(fitted_video_path)
        else None
    )

    if reusable_transcript is not None:
        _progress_skipped(
            context.progress_callback,
            "fit_video",
            f"Using existing video-fitted artifacts: {fitted_video_path}.",
        )
        return VideoFitResult(
            transcript=reusable_transcript,
            video_path=fitted_video_path,
            slowdown_factor=plan.slowdown_factor,
            video_fitted_transcript_path=video_fitted_transcript_path,
            fitted_video_path=fitted_video_path,
        )

    with _progress_stage(
        context.progress_callback,
        "fit_video",
        f"Slowing video by {plan.slowdown_factor:.2f}x.",
    ):
        slow_video(
            context.paths.video_path,
            fitted_video_path,
            slowdown_factor=plan.slowdown_factor,
            overwrite=context.options.overwrite,
            executable=context.options.ffmpeg_executable,
        )

        video_fitted_transcript = scale_transcript_timing(
            transcript,
            slowdown_factor=plan.slowdown_factor,
            min_overrun_seconds=context.options.min_speech_overrun_seconds,
        )
        video_fitted_transcript.metadata = {
            **video_fitted_transcript.metadata,
            "video_fitting_overlong_segments": str(plan.overlong_segment_count),
            "video_fitting_limiting_segment_id": plan.limiting_segment_id or "",
        }
        save_transcript(video_fitted_transcript, video_fitted_transcript_path)

        return VideoFitResult(
            transcript=video_fitted_transcript,
            video_path=fitted_video_path,
            slowdown_factor=plan.slowdown_factor,
            video_fitted_transcript_path=video_fitted_transcript_path,
            fitted_video_path=fitted_video_path,
        )


def _align_speech_track(
    context: DubRunContext,
    speech_timeline_transcript: Transcript,
) -> Path:
    speech_track_path = context.artifact_paths.speech_track_path

    if context.options.resume and reusable_file(speech_track_path):
        _progress_skipped(
            context.progress_callback,
            "align_speech",
            f"Using existing speech track: {speech_track_path}.",
        )
        return speech_track_path

    with _progress_stage(
        context.progress_callback,
        "align_speech",
        "Building timed speech track.",
    ):
        return build_speech_timeline(
            speech_timeline_transcript,
            output_path=speech_track_path,
            sample_rate=context.options.speech_sample_rate,
        )
