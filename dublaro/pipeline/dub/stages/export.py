from datetime import UTC, datetime
from pathlib import Path

from dublaro.pipeline.dub.context import DubRunContext
from dublaro.pipeline.dub.progress import (
    progress_stage as _progress_stage,
)
from dublaro.pipeline.dub.results import (
    ManifestInputs,
    SubtitleExportResult,
)
from dublaro.pipeline.export import export_dubbed_video
from dublaro.pipeline.manifest import (
    DubbingArtifactsManifest,
    DubbingOptionsManifest,
    build_dubbing_manifest,
    save_manifest,
)
from dublaro.pipeline.subtitles import save_srt
from dublaro.schemas import Transcript


def _export_video(
    context: DubRunContext,
    video_for_export_path: Path,
    audio_for_export_path: Path,
    subtitle_path: Path | None,
) -> Path:
    with _progress_stage(
        context.progress_callback,
        "export_video",
        "Exporting dubbed video.",
    ):
        return export_dubbed_video(
            video_for_export_path,
            audio_for_export_path,
            context.paths.output_path,
            subtitle_path=subtitle_path,
            subtitle_embed=context.options.subtitle_embed,
            subtitle_language=context.options.target_language,
            overwrite=context.options.overwrite,
            executable=context.options.ffmpeg_executable,
        )


def _prepare_subtitles_for_export(
    context: DubRunContext,
    speech_timeline_transcript: Transcript,
) -> SubtitleExportResult:
    if not context.options.export_srt and context.options.subtitle_embed == "none":
        return SubtitleExportResult()

    with _progress_stage(
        context.progress_callback,
        "export_srt",
        "Preparing subtitles.",
    ):
        sidecar_srt_path = None
        embedded_srt_path = None

        if context.options.export_srt:
            sidecar_srt_path = save_srt(
                speech_timeline_transcript,
                context.artifact_paths.srt_path,
                text_mode=context.options.srt_text_mode,
            )

        if context.options.subtitle_embed != "none":
            embedded_srt_path = sidecar_srt_path or save_srt(
                speech_timeline_transcript,
                context.artifact_paths.subtitle_embed_srt_path,
                text_mode=context.options.srt_text_mode,
            )

        return SubtitleExportResult(
            sidecar_srt_path=sidecar_srt_path,
            embedded_srt_path=embedded_srt_path,
        )


def _write_manifest(
    context: DubRunContext,
    inputs: ManifestInputs,
) -> Path | None:
    if not context.options.write_manifest:
        return None

    run_options = context.options
    artifacts = context.artifact_paths
    adapters = context.adapters
    resolved_manifest_path = artifacts.manifest_path

    with _progress_stage(
        context.progress_callback,
        "write_manifest",
        "Writing run manifest.",
    ):
        manifest = build_dubbing_manifest(
            started_at=inputs.started_at,
            finished_at=datetime.now(UTC),
            input_video_path=context.paths.video_path,
            output_video_path=inputs.dubbed_video_path,
            source_language=inputs.source_transcript.source_language,
            target_language=run_options.target_language,
            asr_adapter=adapters.asr,
            diarization_adapter=adapters.diarization,
            translation_adapter=adapters.translation,
            text_adapter=adapters.text_adapter,
            dubbing_script_adapter=adapters.dubbing_script,
            source_separation_adapter=adapters.source_separation,
            tts_adapter=adapters.tts,
            speaker_voices=adapters.speaker_voices,
            options=DubbingOptionsManifest(
                asr_sample_rate=run_options.asr_sample_rate,
                diarize=run_options.diarize,
                diarization_min_speakers=run_options.diarization_min_speakers,
                diarization_max_speakers=run_options.diarization_max_speakers,
                speech_sample_rate=run_options.speech_sample_rate,
                repair_timing=run_options.repair_timing,
                max_timing_repair_attempts=run_options.max_timing_repair_attempts,
                timing_repair_target_speedup=run_options.timing_repair_target_speedup,
                fit_speech=run_options.fit_speech,
                max_speech_speedup=run_options.max_speech_speedup,
                min_speech_overrun_seconds=run_options.min_speech_overrun_seconds,
                fit_video=run_options.fit_video,
                max_video_slowdown=run_options.max_video_slowdown,
                mix_original_audio=run_options.mix_original_audio,
                background_mode=run_options.background_mode,
                original_audio_gain=run_options.original_audio_gain,
                ducking_gain=run_options.ducking_gain,
                speech_gain=run_options.speech_gain,
                ducking_margin_seconds=run_options.ducking_margin_seconds,
                ducking_fade_seconds=run_options.ducking_fade_seconds,
                normalize_final_audio=run_options.normalize_final_audio,
                target_final_lufs=run_options.target_final_lufs,
                final_true_peak=run_options.final_true_peak,
                final_loudness_range=run_options.final_loudness_range,
                text_workflow=run_options.text_workflow,
                translation_group_segments=run_options.translation_group_segments,
                max_translation_group_pause_seconds=(
                    run_options.max_translation_group_pause_seconds
                ),
                max_translation_group_duration_seconds=(
                    run_options.max_translation_group_duration_seconds
                ),
                max_translation_sentence_group_duration_seconds=(
                    run_options.max_translation_sentence_group_duration_seconds
                ),
                export_srt=run_options.export_srt,
                srt_text_mode=run_options.srt_text_mode,
                subtitle_embed=run_options.subtitle_embed,
                ffmpeg_executable=run_options.ffmpeg_executable,
                resume=run_options.resume,
                overwrite=run_options.overwrite,
                until_checkpoint=run_options.until_checkpoint,
                start_from_checkpoint=run_options.start_from_checkpoint,
            ),
            artifacts=DubbingArtifactsManifest(
                workspace_dir=str(context.paths.workspace_dir),
                extracted_audio_path=str(inputs.extracted_audio_path),
                source_transcript_path=str(inputs.source_transcript_path),
                diarized_transcript_path=(
                    str(inputs.diarized_transcript_path)
                    if inputs.diarized_transcript_path is not None
                    else None
                ),
                translated_transcript_path=str(inputs.translated_transcript_path),
                adapted_transcript_path=str(inputs.adapted_transcript_path),
                synthesized_transcript_path=str(inputs.synthesized_transcript_path),
                timing_repaired_transcript_path=(
                    str(inputs.timing_repaired_transcript_path)
                    if inputs.timing_repaired_transcript_path is not None
                    else None
                ),
                timing_repaired_speech_dir=(
                    str(inputs.timing_repaired_speech_dir)
                    if inputs.timing_repaired_speech_dir is not None
                    else None
                ),
                speech_dir=str(inputs.speech_dir),
                speech_track_path=str(inputs.speech_track_path),
                dubbed_video_path=str(inputs.dubbed_video_path),
                fitted_transcript_path=(
                    str(inputs.fitted_transcript_path)
                    if inputs.fitted_transcript_path is not None
                    else None
                ),
                fitted_speech_dir=(
                    str(inputs.fitted_speech_dir)
                    if inputs.fitted_speech_dir is not None
                    else None
                ),
                video_fitted_transcript_path=(
                    str(inputs.video_fitted_transcript_path)
                    if inputs.video_fitted_transcript_path is not None
                    else None
                ),
                fitted_video_path=(
                    str(inputs.fitted_video_path)
                    if inputs.fitted_video_path is not None
                    else None
                ),
                video_fitted_original_audio_path=(
                    str(inputs.video_fitted_original_audio_path)
                    if inputs.video_fitted_original_audio_path is not None
                    else None
                ),
                separated_background_audio_path=(
                    str(inputs.separated_background_audio_path)
                    if inputs.separated_background_audio_path is not None
                    else None
                ),
                separated_voice_audio_path=(
                    str(inputs.separated_voice_audio_path)
                    if inputs.separated_voice_audio_path is not None
                    else None
                ),
                video_fitted_background_audio_path=(
                    str(inputs.video_fitted_background_audio_path)
                    if inputs.video_fitted_background_audio_path is not None
                    else None
                ),
                mix_original_audio_path=(
                    str(inputs.mix_original_audio_path)
                    if inputs.mix_original_audio_path is not None
                    else None
                ),
                mixed_audio_path=(
                    str(inputs.mixed_audio_path)
                    if inputs.mixed_audio_path is not None
                    else None
                ),
                normalized_audio_path=(
                    str(inputs.normalized_audio_path)
                    if inputs.normalized_audio_path is not None
                    else None
                ),
                srt_path=str(inputs.srt_path) if inputs.srt_path is not None else None,
                embedded_srt_path=(
                    str(inputs.embedded_srt_path)
                    if inputs.embedded_srt_path is not None
                    else None
                ),
                manifest_path=str(resolved_manifest_path),
            ),
            metadata={
                "source_segment_count": str(len(inputs.source_transcript.segments)),
                "translated_segment_count": str(
                    len(inputs.translated_transcript.segments)
                ),
                "adapted_segment_count": str(len(inputs.adapted_transcript.segments)),
                "synthesized_segment_count": str(
                    len(inputs.synthesized_transcript.segments)
                ),
                "final_speech_segment_count": str(
                    len(inputs.speech_timeline_transcript.segments)
                ),
                "configured_speaker_voice_count": str(
                    len(adapters.speaker_voices or {})
                ),
            },
        )

        return save_manifest(manifest, resolved_manifest_path)
