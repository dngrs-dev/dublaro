from pathlib import Path
from typing import cast

import typer

from dublaro.cli.services.adapter_factories import (
    create_asr_adapter,
    create_diarization_adapter,
    create_dubbing_script_adapter,
    create_source_separation_adapter,
    create_speaker_voice_preflight_settings,
    create_speaker_voices,
    create_text_adapter,
    create_translation_adapter,
    create_tts_adapter,
)
from dublaro.cli_config import ResolvedDubSettings
from dublaro.pipeline.dub import (
    DubbingArtifacts,
    DubbingProgressCallback,
    dub_video,
)
from dublaro.pipeline.dub_plan import BackgroundMode, TextWorkflowMode
from dublaro.pipeline.preflight import DubPreflightReport, validate_dub_preflight
from dublaro.pipeline.subtitles import SrtTextMode, SubtitleEmbedMode


def parse_text_workflow_mode(mode: str) -> TextWorkflowMode:
    allowed_modes = {"translate-then-adapt", "llm-dubbing"}

    if mode not in allowed_modes:
        raise typer.BadParameter(
            "Text workflow must be one of: translate-then-adapt, llm-dubbing."
        )

    return cast(TextWorkflowMode, mode)


def parse_background_mode(mode: str) -> BackgroundMode:
    allowed_modes = {"speech-only", "original", "ducked", "separated"}

    if mode not in allowed_modes:
        raise typer.BadParameter(
            "Background mode must be one of: speech-only, original, ducked, separated."
        )

    return cast(BackgroundMode, mode)


def parse_srt_text_mode(text_mode: str) -> SrtTextMode:
    allowed_modes = {"auto", "source", "translated", "adapted"}

    if text_mode not in allowed_modes:
        raise typer.BadParameter(
            "SRT text must be one of: auto, source, translated, adapted."
        )

    return cast(SrtTextMode, text_mode)


def parse_subtitle_embed_mode(mode: str) -> SubtitleEmbedMode:
    allowed_modes = {"none", "soft", "hard"}

    if mode not in allowed_modes:
        raise typer.BadParameter("Subtitle embed must be one of: none, soft, hard.")

    return cast(SubtitleEmbedMode, mode)


def validate_resolved_dub_settings(
    settings: ResolvedDubSettings,
) -> tuple[TextWorkflowMode, BackgroundMode, SrtTextMode, SubtitleEmbedMode]:
    parsed_text_workflow = parse_text_workflow_mode(settings.text_workflow)
    parsed_background_mode = parse_background_mode(settings.background_mode)
    parsed_srt_text_mode = parse_srt_text_mode(settings.srt_text_mode)
    parsed_subtitle_embed = parse_subtitle_embed_mode(settings.subtitle_embed)

    if (
        parsed_text_workflow == "llm-dubbing"
        and settings.translation_backend != "ollama"
    ):
        raise ValueError("--text-workflow llm-dubbing requires --translator ollama.")

    if settings.manifest_output_path is not None and not settings.write_manifest:
        raise ValueError(
            "--manifest-output cannot be used when manifest writing is disabled."
        )

    if settings.resume and settings.overwrite:
        raise ValueError("--resume cannot be used with --overwrite.")

    if settings.repair_timing and settings.text_adapter_backend != "ollama":
        raise ValueError("--repair-timing currently requires --text-adapter ollama.")

    if settings.timing_repair_target_speedup > settings.max_speech_speedup:
        raise ValueError(
            "--timing-repair-target-speedup cannot be greater than "
            "--max-speech-speedup."
        )

    return (
        parsed_text_workflow,
        parsed_background_mode,
        parsed_srt_text_mode,
        parsed_subtitle_embed,
    )


def run_dub_preflight(
    video_path: Path,
    settings: ResolvedDubSettings,
) -> DubPreflightReport:
    return validate_dub_preflight(
        video_path=video_path,
        output_path=settings.output_path,
        workspace_dir=settings.workspace_dir,
        overwrite=settings.overwrite,
        ffmpeg_executable=settings.ffmpeg_executable,
        asr_backend=settings.asr_backend,
        translation_backend=settings.translation_backend,
        source_language=settings.source_language,
        target_language=settings.target_language,
        install_translation_package=settings.install_package,
        translation_ollama_model=settings.translation_ollama_model,
        translation_ollama_url=settings.translation_ollama_url,
        translation_ollama_timeout_seconds=settings.translation_ollama_timeout_seconds,
        text_adapter_backend=settings.text_adapter_backend,
        background_mode=settings.background_mode,
        source_separation_backend=settings.source_separation_backend,
        demucs_executable=settings.demucs_executable,
        ollama_model=settings.ollama_model,
        ollama_url=settings.ollama_url,
        ollama_timeout_seconds=settings.ollama_timeout_seconds,
        tts_backend=settings.tts_backend,
        piper_model_path=settings.piper_model_path,
        piper_config_path=settings.piper_config_path,
        piper_executable=settings.piper_executable,
        speaker_voices=create_speaker_voice_preflight_settings(settings.voice_profiles),
        export_srt=settings.export_srt,
        srt_output_path=settings.srt_output_path,
        write_manifest=settings.write_manifest,
        manifest_output_path=settings.manifest_output_path,
        resume=settings.resume,
    )


def run_resolved_dub(
    video_path: Path,
    settings: ResolvedDubSettings,
    *,
    parsed_text_workflow: TextWorkflowMode,
    parsed_background_mode: BackgroundMode,
    parsed_srt_text_mode: SrtTextMode,
    parsed_subtitle_embed: SubtitleEmbedMode,
    progress_callback: DubbingProgressCallback | None,
) -> DubbingArtifacts:
    dubbing_script_adapter = (
        create_dubbing_script_adapter(
            settings.translation_backend,
            ollama_model=settings.translation_ollama_model,
            ollama_url=settings.translation_ollama_url,
            ollama_timeout_seconds=settings.translation_ollama_timeout_seconds,
            ollama_temperature=settings.translation_ollama_temperature,
        )
        if parsed_text_workflow == "llm-dubbing"
        else None
    )
    source_separation_adapter = (
        create_source_separation_adapter(
            settings.source_separation_backend,
            demucs_executable=settings.demucs_executable,
            demucs_model=settings.demucs_model,
            demucs_device=settings.demucs_device,
            ffmpeg_executable=settings.ffmpeg_executable,
        )
        if parsed_background_mode == "separated"
        else None
    )
    return dub_video(
        video_path,
        settings.output_path,
        source_language=settings.source_language,
        target_language=settings.target_language,
        workspace_dir=settings.workspace_dir,
        asr_adapter=create_asr_adapter(
            settings.asr_backend,
            model_size=settings.model_size,
            device=settings.device,
            compute_type=settings.compute_type,
        ),
        diarization_adapter=(
            create_diarization_adapter(
                settings.diarization_backend,
                model_id=settings.diarization_model_id,
                device=settings.diarization_device,
                token_env_var=settings.diarization_token_env_var,
            )
            if settings.diarize
            else None
        ),
        translation_adapter=create_translation_adapter(
            settings.translation_backend,
            auto_install=settings.install_package,
            ollama_model=settings.translation_ollama_model,
            ollama_url=settings.translation_ollama_url,
            ollama_timeout_seconds=settings.translation_ollama_timeout_seconds,
            ollama_temperature=settings.translation_ollama_temperature,
        ),
        text_adapter=create_text_adapter(
            settings.text_adapter_backend,
            ollama_model=settings.ollama_model,
            ollama_url=settings.ollama_url,
            ollama_timeout_seconds=settings.ollama_timeout_seconds,
            ollama_temperature=settings.ollama_temperature,
        ),
        text_workflow=parsed_text_workflow,
        dubbing_script_adapter=dubbing_script_adapter,
        source_separation_adapter=source_separation_adapter,
        background_mode=parsed_background_mode,
        tts_adapter=create_tts_adapter(
            settings.tts_backend,
            piper_model_path=settings.piper_model_path,
            piper_config_path=settings.piper_config_path,
            piper_executable=settings.piper_executable,
            piper_speaker=settings.piper_speaker,
        ),
        speaker_voices=create_speaker_voices(settings.voice_profiles),
        diarize=settings.diarize,
        diarization_min_speakers=settings.diarization_min_speakers,
        diarization_max_speakers=settings.diarization_max_speakers,
        translation_group_segments=settings.translation_group_segments,
        max_translation_group_pause_seconds=settings.max_translation_group_pause_seconds,
        max_translation_group_duration_seconds=settings.max_translation_group_duration_seconds,
        max_translation_sentence_group_duration_seconds=(
            settings.max_translation_sentence_group_duration_seconds
        ),
        asr_sample_rate=settings.asr_sample_rate,
        speech_sample_rate=settings.speech_sample_rate,
        repair_timing=settings.repair_timing,
        max_timing_repair_attempts=settings.max_timing_repair_attempts,
        timing_repair_target_speedup=settings.timing_repair_target_speedup,
        fit_speech=settings.fit_speech,
        max_speech_speedup=settings.max_speech_speedup,
        min_speech_overrun_seconds=settings.min_speech_overrun_seconds,
        fit_video=settings.fit_video,
        max_video_slowdown=settings.max_video_slowdown,
        mix_original_audio=settings.mix_original_audio,
        original_audio_gain=settings.original_audio_gain,
        ducking_gain=settings.ducking_gain,
        speech_gain=settings.speech_gain,
        ducking_margin_seconds=settings.ducking_margin_seconds,
        ducking_fade_seconds=settings.ducking_fade_seconds,
        export_srt=settings.export_srt,
        srt_output_path=settings.srt_output_path,
        srt_text_mode=parsed_srt_text_mode,
        subtitle_embed=parsed_subtitle_embed,
        progress_callback=progress_callback,
        write_manifest=settings.write_manifest,
        manifest_output_path=settings.manifest_output_path,
        ffmpeg_executable=settings.ffmpeg_executable,
        resume=settings.resume,
        overwrite=settings.overwrite,
    )
