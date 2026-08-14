from pathlib import Path
from typing import Annotated

import typer

from dublaro.audio.ffmpeg import (
    FFmpegError,
)
from dublaro.cli.dub_runner import (
    run_dub_preflight,
    run_resolved_dub,
)
from dublaro.cli.dub_settings import (
    DubCommandOverrides,
    resolve_dub_command_settings,
)
from dublaro.cli.rendering import (
    console,
    print_adapter_notes,
    print_dub_artifacts,
    print_dub_progress,
    print_preflight_report,
)
from dublaro.config import (
    DublaroConfigError,
)


def dub(
    video_path: Annotated[
        Path,
        typer.Argument(
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            help="Input video file.",
        ),
    ],
    config_path: Annotated[
        Path | None,
        typer.Option(
            "--config",
            "-c",
            help="Path to dublaro.toml config file.",
        ),
    ] = None,
    target_language: Annotated[
        str | None,
        typer.Option(
            "--to",
            help="Target language code.",
        ),
    ] = None,
    output_path: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help="Output dubbed video path.",
        ),
    ] = None,
    output_dir: Annotated[
        Path | None,
        typer.Option(
            "--output-dir",
            help="Directory for the output dubbed video.",
        ),
    ] = None,
    source_language: Annotated[
        str | None,
        typer.Option(
            "--from",
            help="Source language code.",
        ),
    ] = None,
    workspace_dir: Annotated[
        Path | None,
        typer.Option(
            "--workspace",
            help="Directory for intermediate artifacts.",
        ),
    ] = None,
    resume_enabled: Annotated[
        bool | None,
        typer.Option(
            "--resume/--no-resume",
            help="Reuse valid intermediate workspace artifacts.",
        ),
    ] = None,
    asr_backend: Annotated[
        str | None,
        typer.Option(
            "--asr",
            help="ASR backend: fake or faster-whisper.",
        ),
    ] = None,
    translation_backend: Annotated[
        str | None,
        typer.Option(
            "--translator",
            help="Translation backend: fake or argos.",
        ),
    ] = None,
    text_adapter_backend: Annotated[
        str | None,
        typer.Option(
            "--text-adapter",
            help="Text adaptation backend: fake or rules.",
        ),
    ] = None,
    ollama_model: Annotated[
        str | None,
        typer.Option(
            "--ollama-model",
            help="Ollama model used when --text-adapter ollama.",
        ),
    ] = None,
    ollama_url: Annotated[
        str | None,
        typer.Option(
            "--ollama-url",
            help="Ollama server URL used when --text-adapter ollama.",
        ),
    ] = None,
    ollama_timeout_seconds: Annotated[
        float | None,
        typer.Option(
            "--ollama-timeout",
            help="Ollama request timeout in seconds.",
        ),
    ] = None,
    ollama_temperature: Annotated[
        float | None,
        typer.Option(
            "--ollama-temperature",
            help="Ollama generation temperature.",
        ),
    ] = None,
    tts_backend: Annotated[
        str | None,
        typer.Option(
            "--tts",
            help="TTS backend: fake.",
        ),
    ] = None,
    model_size: Annotated[
        str | None,
        typer.Option(
            "--model",
            help="faster-whisper model size.",
        ),
    ] = None,
    device: Annotated[
        str | None,
        typer.Option(
            "--device",
            help="Inference device: cpu or cuda.",
        ),
    ] = None,
    compute_type: Annotated[
        str | None,
        typer.Option(
            "--compute-type",
            help="faster-whisper compute type.",
        ),
    ] = None,
    diarize_enabled: Annotated[
        bool | None,
        typer.Option(
            "--diarize/--no-diarize",
            help="Assign speaker labels to transcript segments.",
        ),
    ] = None,
    diarization_backend: Annotated[
        str | None,
        typer.Option(
            "--diarizer",
            help="Diarization backend: fake.",
        ),
    ] = None,
    diarization_model_id: Annotated[
        str | None,
        typer.Option(
            "--diarization-model",
            help="pyannote model id or local pipeline path.",
        ),
    ] = None,
    diarization_device: Annotated[
        str | None,
        typer.Option(
            "--diarization-device",
            help="Device for pyannote, for example cpu or cuda.",
        ),
    ] = None,
    diarization_token_env_var: Annotated[
        str | None,
        typer.Option(
            "--diarization-token-env",
            help="Environment variable containing the Hugging Face token.",
        ),
    ] = None,
    diarization_min_speakers: Annotated[
        int | None,
        typer.Option(
            "--min-speakers",
            help="Minimum expected speaker count for diarization.",
        ),
    ] = None,
    diarization_max_speakers: Annotated[
        int | None,
        typer.Option(
            "--max-speakers",
            help="Maximum expected speaker count for diarization.",
        ),
    ] = None,
    install_package: Annotated[
        bool | None,
        typer.Option(
            "--install-package/--no-install-package",
            help="Download and install translation package if missing.",
        ),
    ] = None,
    translation_group_segments: Annotated[
        bool | None,
        typer.Option(
            "--group-segments/--no-group-segments",
            help="Translate nearby sentence fragments as one natural unit.",
        ),
    ] = None,
    max_translation_group_pause_seconds: Annotated[
        float | None,
        typer.Option(
            "--max-group-pause",
            help="Maximum pause between segments grouped for translation.",
        ),
    ] = None,
    max_translation_group_duration_seconds: Annotated[
        float | None,
        typer.Option(
            "--max-group-duration",
            help="Maximum duration for one grouped translation unit.",
        ),
    ] = None,
    max_translation_sentence_group_duration_seconds: Annotated[
        float | None,
        typer.Option(
            "--max-sentence-group-duration",
            help="Hard maximum duration for one unfinished sentence translation unit.",
        ),
    ] = None,
    asr_sample_rate: Annotated[
        int | None,
        typer.Option(
            "--asr-sample-rate",
            help="Audio sample rate used for ASR.",
        ),
    ] = None,
    speech_sample_rate: Annotated[
        int | None,
        typer.Option(
            "--speech-sample-rate",
            help="Generated speech sample rate.",
        ),
    ] = None,
    piper_model_path: Annotated[
        Path | None,
        typer.Option(
            "--piper-model",
            help="Path to Piper .onnx voice model.",
        ),
    ] = None,
    piper_config_path: Annotated[
        Path | None,
        typer.Option(
            "--piper-config",
            help="Path to Piper .onnx.json voice config.",
        ),
    ] = None,
    piper_executable: Annotated[
        str | None,
        typer.Option(
            "--piper-executable",
            help="Piper executable name or path.",
        ),
    ] = None,
    piper_speaker: Annotated[
        int | None,
        typer.Option(
            "--piper-speaker",
            help="Piper speaker id for multi-speaker voices.",
        ),
    ] = None,
    repair_timing_enabled: Annotated[
        bool | None,
        typer.Option(
            "--repair-timing/--no-repair-timing",
            help="Rewrite and resynthesize overlong segments before speech/video fitting.",
        ),
    ] = None,
    max_timing_repair_attempts: Annotated[
        int | None,
        typer.Option(
            "--timing-repair-attempts",
            help="Maximum rewrite attempts for each overlong segment.",
        ),
    ] = None,
    timing_repair_target_speedup: Annotated[
        float | None,
        typer.Option(
            "--timing-repair-target-speedup",
            help="Repair text when generated speech needs more than this speedup.",
        ),
    ] = None,
    fit_speech_enabled: Annotated[
        bool | None,
        typer.Option(
            "--fit-speech/--no-fit-speech",
            help="Speed up overlong generated speech clips before alignment.",
        ),
    ] = None,
    max_speech_speedup: Annotated[
        float | None,
        typer.Option(
            "--max-speech-speedup",
            help="Maximum allowed audio speedup factor when fitting speech.",
        ),
    ] = None,
    min_speech_overrun_seconds: Annotated[
        float | None,
        typer.Option(
            "--min-speech-overrun",
            help="Only fit clips longer than this tolerance.",
        ),
    ] = None,
    fit_video_enabled: Annotated[
        bool | None,
        typer.Option(
            "--fit-video/--no-fit-video",
            help="Slow the video when generated speech is still too long.",
        ),
    ] = None,
    max_video_slowdown: Annotated[
        float | None,
        typer.Option(
            "--max-video-slowdown",
            help="Maximum allowed video slowdown factor.",
        ),
    ] = None,
    mix_original_audio_enabled: Annotated[
        bool | None,
        typer.Option(
            "--mix-original-audio/--no-mix-original-audio",
            help="Mix dubbed speech over lowered original audio.",
        ),
    ] = None,
    original_audio_gain: Annotated[
        float | None,
        typer.Option(
            "--original-audio-gain",
            help="Original audio volume multiplier outside dubbed speech.",
        ),
    ] = None,
    ducking_gain: Annotated[
        float | None,
        typer.Option(
            "--ducking-gain",
            help="Original audio volume multiplier during dubbed speech.",
        ),
    ] = None,
    speech_gain: Annotated[
        float | None,
        typer.Option(
            "--speech-gain",
            help="Dubbed speech volume multiplier.",
        ),
    ] = None,
    ducking_margin_seconds: Annotated[
        float | None,
        typer.Option(
            "--ducking-margin",
            help="Extra ducking time before and after each speech segment.",
        ),
    ] = None,
    ducking_fade_seconds: Annotated[
        float | None,
        typer.Option(
            "--ducking-fade",
            help="Fade time for ducking transitions.",
        ),
    ] = None,
    export_srt_enabled: Annotated[
        bool | None,
        typer.Option(
            "--export-srt/--no-export-srt",
            help="Save an external SRT subtitle file for the final spoken text.",
        ),
    ] = None,
    srt_output_path: Annotated[
        Path | None,
        typer.Option(
            "--srt-output",
            help="Output SRT path. Defaults to output video path with .srt extension.",
        ),
    ] = None,
    srt_text_mode: Annotated[
        str | None,
        typer.Option(
            "--srt-text",
            help="SRT text: auto, source, translated, or adapted.",
        ),
    ] = None,
    subtitle_embed: Annotated[
        str | None,
        typer.Option(
            "--subtitle-embed",
            help="Embed subtitles into output video: none, soft, or hard.",
        ),
    ] = None,
    write_manifest_enabled: Annotated[
        bool | None,
        typer.Option(
            "--manifest/--no-manifest",
            help="Save a JSON manifest describing this dubbing run.",
        ),
    ] = None,
    manifest_output_path: Annotated[
        Path | None,
        typer.Option(
            "--manifest-output",
            help="Output manifest path. Defaults to the workspace manifest path.",
        ),
    ] = None,
    preflight_enabled: Annotated[
        bool | None,
        typer.Option(
            "--preflight/--no-preflight",
            help="Check tools and paths before starting the dubbing run.",
        ),
    ] = None,
    ffmpeg_executable: Annotated[
        str | None,
        typer.Option(
            "--ffmpeg",
            help="ffmpeg executable name or path.",
        ),
    ] = None,
    overwrite: Annotated[
        bool | None,
        typer.Option(
            "--overwrite/--no-overwrite",
            help="Replace existing intermediate and output files.",
        ),
    ] = None,
) -> None:
    """Run the full dubbing pipeline."""
    try:
        resolved_dub = resolve_dub_command_settings(
            video_path=video_path,
            config_path=config_path,
            overrides=DubCommandOverrides(
                source_language=source_language,
                target_language=target_language,
                output_path=output_path,
                output_dir=output_dir,
                workspace_dir=workspace_dir,
                resume_enabled=resume_enabled,
                overwrite=overwrite,
                preflight_enabled=preflight_enabled,
                ffmpeg_executable=ffmpeg_executable,
                asr_sample_rate=asr_sample_rate,
                speech_sample_rate=speech_sample_rate,
                asr_backend=asr_backend,
                model_size=model_size,
                device=device,
                compute_type=compute_type,
                diarize_enabled=diarize_enabled,
                diarization_backend=diarization_backend,
                diarization_model_id=diarization_model_id,
                diarization_device=diarization_device,
                diarization_token_env_var=diarization_token_env_var,
                diarization_min_speakers=diarization_min_speakers,
                diarization_max_speakers=diarization_max_speakers,
                translation_backend=translation_backend,
                install_package=install_package,
                translation_group_segments=translation_group_segments,
                max_translation_group_pause_seconds=max_translation_group_pause_seconds,
                max_translation_group_duration_seconds=(
                    max_translation_group_duration_seconds
                ),
                max_translation_sentence_group_duration_seconds=(
                    max_translation_sentence_group_duration_seconds
                ),
                text_adapter_backend=text_adapter_backend,
                ollama_model=ollama_model,
                ollama_url=ollama_url,
                ollama_timeout_seconds=ollama_timeout_seconds,
                ollama_temperature=ollama_temperature,
                tts_backend=tts_backend,
                piper_model_path=piper_model_path,
                piper_config_path=piper_config_path,
                piper_executable=piper_executable,
                piper_speaker=piper_speaker,
                repair_timing_enabled=repair_timing_enabled,
                max_timing_repair_attempts=max_timing_repair_attempts,
                timing_repair_target_speedup=timing_repair_target_speedup,
                fit_speech_enabled=fit_speech_enabled,
                max_speech_speedup=max_speech_speedup,
                min_speech_overrun_seconds=min_speech_overrun_seconds,
                fit_video_enabled=fit_video_enabled,
                max_video_slowdown=max_video_slowdown,
                mix_original_audio_enabled=mix_original_audio_enabled,
                original_audio_gain=original_audio_gain,
                ducking_gain=ducking_gain,
                speech_gain=speech_gain,
                ducking_margin_seconds=ducking_margin_seconds,
                ducking_fade_seconds=ducking_fade_seconds,
                export_srt_enabled=export_srt_enabled,
                srt_output_path=srt_output_path,
                srt_text_mode=srt_text_mode,
                subtitle_embed=subtitle_embed,
                write_manifest_enabled=write_manifest_enabled,
                manifest_output_path=manifest_output_path,
            ),
        )
    except (DublaroConfigError, ValueError, typer.BadParameter) as error:
        console.print(f"[red]error:[/red] {error}")
        raise typer.Exit(code=1) from error

    settings = resolved_dub.settings

    if settings.preflight:
        report = run_dub_preflight(video_path, settings)
        print_preflight_report(report)

        if report.has_errors:
            raise typer.Exit(code=1)

    try:
        artifacts = run_resolved_dub(
            video_path,
            settings,
            parsed_srt_text_mode=resolved_dub.srt_text_mode,
            parsed_subtitle_embed=resolved_dub.subtitle_embed,
            progress_callback=print_dub_progress,
        )
    except (
        FFmpegError,
        FileExistsError,
        FileNotFoundError,
        RuntimeError,
        ValueError,
    ) as error:
        console.print(f"[red]error:[/red] {error}")
        raise typer.Exit(code=1) from error

    print_dub_artifacts(artifacts)
    print_adapter_notes(settings)
