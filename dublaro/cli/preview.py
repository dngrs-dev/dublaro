from dataclasses import dataclass
from pathlib import Path

from dublaro.adapters.tts import TtsAdapter
from dublaro.cli.factories import (
    create_speaker_voices,
    create_tts_adapter,
)
from dublaro.cli_config import DubCliOverrides, resolve_dub_settings
from dublaro.config import LoadedConfig, load_config, resolve_config_path
from dublaro.pipeline.speakers import (
    find_unconfigured_speakers,
    find_unused_voice_profiles,
    summarize_transcript_speakers,
)
from dublaro.pipeline.timing import TimingPreview, preview_speech_timing
from dublaro.pipeline.transcribe import load_transcript
from dublaro.pipeline.units import SegmentGroup, group_segments_for_translation
from dublaro.pipeline.voice_preview import VoiceSample, synthesize_voice_samples
from dublaro.schemas import Segment


@dataclass(frozen=True)
class TranslationUnitsPreview:
    groups: list[SegmentGroup]
    segment_count: int


@dataclass(frozen=True)
class SpeakerVoicePreview:
    source: str
    display_name: str | None
    tts_backend: str
    piper_model_path: Path | None
    piper_config_path: Path | None
    piper_speaker: int | None
    piper_executable: str


@dataclass(frozen=True)
class SpeakerPreviewRow:
    speaker_id: str
    segment_count: int
    total_duration_seconds: float
    window: str
    voice_route: str


@dataclass(frozen=True)
class SpeakerPreview:
    rows: list[SpeakerPreviewRow]
    segment_count: int
    configured_speaker_count: int
    unconfigured_speakers: list[str]
    unused_voice_profiles: list[str]


@dataclass(frozen=True)
class VoiceSamplesPreview:
    samples: list[VoiceSample]
    output_dir: Path


@dataclass(frozen=True)
class TimingPreviewReport:
    previews: list[TimingPreview]
    shown_previews: list[TimingPreview]
    attention_count: int
    video_fit_count: int


@dataclass(frozen=True)
class TimingRepairPreviewRow:
    segment_id: str
    start: float
    end: float
    speaker: str | None
    status: str
    attempts: int | None
    target_duration_seconds: float | None
    max_audio_duration_seconds: float | None
    audio_duration_before_seconds: float | None
    audio_duration_after_seconds: float | None
    required_speedup_before: float | None
    required_speedup_after: float | None
    reason: str
    model_reason: str | None
    text: str


@dataclass(frozen=True)
class TimingRepairPreviewReport:
    transcript_path: Path
    total_segments: int
    rows: list[TimingRepairPreviewRow]
    metadata: dict[str, str]

    @property
    def attempted_count(self) -> int:
        return sum(row.status != "not_attempted" for row in self.rows)

    @property
    def repaired_count(self) -> int:
        return sum(row.status == "repaired" for row in self.rows)

    @property
    def improved_count(self) -> int:
        return sum(row.status == "improved" for row in self.rows)

    @property
    def not_improved_count(self) -> int:
        return sum(row.status == "not_improved" for row in self.rows)

    @property
    def not_attempted_count(self) -> int:
        return sum(row.status == "not_attempted" for row in self.rows)


def build_translation_units_preview(
    transcript_path: Path,
    *,
    max_group_pause_seconds: float,
    max_group_duration_seconds: float,
    max_sentence_group_duration_seconds: float,
) -> TranslationUnitsPreview:
    transcript = load_transcript(transcript_path)
    groups = group_segments_for_translation(
        transcript,
        max_pause_seconds=max_group_pause_seconds,
        max_duration_seconds=max_group_duration_seconds,
        max_sentence_duration_seconds=max_sentence_group_duration_seconds,
    )

    return TranslationUnitsPreview(
        groups=groups,
        segment_count=len(transcript.segments),
    )


def build_speaker_preview(
    transcript_path: Path,
    *,
    config_path: Path | None,
) -> SpeakerPreview:
    transcript = load_transcript(transcript_path)
    loaded_config = load_config(config_path)
    summaries = summarize_transcript_speakers(transcript)
    configured_speaker_ids = tuple(loaded_config.config.voices)

    rows = [
        SpeakerPreviewRow(
            speaker_id=summary.speaker_id,
            segment_count=summary.segment_count,
            total_duration_seconds=summary.total_duration_seconds,
            window=format_speaker_window(summary.first_start, summary.last_end),
            voice_route=format_voice_route(
                preview_speaker_voice(summary.speaker_id, loaded_config)
            ),
        )
        for summary in summaries
    ]

    return SpeakerPreview(
        rows=rows,
        segment_count=len(transcript.segments),
        configured_speaker_count=len(configured_speaker_ids),
        unconfigured_speakers=find_unconfigured_speakers(
            transcript,
            configured_speaker_ids,
        ),
        unused_voice_profiles=find_unused_voice_profiles(
            transcript,
            configured_speaker_ids,
        ),
    )


def build_voice_samples_preview(
    *,
    config_path: Path | None,
    text: str,
    output_dir: Path | None,
    language: str | None,
    sample_rate: int | None,
) -> VoiceSamplesPreview:
    loaded_config = load_config(config_path)

    if language is None and loaded_config.config.dub.target_language is None:
        raise ValueError(
            "--language is required when dub.target_language is not set in config."
        )

    settings = resolve_dub_settings(
        video_path=Path("voice-preview.mp4"),
        loaded_config=loaded_config,
        overrides=DubCliOverrides(
            target_language=language,
            speech_sample_rate=sample_rate,
        ),
    )

    speaker_voices = create_speaker_voices(settings.voice_profiles)

    fallback_adapter: TtsAdapter | None = None
    if not speaker_voices:
        fallback_adapter = create_tts_adapter(
            settings.tts_backend,
            piper_model_path=settings.piper_model_path,
            piper_config_path=settings.piper_config_path,
            piper_executable=settings.piper_executable,
            piper_speaker=settings.piper_speaker,
        )

    sample_output_dir = output_dir or settings.workspace_dir / "voice-samples"

    samples = synthesize_voice_samples(
        text=text,
        output_dir=sample_output_dir,
        language=settings.target_language,
        sample_rate=settings.speech_sample_rate,
        speaker_voices=speaker_voices,
        fallback_adapter=fallback_adapter,
        fallback_tts_backend=settings.tts_backend,
    )

    return VoiceSamplesPreview(samples=samples, output_dir=sample_output_dir)


def build_timing_preview_report(
    transcript_path: Path,
    *,
    max_speedup: float,
    min_overrun_seconds: float,
    only_issues: bool,
) -> TimingPreviewReport:
    transcript = load_transcript(transcript_path)
    previews = preview_speech_timing(
        transcript,
        max_speedup=max_speedup,
        min_overrun_seconds=min_overrun_seconds,
    )

    shown_previews = (
        [preview for preview in previews if timing_preview_needs_attention(preview)]
        if only_issues
        else previews
    )

    return TimingPreviewReport(
        previews=previews,
        shown_previews=shown_previews,
        attention_count=sum(
            timing_preview_needs_attention(preview) for preview in previews
        ),
        video_fit_count=sum(preview.needs_video_fit for preview in previews),
    )


def build_timing_repair_preview_report(
    transcript_path: Path,
    *,
    include_all: bool,
) -> TimingRepairPreviewReport:
    transcript = load_transcript(transcript_path)
    rows: list[TimingRepairPreviewRow] = []

    for segment in transcript.sorted_segments():
        metadata = segment.metadata
        status = metadata.get("timing_repair_status")

        if status is None:
            if not include_all:
                continue

            rows.append(
                TimingRepairPreviewRow(
                    segment_id=segment.id,
                    start=segment.start,
                    end=segment.end,
                    speaker=segment.speaker,
                    status="not_attempted",
                    attempts=None,
                    target_duration_seconds=segment.duration,
                    max_audio_duration_seconds=None,
                    audio_duration_before_seconds=None,
                    audio_duration_after_seconds=None,
                    required_speedup_before=None,
                    required_speedup_after=None,
                    reason="not-attempted",
                    model_reason=None,
                    text=_timing_repair_segment_text(segment),
                )
            )
            continue

        attempts = _metadata_int(metadata, "timing_repair_attempts")
        before = _metadata_float(
            metadata,
            "timing_repair_original_audio_duration_seconds",
        )
        after = _metadata_float(metadata, "timing_repair_best_audio_duration_seconds")
        max_audio = _metadata_float(
            metadata, "timing_repair_max_audio_duration_seconds"
        )

        rows.append(
            TimingRepairPreviewRow(
                segment_id=segment.id,
                start=segment.start,
                end=segment.end,
                speaker=segment.speaker,
                status=status,
                attempts=attempts,
                target_duration_seconds=_metadata_float(
                    metadata,
                    "timing_repair_target_duration_seconds",
                ),
                max_audio_duration_seconds=max_audio,
                audio_duration_before_seconds=before,
                audio_duration_after_seconds=after,
                required_speedup_before=_metadata_float(
                    metadata,
                    "timing_repair_required_speedup_before",
                ),
                required_speedup_after=_metadata_float(
                    metadata,
                    "timing_repair_required_speedup_after",
                ),
                reason=_timing_repair_reason(
                    status=status,
                    attempts=attempts,
                    before=before,
                    after=after,
                    max_audio=max_audio,
                ),
                model_reason=_metadata_text(
                    metadata,
                    "timing_repair_model_reason",
                ),
                text=_timing_repair_segment_text(segment),
            )
        )

    return TimingRepairPreviewReport(
        transcript_path=transcript_path,
        total_segments=len(transcript.segments),
        rows=rows,
        metadata=dict(transcript.metadata),
    )


def preview_speaker_voice(
    speaker_id: str,
    loaded_config: LoadedConfig,
) -> SpeakerVoicePreview:
    config = loaded_config.config
    base_dir = loaded_config.base_dir
    fallback_tts = config.dub.tts
    voice_config = config.voices.get(speaker_id)

    if voice_config is None:
        return SpeakerVoicePreview(
            source="fallback",
            display_name=None,
            tts_backend=fallback_tts.backend or "fake",
            piper_model_path=resolve_config_path(
                fallback_tts.piper_model_path, base_dir
            ),
            piper_config_path=resolve_config_path(
                fallback_tts.piper_config_path, base_dir
            ),
            piper_speaker=fallback_tts.piper_speaker,
            piper_executable=fallback_tts.piper_executable or "piper",
        )

    return SpeakerVoicePreview(
        source="configured",
        display_name=voice_config.display_name,
        tts_backend=voice_config.tts_backend or fallback_tts.backend or "fake",
        piper_model_path=(
            resolve_config_path(voice_config.piper_model_path, base_dir)
            or resolve_config_path(fallback_tts.piper_model_path, base_dir)
        ),
        piper_config_path=(
            resolve_config_path(voice_config.piper_config_path, base_dir)
            or resolve_config_path(fallback_tts.piper_config_path, base_dir)
        ),
        piper_speaker=(
            voice_config.piper_speaker
            if voice_config.piper_speaker is not None
            else fallback_tts.piper_speaker
        ),
        piper_executable=(
            voice_config.piper_executable or fallback_tts.piper_executable or "piper"
        ),
    )


def format_speaker_window(first_start: float, last_end: float) -> str:
    return f"{first_start:.2f}-{last_end:.2f}s"


def format_voice_route(preview: SpeakerVoicePreview) -> str:
    parts = [preview.source]

    if preview.display_name:
        parts.append(preview.display_name)

    parts.append(preview.tts_backend)

    if preview.tts_backend == "piper":
        parts.append(
            str(preview.piper_model_path)
            if preview.piper_model_path
            else "(missing model)"
        )

        if preview.piper_speaker is not None:
            parts.append(f"speaker={preview.piper_speaker}")

    return " | ".join(parts)


def format_optional_seconds(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.2f}s"


def format_optional_factor(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.2f}x"


def timing_preview_needs_attention(preview: TimingPreview) -> bool:
    return preview.status != "ok"


def _metadata_text(metadata: dict[str, str], key: str) -> str | None:
    value = metadata.get(key)
    if value is None:
        return None

    cleaned = " ".join(value.split())
    return cleaned or None


def _metadata_float(metadata: dict[str, str], key: str) -> float | None:
    value = metadata.get(key)
    if value is None:
        return None

    try:
        return float(value)
    except ValueError:
        return None


def _metadata_int(metadata: dict[str, str], key: str) -> int | None:
    value = metadata.get(key)
    if value is None:
        return None

    try:
        return int(value)
    except ValueError:
        return None


def _timing_repair_reason(
    *,
    status: str,
    attempts: int | None,
    before: float | None,
    after: float | None,
    max_audio: float | None,
) -> str:
    if status == "repaired":
        return "fits-target"

    if status == "improved":
        return "shorter-but-over-target"

    if status == "not_improved":
        if attempts == 0:
            return "no-shorter-text"

        if before is not None and after is not None and after >= before:
            return "candidate-audio-not-shorter"

        if after is not None and max_audio is not None and after > max_audio:
            return "still-over-target"

        return "not-improved"

    return status


def _timing_repair_segment_text(segment: Segment) -> str:
    return " ".join(
        (segment.adapted_text or segment.translated_text or segment.source_text).split()
    )
