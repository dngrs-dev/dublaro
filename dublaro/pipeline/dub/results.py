from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from dublaro.schemas import Transcript


@dataclass(frozen=True)
class TextWorkflowResult:
    translated_transcript: Transcript
    adapted_transcript: Transcript


@dataclass(frozen=True)
class SpeechTimingResult:
    transcript: Transcript
    fitted_transcript_path: Path | None = None
    fitted_speech_dir: Path | None = None


@dataclass(frozen=True)
class TimingRepairResult:
    transcript: Transcript
    timing_repaired_transcript_path: Path | None = None
    timing_repaired_speech_dir: Path | None = None


@dataclass(frozen=True)
class VideoFitResult:
    transcript: Transcript
    video_path: Path
    slowdown_factor: float = 1.0
    video_fitted_transcript_path: Path | None = None
    fitted_video_path: Path | None = None


@dataclass(frozen=True)
class ExportAudioResult:
    audio_path: Path
    mix_original_audio_path: Path | None = None
    mixed_audio_path: Path | None = None
    video_fitted_original_audio_path: Path | None = None
    separated_background_audio_path: Path | None = None
    separated_voice_audio_path: Path | None = None
    video_fitted_background_audio_path: Path | None = None


@dataclass(frozen=True)
class AudioNormalizationResult:
    audio_path: Path
    normalized_audio_path: Path | None = None


@dataclass(frozen=True)
class SubtitleExportResult:
    sidecar_srt_path: Path | None = None
    embedded_srt_path: Path | None = None


@dataclass(frozen=True)
class ManifestInputs:
    started_at: datetime
    extracted_audio_path: Path
    source_transcript_path: Path
    diarized_transcript_path: Path | None
    translated_transcript_path: Path
    adapted_transcript_path: Path
    synthesized_transcript_path: Path
    timing_repaired_transcript_path: Path | None
    timing_repaired_speech_dir: Path | None
    speech_dir: Path
    speech_track_path: Path
    dubbed_video_path: Path
    fitted_transcript_path: Path | None
    fitted_speech_dir: Path | None
    video_fitted_transcript_path: Path | None
    fitted_video_path: Path | None
    video_fitted_original_audio_path: Path | None
    separated_background_audio_path: Path | None
    separated_voice_audio_path: Path | None
    video_fitted_background_audio_path: Path | None
    mix_original_audio_path: Path | None
    mixed_audio_path: Path | None
    normalized_audio_path: Path | None
    srt_path: Path | None
    embedded_srt_path: Path | None
    source_transcript: Transcript
    translated_transcript: Transcript
    adapted_transcript: Transcript
    synthesized_transcript: Transcript
    speech_timeline_transcript: Transcript
