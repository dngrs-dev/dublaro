from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from dublaro.adapters.asr import AsrAdapter
from dublaro.adapters.diarization import DiarizationAdapter
from dublaro.adapters.text_adapter import TextAdapter
from dublaro.adapters.translation import TranslationAdapter
from dublaro.adapters.tts import TtsAdapter
from dublaro.pipeline.subtitles import SrtTextMode, SubtitleEmbedMode
from dublaro.pipeline.voices import SpeakerVoice


@dataclass(frozen=True)
class DubAdapters:
    asr: AsrAdapter
    translation: TranslationAdapter
    text_adapter: TextAdapter
    tts: TtsAdapter
    diarization: DiarizationAdapter | None = None
    speaker_voices: Mapping[str, SpeakerVoice] | None = None


@dataclass(frozen=True)
class DubOptions:
    source_language: str | None
    target_language: str
    asr_sample_rate: int = 16_000
    speech_sample_rate: int = 24_000
    repair_timing: bool = False
    max_timing_repair_attempts: int = 2
    timing_repair_target_speedup: float = 1.15
    fit_speech: bool = False
    max_speech_speedup: float = 1.35
    min_speech_overrun_seconds: float = 0.05
    fit_video: bool = False
    max_video_slowdown: float = 1.5
    mix_original_audio: bool = False
    original_audio_gain: float = 1.0
    ducking_gain: float = 0.25
    speech_gain: float = 1.0
    ducking_margin_seconds: float = 0.05
    ducking_fade_seconds: float = 0.05
    translation_group_segments: bool = True
    max_translation_group_pause_seconds: float = 0.8
    max_translation_group_duration_seconds: float = 12.0
    max_translation_sentence_group_duration_seconds: float = 24.0
    diarize: bool = False
    diarization_min_speakers: int | None = None
    diarization_max_speakers: int | None = None
    export_srt: bool = False
    srt_output_path: Path | None = None
    srt_text_mode: SrtTextMode = "adapted"
    subtitle_embed: SubtitleEmbedMode = "none"
    write_manifest: bool = True
    manifest_output_path: Path | None = None
    ffmpeg_executable: str = "ffmpeg"
    resume: bool = False
    overwrite: bool = False

    def __post_init__(self) -> None:
        if self.max_timing_repair_attempts < 1:
            raise ValueError("max_timing_repair_attempts must be >= 1")

        if self.timing_repair_target_speedup < 1.0:
            raise ValueError("timing_repair_target_speedup must be >= 1.0")

        if self.timing_repair_target_speedup > self.max_speech_speedup:
            raise ValueError(
                "timing_repair_target_speedup cannot be greater than max_speech_speedup"
            )

        if self.manifest_output_path is not None and not self.write_manifest:
            raise ValueError(
                "manifest_output_path cannot be used when write_manifest is False."
            )

        if self.resume and self.overwrite:
            raise ValueError("resume cannot be used with overwrite.")

        if self.max_video_slowdown < 1.0:
            raise ValueError("max_video_slowdown must be >= 1.0")


@dataclass(frozen=True)
class DubArtifactPaths:
    extracted_audio_path: Path
    source_transcript_path: Path
    translated_transcript_path: Path
    adapted_transcript_path: Path
    synthesized_transcript_path: Path
    diarized_transcript_path: Path
    speech_dir: Path
    timing_repaired_transcript_path: Path
    timing_repaired_speech_dir: Path
    speech_track_path: Path
    fitted_transcript_path: Path
    fitted_speech_dir: Path
    video_fitted_transcript_path: Path
    fitted_video_path: Path
    video_fitted_original_audio_path: Path
    mix_original_audio_path: Path
    mixed_audio_path: Path
    srt_path: Path
    subtitle_embed_srt_path: Path
    manifest_path: Path


@dataclass(frozen=True)
class DubPaths:
    video_path: Path
    output_path: Path
    workspace_dir: Path

    @classmethod
    def build(
        cls,
        video_path: str | Path,
        output_path: str | Path,
        workspace_dir: str | Path,
    ) -> "DubPaths":
        return cls(
            video_path=Path(video_path),
            output_path=Path(output_path),
            workspace_dir=Path(workspace_dir),
        )

    def artifacts(self, options: DubOptions) -> DubArtifactPaths:
        source_label = options.source_language or "auto"
        stem = self.video_path.stem

        timing_stem = f"{stem}.{options.target_language}"
        if options.fit_video:
            timing_stem = f"{timing_stem}.video-fitted"

        return DubArtifactPaths(
            extracted_audio_path=self.workspace_dir / f"{stem}.audio.wav",
            source_transcript_path=self.workspace_dir / f"{stem}.{source_label}.json",
            translated_transcript_path=(
                self.workspace_dir / f"{stem}.{options.target_language}.json"
            ),
            adapted_transcript_path=(
                self.workspace_dir / f"{stem}.{options.target_language}.adapted.json"
            ),
            synthesized_transcript_path=(
                self.workspace_dir
                / f"{stem}.{options.target_language}.synthesized.json"
            ),
            diarized_transcript_path=self.workspace_dir
            / f"{stem}.{source_label}.diarized.json",
            speech_dir=self.workspace_dir / f"{stem}.{options.target_language}.speech",
            timing_repaired_transcript_path=(
                self.workspace_dir
                / f"{stem}.{options.target_language}.timing-repaired.json"
            ),
            timing_repaired_speech_dir=(
                self.workspace_dir
                / f"{stem}.{options.target_language}.timing-repaired-speech"
            ),
            speech_track_path=self.workspace_dir / f"{timing_stem}.speech-track.wav",
            fitted_transcript_path=(
                self.workspace_dir / f"{stem}.{options.target_language}.fitted.json"
            ),
            fitted_speech_dir=(
                self.workspace_dir / f"{stem}.{options.target_language}.fitted-speech"
            ),
            video_fitted_transcript_path=(
                self.workspace_dir
                / f"{stem}.{options.target_language}.video-fitted.json"
            ),
            fitted_video_path=(
                self.workspace_dir
                / f"{stem}.{options.target_language}.video-fitted{self.video_path.suffix}"
            ),
            video_fitted_original_audio_path=(
                self.workspace_dir
                / f"{stem}.{options.target_language}.original-video-fitted.wav"
            ),
            mix_original_audio_path=self.workspace_dir / f"{stem}.original-mix.wav",
            mixed_audio_path=self.workspace_dir / f"{timing_stem}.mixed.wav",
            srt_path=options.srt_output_path or self.output_path.with_suffix(".srt"),
            subtitle_embed_srt_path=self.workspace_dir / f"{timing_stem}.embed.srt",
            manifest_path=(
                options.manifest_output_path
                or self.workspace_dir
                / f"{stem}.{options.target_language}.manifest.json"
            ),
        )
