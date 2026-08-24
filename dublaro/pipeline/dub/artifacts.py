from dataclasses import dataclass
from pathlib import Path

from dublaro.pipeline.checkpoints import DubCheckpoint
from dublaro.pipeline.dub.context import DubRunContext


@dataclass(frozen=True)
class DubbingArtifacts:
    workspace_dir: Path
    extracted_audio_path: Path
    source_transcript_path: Path
    diarized_transcript_path: Path | None
    translated_transcript_path: Path
    adapted_transcript_path: Path
    synthesized_transcript_path: Path
    speech_dir: Path
    timing_repaired_transcript_path: Path | None
    timing_repaired_speech_dir: Path | None
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
    manifest_path: Path | None
    stopped_at_checkpoint: DubCheckpoint | None = None

    @property
    def completed(self) -> bool:
        return self.stopped_at_checkpoint is None


@dataclass
class DubRunState:
    extracted_audio_path: Path
    source_transcript_path: Path
    translated_transcript_path: Path
    adapted_transcript_path: Path
    synthesized_transcript_path: Path
    speech_dir: Path
    speech_track_path: Path
    dubbed_video_path: Path
    diarized_transcript_path: Path | None = None
    timing_repaired_transcript_path: Path | None = None
    timing_repaired_speech_dir: Path | None = None
    fitted_transcript_path: Path | None = None
    fitted_speech_dir: Path | None = None
    video_fitted_transcript_path: Path | None = None
    fitted_video_path: Path | None = None
    video_fitted_original_audio_path: Path | None = None
    separated_background_audio_path: Path | None = None
    separated_voice_audio_path: Path | None = None
    video_fitted_background_audio_path: Path | None = None
    mix_original_audio_path: Path | None = None
    mixed_audio_path: Path | None = None
    normalized_audio_path: Path | None = None
    srt_path: Path | None = None
    embedded_srt_path: Path | None = None
    manifest_path: Path | None = None

    @classmethod
    def initial(cls, context: DubRunContext) -> "DubRunState":
        artifact_paths = context.artifact_paths

        return cls(
            extracted_audio_path=artifact_paths.extracted_audio_path,
            source_transcript_path=artifact_paths.source_transcript_path,
            translated_transcript_path=artifact_paths.translated_transcript_path,
            adapted_transcript_path=artifact_paths.adapted_transcript_path,
            synthesized_transcript_path=artifact_paths.synthesized_transcript_path,
            speech_dir=artifact_paths.speech_dir,
            speech_track_path=artifact_paths.speech_track_path,
            dubbed_video_path=context.paths.output_path,
        )

    def artifacts(
        self,
        context: DubRunContext,
        *,
        stopped_at_checkpoint: DubCheckpoint | None = None,
    ) -> DubbingArtifacts:
        return DubbingArtifacts(
            workspace_dir=context.paths.workspace_dir,
            extracted_audio_path=self.extracted_audio_path,
            source_transcript_path=self.source_transcript_path,
            diarized_transcript_path=self.diarized_transcript_path,
            translated_transcript_path=self.translated_transcript_path,
            adapted_transcript_path=self.adapted_transcript_path,
            synthesized_transcript_path=self.synthesized_transcript_path,
            speech_dir=self.speech_dir,
            timing_repaired_transcript_path=self.timing_repaired_transcript_path,
            timing_repaired_speech_dir=self.timing_repaired_speech_dir,
            speech_track_path=self.speech_track_path,
            dubbed_video_path=self.dubbed_video_path,
            fitted_transcript_path=self.fitted_transcript_path,
            fitted_speech_dir=self.fitted_speech_dir,
            video_fitted_transcript_path=self.video_fitted_transcript_path,
            fitted_video_path=self.fitted_video_path,
            video_fitted_original_audio_path=self.video_fitted_original_audio_path,
            separated_background_audio_path=self.separated_background_audio_path,
            separated_voice_audio_path=self.separated_voice_audio_path,
            video_fitted_background_audio_path=self.video_fitted_background_audio_path,
            mix_original_audio_path=self.mix_original_audio_path,
            mixed_audio_path=self.mixed_audio_path,
            normalized_audio_path=self.normalized_audio_path,
            srt_path=self.srt_path,
            embedded_srt_path=self.embedded_srt_path,
            manifest_path=self.manifest_path,
            stopped_at_checkpoint=stopped_at_checkpoint,
        )
