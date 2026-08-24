from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from dublaro.pipeline.checkpoints import DubCheckpoint, checkpoint_between

PreflightSeverity = Literal["error", "warning"]


@dataclass(frozen=True)
class PreflightIssue:
    severity: PreflightSeverity
    code: str
    message: str
    hint: str | None = None


@dataclass(frozen=True)
class SpeakerVoicePreflightSettings:
    tts_backend: str
    piper_model_path: str | Path | None = None
    piper_config_path: str | Path | None = None
    piper_executable: str = "piper"


@dataclass(frozen=True)
class PreflightScope:
    translate_text: bool = True
    adapt_text: bool = True
    synthesize_speech: bool = True
    mix_audio: bool = True
    export_video: bool = True
    export_srt: bool = True
    write_manifest: bool = True

    @classmethod
    def from_until_checkpoint(
        cls,
        until_checkpoint: DubCheckpoint | None,
    ) -> "PreflightScope":
        return cls.from_checkpoints(
            start_from_checkpoint=None,
            until_checkpoint=until_checkpoint,
        )

    @classmethod
    def from_checkpoints(
        cls,
        *,
        start_from_checkpoint: DubCheckpoint | None,
        until_checkpoint: DubCheckpoint | None,
    ) -> "PreflightScope":
        return cls(
            translate_text=checkpoint_between(
                start_from_checkpoint=start_from_checkpoint,
                until_checkpoint=until_checkpoint,
                checkpoint="translated",
            ),
            adapt_text=checkpoint_between(
                start_from_checkpoint=start_from_checkpoint,
                until_checkpoint=until_checkpoint,
                checkpoint="adapted",
            ),
            synthesize_speech=checkpoint_between(
                start_from_checkpoint=start_from_checkpoint,
                until_checkpoint=until_checkpoint,
                checkpoint="synthesized",
            ),
            mix_audio=checkpoint_between(
                start_from_checkpoint=start_from_checkpoint,
                until_checkpoint=until_checkpoint,
                checkpoint="mixed",
            ),
            export_video=checkpoint_between(
                start_from_checkpoint=start_from_checkpoint,
                until_checkpoint=until_checkpoint,
                checkpoint="exported",
            ),
            export_srt=checkpoint_between(
                start_from_checkpoint=start_from_checkpoint,
                until_checkpoint=until_checkpoint,
                checkpoint="subtitles",
            ),
            write_manifest=checkpoint_between(
                start_from_checkpoint=start_from_checkpoint,
                until_checkpoint=until_checkpoint,
                checkpoint="manifest",
            ),
        )


@dataclass(frozen=True)
class DubPreflightReport:
    issues: tuple[PreflightIssue, ...]

    @property
    def errors(self) -> list[PreflightIssue]:
        return [issue for issue in self.issues if issue.severity == "error"]

    @property
    def warnings(self) -> list[PreflightIssue]:
        return [issue for issue in self.issues if issue.severity == "warning"]

    @property
    def has_errors(self) -> bool:
        return bool(self.errors)

    @property
    def passed(self) -> bool:
        return not self.has_errors
