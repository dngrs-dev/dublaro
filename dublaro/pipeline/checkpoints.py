from typing import Final, Literal, cast

DubCheckpoint = Literal[
    "audio",
    "transcribed",
    "diarized",
    "translated",
    "adapted",
    "synthesized",
    "timing-repaired",
    "fitted",
    "video-fitted",
    "aligned",
    "mixed",
    "normalized",
    "subtitles",
    "exported",
    "manifest",
]

DUB_CHECKPOINTS: Final[tuple[DubCheckpoint, ...]] = (
    "audio",
    "transcribed",
    "diarized",
    "translated",
    "adapted",
    "synthesized",
    "timing-repaired",
    "fitted",
    "video-fitted",
    "aligned",
    "mixed",
    "normalized",
    "subtitles",
    "exported",
    "manifest",
)

DUB_CHECKPOINT_ORDER: Final[dict[DubCheckpoint, int]] = {
    checkpoint: index for index, checkpoint in enumerate(DUB_CHECKPOINTS)
}

DUB_CHECKPOINT_ARTIFACT_NAMES: Final[dict[DubCheckpoint, str]] = {
    "audio": "extracted_audio_path",
    "transcribed": "source_transcript_path",
    "diarized": "diarized_transcript_path",
    "translated": "translated_transcript_path",
    "adapted": "adapted_transcript_path",
    "synthesized": "synthesized_transcript_path",
    "timing-repaired": "timing_repaired_transcript_path",
    "fitted": "fitted_transcript_path",
    "video-fitted": "video_fitted_transcript_path",
    "aligned": "speech_track_path",
    "mixed": "mixed_audio_path",
    "normalized": "normalized_audio_path",
    "subtitles": "srt_path",
    "exported": "dubbed_video_path",
    "manifest": "manifest_path",
}


def dub_checkpoint_artifact_name(checkpoint: DubCheckpoint) -> str:
    return DUB_CHECKPOINT_ARTIFACT_NAMES[checkpoint]


def format_dub_checkpoints() -> str:
    return ", ".join(DUB_CHECKPOINTS)


def parse_dub_checkpoint(value: str) -> DubCheckpoint:
    if value in DUB_CHECKPOINT_ORDER:
        return cast(DubCheckpoint, value)

    raise ValueError(f"Checkpoint must be one of: {format_dub_checkpoints()}.")


def will_reach_checkpoint(
    until_checkpoint: DubCheckpoint | None,
    checkpoint: DubCheckpoint,
) -> bool:
    if until_checkpoint is None:
        return True

    return DUB_CHECKPOINT_ORDER[until_checkpoint] >= DUB_CHECKPOINT_ORDER[checkpoint]
