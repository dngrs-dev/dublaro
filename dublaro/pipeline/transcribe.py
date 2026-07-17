from pathlib import Path

from dublaro.adapters.asr.base import AsrAdapter, TranscriptionOptions
from dublaro.schemas.transcript import Transcript


def transcribe_audio(
    audio_path: str | Path,
    adapter: AsrAdapter,
    options: TranscriptionOptions | None = None,
) -> Transcript:
    audio_file = Path(audio_path)

    if not audio_file.exists():
        raise FileNotFoundError(f"Audio file does not exist: {audio_file}")

    if not audio_file.is_file():
        raise ValueError(f"Audio path is not a file: {audio_file}")

    return adapter.transcribe(
        audio_file,
        options or TranscriptionOptions(),
    )


def default_transcript_path(audio_path: str | Path) -> Path:
    audio_file = Path(audio_path)
    return audio_file.with_suffix(".transcript.json")


def save_transcript(
    transcript: Transcript,
    output_path: str | Path,
) -> Path:
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(
        transcript.model_dump_json(indent=2),
        encoding="utf-8",
    )
    return output_file


def load_transcript(path: str | Path) -> Transcript:
    transcript_file = Path(path)
    return Transcript.model_validate_json(transcript_file.read_text(encoding="utf-8"))
