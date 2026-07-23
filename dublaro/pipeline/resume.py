from pathlib import Path

from dublaro.pipeline.transcribe import load_transcript
from dublaro.schemas import Transcript


def reusable_file(path: str | Path | None) -> bool:
    if path is None:
        return False

    file = Path(path)
    return file.exists() and file.is_file() and file.stat().st_size > 0


def load_reusable_transcript(path: str | Path) -> Transcript | None:
    if not reusable_file(path):
        return None

    try:
        return load_transcript(path)
    except Exception:
        return None


def load_reusable_synthesized_transcript(path: str | Path) -> Transcript | None:
    transcript = load_reusable_transcript(path)

    if transcript is None:
        return None

    if not generated_audio_files_exist(transcript):
        return None

    return transcript


def generated_audio_files_exist(transcript: Transcript) -> bool:
    for segment in transcript.segments:
        text = segment.adapted_text or segment.translated_text or segment.source_text

        if not text.strip():
            continue

        if not reusable_file(segment.generated_audio_path):
            return False

    return True
