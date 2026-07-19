from dataclasses import dataclass
from pathlib import Path

from dublaro.adapters.asr import AsrAdapter, TranscriptionOptions
from dublaro.adapters.text_adapter import TextAdapter
from dublaro.adapters.translation import TranslationAdapter
from dublaro.adapters.tts import TtsAdapter
from dublaro.audio.ffmpeg import extract_audio_from_video
from dublaro.pipeline.adapt_text import adapt_transcript_text
from dublaro.pipeline.align import build_speech_timeline
from dublaro.pipeline.export import export_dubbed_video
from dublaro.pipeline.synthesize import synthesize_transcript_speech
from dublaro.pipeline.transcribe import save_transcript, transcribe_audio
from dublaro.pipeline.translate import translate_transcript


@dataclass(frozen=True)
class DubbingArtifacts:
    workspace_dir: Path
    extracted_audio_path: Path
    source_transcript_path: Path
    translated_transcript_path: Path
    adapted_transcript_path: Path
    synthesized_transcript_path: Path
    speech_dir: Path
    speech_track_path: Path
    dubbed_video_path: Path


def dub_video(
    video_path: str | Path,
    output_path: str | Path,
    *,
    source_language: str | None,
    target_language: str,
    workspace_dir: str | Path,
    asr_adapter: AsrAdapter,
    translation_adapter: TranslationAdapter,
    text_adapter: TextAdapter,
    tts_adapter: TtsAdapter,
    asr_sample_rate: int = 16_000,
    speech_sample_rate: int = 24_000,
    overwrite: bool = False,
) -> DubbingArtifacts:
    video_file = Path(video_path)
    output_file = Path(output_path)
    workspace = Path(workspace_dir)
    workspace.mkdir(parents=True, exist_ok=True)

    source_label = source_language or "auto"
    stem = video_file.stem

    extracted_audio_path = workspace / f"{stem}.audio.wav"
    source_transcript_path = workspace / f"{stem}.{source_label}.json"
    translated_transcript_path = workspace / f"{stem}.{target_language}.json"
    adapted_transcript_path = workspace / f"{stem}.{target_language}.adapted.json"
    synthesized_transcript_path = (
        workspace / f"{stem}.{target_language}.synthesized.json"
    )
    speech_dir = workspace / f"{stem}.{target_language}.speech"
    speech_track_path = workspace / f"{stem}.{target_language}.speech-track.wav"

    extracted_audio_path = extract_audio_from_video(
        video_file,
        extracted_audio_path,
        sample_rate=asr_sample_rate,
        channels=1,
        overwrite=overwrite,
    )

    source_transcript = transcribe_audio(
        extracted_audio_path,
        adapter=asr_adapter,
        options=TranscriptionOptions(source_language=source_language),
    )
    save_transcript(source_transcript, source_transcript_path)

    translated_transcript = translate_transcript(
        source_transcript,
        adapter=translation_adapter,
        target_language=target_language,
        source_language=source_language,
    )
    save_transcript(translated_transcript, translated_transcript_path)

    adapted_transcript = adapt_transcript_text(
        translated_transcript,
        adapter=text_adapter,
        target_language=target_language,
        source_language=source_language,
    )
    save_transcript(adapted_transcript, adapted_transcript_path)

    synthesized_transcript = synthesize_transcript_speech(
        adapted_transcript,
        adapter=tts_adapter,
        output_dir=speech_dir,
        language=target_language,
        sample_rate=speech_sample_rate,
    )
    save_transcript(synthesized_transcript, synthesized_transcript_path)

    speech_track_path = build_speech_timeline(
        synthesized_transcript,
        output_path=speech_track_path,
        sample_rate=speech_sample_rate,
    )

    dubbed_video_path = export_dubbed_video(
        video_file,
        speech_track_path,
        output_file,
        overwrite=overwrite,
    )

    return DubbingArtifacts(
        workspace_dir=workspace,
        extracted_audio_path=extracted_audio_path,
        source_transcript_path=source_transcript_path,
        translated_transcript_path=translated_transcript_path,
        adapted_transcript_path=adapted_transcript_path,
        synthesized_transcript_path=synthesized_transcript_path,
        speech_dir=speech_dir,
        speech_track_path=speech_track_path,
        dubbed_video_path=dubbed_video_path,
    )
