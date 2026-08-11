from array import array
from pathlib import Path

from dublaro.adapters.text_adapter import TextAdaptationOptions, TextTimingRepairOptions
from dublaro.adapters.tts import SpeechSynthesisOptions
from dublaro.audio.wav import write_mono_pcm16_wav
from dublaro.pipeline.timing_repair import repair_overlong_speech_segments
from dublaro.schemas import Segment, Transcript


class RepairingTextAdapter:
    name = "repairing"

    def __init__(self) -> None:
        self.calls: list[TextTimingRepairOptions] = []

    def adapt_segment(
        self,
        segment: Segment,
        options: TextAdaptationOptions,
    ) -> str:
        return segment.translated_text

    def repair_segment_timing(
        self,
        segment: Segment,
        options: TextTimingRepairOptions,
    ) -> str:
        self.calls.append(options)
        return "Short text."


class LengthBasedTtsAdapter:
    name = "length-based"

    def synthesize_segment(
        self,
        segment: Segment,
        output_path: Path,
        options: SpeechSynthesisOptions,
    ) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)

        text = segment.adapted_text or segment.translated_text or segment.source_text
        duration_seconds = 1.0 if text == "Short text." else 2.0

        write_mono_pcm16_wav(
            output_path,
            samples=array("h", [0] * int(duration_seconds * options.sample_rate)),
            sample_rate=options.sample_rate,
        )
        return output_path


def test_repair_overlong_speech_segments_rewrites_and_resynthesizes(
    tmp_path: Path,
) -> None:
    original_audio_path = tmp_path / "seg-1.wav"
    write_mono_pcm16_wav(
        original_audio_path,
        samples=array("h", [0] * 16_000),
        sample_rate=8_000,
    )

    text_adapter = RepairingTextAdapter()

    result = repair_overlong_speech_segments(
        Transcript(
            id="audio",
            source_language="en",
            target_language="pl",
            segments=[
                Segment(
                    id="seg-1",
                    start=0.0,
                    end=1.0,
                    source_text="That is everything there is to say.",
                    translated_text="To wszystko, co mozna powiedziec.",
                    adapted_text="To jest juz wszystko, co mozna powiedziec.",
                    generated_audio_path=str(original_audio_path),
                )
            ],
        ),
        text_adapter=text_adapter,
        tts_adapter=LengthBasedTtsAdapter(),
        output_dir=tmp_path / "repair",
        language="pl",
        sample_rate=8_000,
        max_attempts=2,
        target_speedup=1.15,
        min_overrun_seconds=0.05,
    )

    segment = result.segments[0]

    assert text_adapter.calls
    assert segment.adapted_text == "Short text."
    assert segment.generated_audio_path == str(
        tmp_path / "repair" / "seg-1.repair-1.wav"
    )
    assert segment.metadata["timing_repair_status"] == "repaired"
    assert segment.metadata["timing_repair_required_speedup_before"] == "2"
    assert segment.metadata["timing_repair_required_speedup_after"] == "1"
    assert result.metadata["timing_repair_attempted_segments"] == "1"
    assert result.metadata["timing_repair_repaired_segments"] == "1"


def test_repair_overlong_speech_segments_skips_segments_inside_target(
    tmp_path: Path,
) -> None:
    original_audio_path = tmp_path / "seg-1.wav"
    write_mono_pcm16_wav(
        original_audio_path,
        samples=array("h", [0] * 8_800),
        sample_rate=8_000,
    )

    text_adapter = RepairingTextAdapter()

    result = repair_overlong_speech_segments(
        Transcript(
            id="audio",
            source_language="en",
            target_language="pl",
            segments=[
                Segment(
                    id="seg-1",
                    start=0.0,
                    end=1.0,
                    adapted_text="Already fine.",
                    generated_audio_path=str(original_audio_path),
                )
            ],
        ),
        text_adapter=text_adapter,
        tts_adapter=LengthBasedTtsAdapter(),
        output_dir=tmp_path / "repair",
        language="pl",
        sample_rate=8_000,
        target_speedup=1.15,
        min_overrun_seconds=0.05,
    )

    assert text_adapter.calls == []
    assert result.segments[0].generated_audio_path == str(original_audio_path)
    assert result.metadata["timing_repair_attempted_segments"] == "0"
