from dublaro.adapters.dubbing_script import (
    DubbingScriptOptions,
    DubbingScriptResult,
)
from dublaro.pipeline.dubbing_script import generate_dubbing_script_transcripts
from dublaro.schemas import Segment, Transcript


class FakeDubbingScriptAdapter:
    name = "fake-dubbing-script"

    def generate_segment_script(
        self,
        segment: Segment,
        options: DubbingScriptOptions,
    ) -> DubbingScriptResult:
        return DubbingScriptResult(
            translated_text=f"{options.target_language}: {segment.source_text}",
            adapted_text=f"{options.target_language}: short",
            reason="test",
        )


def test_generate_dubbing_script_transcripts_writes_both_text_fields() -> None:
    transcript = Transcript(
        id="lesson",
        source_language="en",
        segments=[
            Segment(
                id="seg-1",
                start=0.0,
                end=1.0,
                source_text="Hello world.",
            )
        ],
    )

    result = generate_dubbing_script_transcripts(
        transcript,
        adapter=FakeDubbingScriptAdapter(),
        target_language="pl",
    )

    assert result.translated.target_language == "pl"
    assert result.adapted.target_language == "pl"
    assert result.translated.segments[0].translated_text == "pl: Hello world."
    assert result.translated.segments[0].adapted_text == ""
    assert result.adapted.segments[0].translated_text == "pl: Hello world."
    assert result.adapted.segments[0].adapted_text == "pl: short"
    assert result.adapted.segments[0].metadata["text_workflow"] == "llm-dubbing"
    assert result.adapted.segments[0].metadata["dubbing_script_reason"] == "test"
