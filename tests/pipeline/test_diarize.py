from dublaro.adapters.diarization import DiarizationTurn
from dublaro.pipeline.diarize import assign_speakers_to_transcript
from dublaro.schemas import Segment, Transcript


def test_assign_speakers_to_transcript_uses_largest_overlap() -> None:
    transcript = Transcript(
        id="audio",
        source_language="en",
        segments=[
            Segment(id="seg-0001", start=0.0, end=2.0, source_text="Hello"),
            Segment(id="seg-0002", start=2.0, end=4.0, source_text="Again"),
        ],
    )

    diarized = assign_speakers_to_transcript(
        transcript,
        [
            DiarizationTurn(start=0.0, end=1.5, speaker="speaker-1"),
            DiarizationTurn(start=1.5, end=4.0, speaker="speaker-2"),
        ],
        adapter_name="fake-diarization",
    )

    assert diarized.segments[0].speaker == "speaker-1"
    assert diarized.segments[1].speaker == "speaker-2"
    assert diarized.metadata["diarization_adapter"] == "fake-diarization"
    assert diarized.metadata["diarization_speaker_count"] == "2"
