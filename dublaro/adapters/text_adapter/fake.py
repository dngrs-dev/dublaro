from dublaro.adapters.text_adapter.base import TextAdaptationOptions
from dublaro.schemas import Segment


class FakeTextAdapter:
    name = "fake-text-adapter"

    def adapt_segment(
        self,
        segment: Segment,
        options: TextAdaptationOptions,
    ) -> str:
        text = segment.translated_text or segment.source_text
        return " ".join(text.split())
