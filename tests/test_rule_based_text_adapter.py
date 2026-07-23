from dublaro.adapters.text_adapter import RuleBasedTextAdapter
from dublaro.adapters.text_adapter.base import TextAdaptationOptions
from dublaro.schemas import Segment


def test_rule_based_text_adapter_keeps_text_that_fits() -> None:
    adapter = RuleBasedTextAdapter()

    result = adapter.adapt_segment(
        Segment(
            id="seg-0001",
            start=0.0,
            end=2.0,
            translated_text="Short sentence.",
        ),
        TextAdaptationOptions(target_language="en", max_chars_per_second=20.0),
    )

    assert result == "Short sentence."


def test_rule_based_text_adapter_shortens_text_to_timing_budget() -> None:
    adapter = RuleBasedTextAdapter()

    result = adapter.adapt_segment(
        Segment(
            id="seg-0001",
            start=0.0,
            end=1.0,
            translated_text="This is a very long sentence that does not fit",
        ),
        TextAdaptationOptions(target_language="en", max_chars_per_second=20.0),
    )

    assert result == "This is a very long"
    assert len(result) <= 20


def test_rule_based_text_adapter_removes_low_value_phrases() -> None:
    adapter = RuleBasedTextAdapter()

    result = adapter.adapt_segment(
        Segment(
            id="seg-0001",
            start=0.0,
            end=1.0,
            translated_text="You know, this is important",
        ),
        TextAdaptationOptions(target_language="en", max_chars_per_second=18.0),
    )

    assert result == "this is important"
