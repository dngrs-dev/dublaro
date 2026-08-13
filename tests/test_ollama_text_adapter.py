from collections.abc import Mapping

import pytest
from dublaro.adapters.text_adapter.base import (
    TextAdaptationOptions,
    TextTimingRepairOptions,
)
from dublaro.adapters.text_adapter.ollama import (
    OllamaTextAdapter,
    check_ollama_model_available,
)
from dublaro.schemas import Segment


class RecordingOllamaTextAdapter(OllamaTextAdapter):
    def __init__(self, response: dict[str, object]) -> None:
        super().__init__(model="llama3.1", url="http://ollama.local:11434")
        self.response = response
        self.requests: list[dict[str, object]] = []

    def _request_json(
        self,
        endpoint: str,
        *,
        method: str,
        payload: Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        self.requests.append(
            {
                "endpoint": endpoint,
                "method": method,
                "payload": dict(payload or {}),
            }
        )
        return self.response


def test_ollama_text_adapter_adapts_segment_text() -> None:
    adapter = RecordingOllamaTextAdapter(
        {
            "response": (
                '{"text": "Czesc. Lubisz je?", "reason": "Shortened repeated wording."}'
            )
        }
    )

    result = adapter.adapt_segment_result(
        Segment(
            id="seg-1",
            start=0.0,
            end=1.0,
            source_text="Let's talk about trees. Do you like trees?",
            translated_text="Porozmawiajmy o drzewach. Lubisz drzewa?",
        ),
        TextAdaptationOptions(
            source_language="en",
            target_language="pl",
            max_chars_per_second=20.0,
        ),
    )

    assert result.text == "Czesc. Lubisz je?"
    assert result.reason == "Shortened repeated wording."

    payload = adapter.requests[0]["payload"]
    assert isinstance(payload, dict)
    assert payload["model"] == "llama3.1"
    assert payload["format"] == "json"
    assert "Character budget: 20 characters" in str(payload["prompt"])


def test_ollama_text_adapter_rejects_missing_response_text() -> None:
    adapter = RecordingOllamaTextAdapter({})

    with pytest.raises(ValueError, match="adapted text"):
        adapter.adapt_segment(
            Segment(id="seg-1", start=0.0, end=1.0, translated_text="Hello"),
            TextAdaptationOptions(target_language="pl"),
        )


def test_ollama_text_adapter_rejects_structured_response_without_text() -> None:
    adapter = RecordingOllamaTextAdapter({"response": '{"reason": "No text."}'})

    with pytest.raises(ValueError, match="structured response"):
        adapter.adapt_segment(
            Segment(id="seg-1", start=0.0, end=1.0, translated_text="Hello"),
            TextAdaptationOptions(target_language="pl"),
        )


def test_ollama_text_adapter_reads_available_models() -> None:
    adapter = RecordingOllamaTextAdapter(
        {"models": [{"name": "llama3.1:latest"}, {"name": "mistral"}]}
    )

    assert adapter.available_models() == ("llama3.1:latest", "mistral")


def test_check_ollama_model_available_accepts_latest_tag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_available_models(self: OllamaTextAdapter) -> tuple[str, ...]:
        return ("llama3.1:latest",)

    monkeypatch.setattr(OllamaTextAdapter, "available_models", fake_available_models)

    assert check_ollama_model_available(
        model="llama3.1",
        url="http://ollama.local:11434",
        timeout_seconds=1.0,
    )


def test_ollama_text_adapter_strips_timing_repair_label() -> None:
    adapter = RecordingOllamaTextAdapter(
        {"response": "Repaired translated text to jest fajnie"}
    )

    result = adapter.repair_segment_timing(
        Segment(
            id="seg-1",
            start=0.0,
            end=1.0,
            source_text="And that's cool.",
            translated_text="I to jest super.",
            adapted_text="To jest fajnie.",
        ),
        TextTimingRepairOptions(
            source_language="en",
            target_language="pl",
            target_duration_seconds=1.0,
            current_audio_duration_seconds=1.5,
            max_audio_duration_seconds=1.15,
            attempt=1,
            max_attempts=2,
        ),
    )

    assert result == "to jest fajnie"
