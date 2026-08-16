from collections.abc import Mapping

import pytest
from dublaro.adapters.translation.base import TranslationOptions
from dublaro.adapters.translation.ollama import OllamaTranslationAdapter


class RecordingOllamaTranslationAdapter(OllamaTranslationAdapter):
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


def test_ollama_translation_adapter_translates_text() -> None:
    adapter = RecordingOllamaTranslationAdapter(
        {
            "response": (
                '{"text": "Czesc swiecie.", "reason": "Straightforward greeting."}'
            )
        }
    )

    result = adapter.translate_text_result(
        "Hello world.",
        TranslationOptions(source_language="en", target_language="pl"),
    )

    assert result.text == "Czesc swiecie."
    assert result.reason == "Straightforward greeting."

    payload = adapter.requests[0]["payload"]
    assert isinstance(payload, dict)
    assert payload["model"] == "llama3.1"
    assert payload["format"] == "json"
    assert "Source language: en" in str(payload["prompt"])
    assert "Target language: pl" in str(payload["prompt"])


def test_ollama_translation_adapter_keeps_empty_text_empty() -> None:
    adapter = RecordingOllamaTranslationAdapter({"response": "unused"})

    assert (
        adapter.translate_text(
            "",
            TranslationOptions(source_language="en", target_language="pl"),
        )
        == ""
    )
    assert adapter.requests == []


def test_ollama_translation_adapter_rejects_missing_response_text() -> None:
    adapter = RecordingOllamaTranslationAdapter({})

    with pytest.raises(ValueError, match="translated text"):
        adapter.translate_text(
            "Hello world.",
            TranslationOptions(source_language="en", target_language="pl"),
        )


def test_ollama_translation_adapter_rejects_structured_response_without_text() -> None:
    adapter = RecordingOllamaTranslationAdapter({"response": '{"reason": "No text."}'})

    with pytest.raises(ValueError, match="structured response"):
        adapter.translate_text(
            "Hello world.",
            TranslationOptions(source_language="en", target_language="pl"),
        )


def test_ollama_translation_adapter_strips_fallback_label() -> None:
    adapter = RecordingOllamaTranslationAdapter(
        {"response": "Translated text Czesc swiecie."}
    )

    assert (
        adapter.translate_text(
            "Hello world.",
            TranslationOptions(source_language="en", target_language="pl"),
        )
        == "Czesc swiecie."
    )
