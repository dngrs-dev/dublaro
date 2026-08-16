from collections.abc import Mapping

import pytest
from dublaro.adapters.dubbing_script import DubbingScriptOptions
from dublaro.adapters.dubbing_script.ollama import OllamaDubbingScriptAdapter
from dublaro.schemas import Segment


class RecordingOllamaDubbingScriptAdapter(OllamaDubbingScriptAdapter):
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


def test_ollama_dubbing_script_adapter_generates_translated_and_adapted_text() -> None:
    adapter = RecordingOllamaDubbingScriptAdapter(
        {
            "response": (
                '{"translated_text": "Porozmawiajmy o drzewach. Lubisz drzewa?", '
                '"adapted_text": "Pogadajmy o drzewach. Lubisz je?", '
                '"reason": "Shortened repeated wording."}'
            )
        }
    )

    result = adapter.generate_segment_script(
        Segment(
            id="seg-1",
            start=0.0,
            end=2.0,
            source_text="Let's talk about trees. Do you like trees?",
        ),
        DubbingScriptOptions(source_language="en", target_language="pl"),
    )

    assert result.translated_text == "Porozmawiajmy o drzewach. Lubisz drzewa?"
    assert result.adapted_text == "Pogadajmy o drzewach. Lubisz je?"
    assert result.reason == "Shortened repeated wording."

    payload = adapter.requests[0]["payload"]
    assert isinstance(payload, dict)
    assert payload["format"] == "json"
    assert "translated_text" in str(payload["prompt"])
    assert "adapted_text" in str(payload["prompt"])


def test_ollama_dubbing_script_adapter_rejects_missing_adapted_text() -> None:
    adapter = RecordingOllamaDubbingScriptAdapter(
        {"response": '{"translated_text": "Czesc."}'}
    )

    with pytest.raises(ValueError, match="adapted_text"):
        adapter.generate_segment_script(
            Segment(id="seg-1", start=0.0, end=1.0, source_text="Hello."),
            DubbingScriptOptions(source_language="en", target_language="pl"),
        )
