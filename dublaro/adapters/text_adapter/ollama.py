import json
from collections.abc import Mapping
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from dublaro.adapters.text_adapter.base import TextAdaptationOptions
from dublaro.schemas import Segment

DEFAULT_OLLAMA_MODEL = "llama3.1"
DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_OLLAMA_TIMEOUT_SECONDS = 30.0
DEFAULT_OLLAMA_PREFLIGHT_TIMEOUT_SECONDS = 5.0
DEFAULT_OLLAMA_TEMPERATURE = 0.2


class OllamaTextAdapter:
    name = "ollama"

    def __init__(
        self,
        *,
        model: str = DEFAULT_OLLAMA_MODEL,
        url: str = DEFAULT_OLLAMA_URL,
        timeout_seconds: float = DEFAULT_OLLAMA_TIMEOUT_SECONDS,
        temperature: float = DEFAULT_OLLAMA_TEMPERATURE,
    ) -> None:
        model = model.strip()
        url = url.strip().rstrip("/")

        if not model:
            raise ValueError("Ollama model cannot be empty.")
        if not url:
            raise ValueError("Ollama URL cannot be empty.")
        if timeout_seconds <= 0:
            raise ValueError("Ollama timeout must be greater than 0.")
        if temperature < 0 or temperature > 2:
            raise ValueError("Ollama temperature must be between 0 and 2.")

        self.model = model
        self.url = url
        self.timeout_seconds = timeout_seconds
        self.temperature = temperature

    def adapt_segment(
        self,
        segment: Segment,
        options: TextAdaptationOptions,
    ) -> str:
        translated_text = _normalize_spacing(
            segment.translated_text or segment.source_text
        )
        if not translated_text:
            return ""

        payload: dict[str, object] = {
            "model": self.model,
            "prompt": _build_prompt(segment, options, translated_text),
            "stream": False,
            "options": {"temperature": self.temperature},
        }

        response = self._request_json("/api/generate", method="POST", payload=payload)
        adapted_text = response.get("response")

        if not isinstance(adapted_text, str):
            raise ValueError("Ollama response did not include adapted text.")

        cleaned = _clean_llm_output(adapted_text)
        return cleaned or translated_text

    def available_models(self) -> tuple[str, ...]:
        response = self._request_json("/api/tags", method="GET")
        models = response.get("models")

        if not isinstance(models, list):
            return ()

        names: list[str] = []
        for model in models:
            if not isinstance(model, dict):
                continue

            name = model.get("name")
            if isinstance(name, str):
                names.append(name)

        return tuple(names)

    def _request_json(
        self,
        endpoint: str,
        *,
        method: str,
        payload: Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        headers = {"Accept": "application/json"}

        if data is not None:
            headers["Content-Type"] = "application/json"

        request = Request(
            f"{self.url}{endpoint}",
            data=data,
            headers=headers,
            method=method,
        )

        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                raw_response = response.read().decode("utf-8")
        except HTTPError as error:
            body = error.read().decode("utf-8", errors="replace").strip()
            message = f"Ollama request failed with HTTP {error.code}"
            if body:
                message = f"{message}: {body}"
            raise ValueError(message) from error
        except (OSError, TimeoutError, URLError) as error:
            raise ValueError(
                f"Could not connect to Ollama at {self.url}: {error}"
            ) from error

        try:
            parsed = json.loads(raw_response)
        except json.JSONDecodeError as error:
            raise ValueError("Ollama returned invalid JSON.") from error

        if not isinstance(parsed, dict):
            raise ValueError("Ollama returned an unexpected response.")

        return parsed


def check_ollama_model_available(
    *,
    model: str,
    url: str = DEFAULT_OLLAMA_URL,
    timeout_seconds: float = DEFAULT_OLLAMA_PREFLIGHT_TIMEOUT_SECONDS,
) -> bool:
    adapter = OllamaTextAdapter(
        model=model,
        url=url,
        timeout_seconds=timeout_seconds,
    )
    return any(
        _model_matches(model, available) for available in adapter.available_models()
    )


def _build_prompt(
    segment: Segment,
    options: TextAdaptationOptions,
    translated_text: str,
) -> str:
    budget = _character_budget(segment, options)
    source_text = _normalize_spacing(segment.source_text)
    source_language = options.source_language or segment.source_language or "unknown"
    target_language = options.target_language or segment.target_language or "unknown"
    budget_text = "no hard budget" if budget is None else f"{budget} characters"

    return f"""You adapt translated dubbing text so it can be spoken naturally within timing.

Rules:
- Return only the adapted {target_language} text.
- Preserve the meaning, questions, names, and numbers.
- Prefer shorter natural wording.
- Do not delete a whole sentence unless it is truly unavoidable.
- Do not add explanations, labels, quotes, or markdown.

Source language: {source_language}
Target language: {target_language}
Segment duration: {segment.duration:.2f} seconds
Character budget: {budget_text}

Original source text:
{source_text}

Translated text:
{translated_text}

Adapted text:"""


def _character_budget(
    segment: Segment,
    options: TextAdaptationOptions,
) -> int | None:
    if segment.duration <= 0:
        return None

    return max(8, int(segment.duration * options.max_chars_per_second))


def _clean_llm_output(text: str) -> str:
    cleaned = _normalize_spacing(text)
    cleaned = _strip_label(cleaned)

    if len(cleaned) >= 2 and cleaned[0] == cleaned[-1] and cleaned[0] in {"'", '"'}:
        cleaned = cleaned[1:-1].strip()

    return cleaned


def _strip_label(text: str) -> str:
    lowered = text.lower()

    for label in ("adapted text:", "adapted:", "output:", "result:"):
        if lowered.startswith(label):
            return text[len(label) :].strip()

    return text


def _normalize_spacing(text: str) -> str:
    return " ".join(text.split())


def _model_matches(configured_model: str, available_model: str) -> bool:
    configured = configured_model.strip()
    available = available_model.strip()

    return (
        available == configured
        or available == f"{configured}:latest"
        or available.split(":", 1)[0] == configured
    )
