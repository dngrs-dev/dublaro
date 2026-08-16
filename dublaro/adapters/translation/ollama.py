import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from dublaro.adapters.translation.base import TranslationOptions

DEFAULT_OLLAMA_TRANSLATION_MODEL = "llama3.1"
DEFAULT_OLLAMA_TRANSLATION_URL = "http://localhost:11434"
DEFAULT_OLLAMA_TRANSLATION_TIMEOUT_SECONDS = 120.0
DEFAULT_OLLAMA_TRANSLATION_TEMPERATURE = 0.1

_LABEL_PREFIX_RE = re.compile(
    r"^\s*(?:translated text|translation|output|result)\s*[:-]?\s+",
    re.IGNORECASE,
)

_CODE_FENCE_RE = re.compile(
    r"^\s*```(?:json)?\s*(.*?)\s*```\s*$",
    re.IGNORECASE | re.DOTALL,
)


@dataclass(frozen=True)
class OllamaTranslationResult:
    text: str
    reason: str | None = None


class OllamaTranslationAdapter:
    name = "ollama"

    def __init__(
        self,
        *,
        model: str = DEFAULT_OLLAMA_TRANSLATION_MODEL,
        url: str = DEFAULT_OLLAMA_TRANSLATION_URL,
        timeout_seconds: float = DEFAULT_OLLAMA_TRANSLATION_TIMEOUT_SECONDS,
        temperature: float = DEFAULT_OLLAMA_TRANSLATION_TEMPERATURE,
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

    def translate_text(self, text: str, options: TranslationOptions) -> str:
        return self.translate_text_result(text, options).text

    def translate_text_result(
        self,
        text: str,
        options: TranslationOptions,
    ) -> OllamaTranslationResult:
        source_text = _normalize_spacing(text)
        if not source_text:
            return OllamaTranslationResult(text="")

        result = self._generate_translation_result(_build_prompt(source_text, options))
        cleaned = _clean_translation_output(result.text)

        return OllamaTranslationResult(
            text=cleaned or source_text,
            reason=result.reason,
        )

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

    def _generate_translation_result(
        self,
        prompt: str,
    ) -> OllamaTranslationResult:
        payload: dict[str, object] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {"temperature": self.temperature},
        }

        response = self._request_json("/api/generate", method="POST", payload=payload)
        response_text = response.get("response")

        if not isinstance(response_text, str):
            raise ValueError("Ollama response did not include translated text.")

        structured = _parse_structured_translation_result(response_text)
        if structured is not None:
            return structured

        cleaned = _clean_translation_output(response_text)
        if not cleaned:
            raise ValueError("Ollama response did not include translated text.")

        return OllamaTranslationResult(text=cleaned)

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


def _build_prompt(text: str, options: TranslationOptions) -> str:
    source_language = options.source_language or "auto-detected"
    target_language = options.target_language

    return f"""You are translating text for video dubbing.

Return valid JSON only. No markdown, no labels, no code fences.
Use exactly this shape: {{"text": "...", "reason": "..."}}.

Rules for "text":
- Translate the source text into {target_language}.
- Preserve all meaning, questions, names, numbers, and intent.
- Use natural spoken language.
- Do not adapt for timing yet.
- Do not add explanations, labels, quotes, or markdown inside "text".

Rules for "reason":
- Write a short English explanation of translation choices.
- If the translation is straightforward, say that.

Source language: {source_language}
Target language: {target_language}

Source text:
{text}

JSON response:"""


def _parse_structured_translation_result(
    raw_text: str,
) -> OllamaTranslationResult | None:
    json_text = _extract_json_object(raw_text)
    if json_text is None:
        return None

    try:
        parsed = json.loads(json_text)
    except json.JSONDecodeError as error:
        raise ValueError(
            "Ollama returned invalid structured translation response."
        ) from error

    if not isinstance(parsed, dict):
        return None

    text = parsed.get("text")
    if not isinstance(text, str):
        raise ValueError('Ollama structured response did not include "text".')

    cleaned_text = _clean_translation_output(text)
    if not cleaned_text:
        raise ValueError('Ollama structured response included empty "text".')

    reason = parsed.get("reason")
    cleaned_reason = _clean_reason(reason) if isinstance(reason, str) else None

    return OllamaTranslationResult(text=cleaned_text, reason=cleaned_reason or None)


def _extract_json_object(raw_text: str) -> str | None:
    stripped = _strip_code_fence(raw_text).strip()

    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped

    start = stripped.find("{")
    end = stripped.rfind("}")

    if start == -1 or end <= start:
        return None

    return stripped[start : end + 1]


def _strip_code_fence(text: str) -> str:
    match = _CODE_FENCE_RE.match(text)
    if match is None:
        return text

    return match.group(1)


def _clean_translation_output(text: str) -> str:
    cleaned = _normalize_spacing(text)
    cleaned = _strip_wrapping_quotes(cleaned)
    cleaned = _LABEL_PREFIX_RE.sub("", cleaned, count=1).strip()
    cleaned = _strip_wrapping_quotes(cleaned)
    return _normalize_spacing(cleaned)


def _clean_reason(text: str) -> str:
    return _normalize_spacing(_strip_wrapping_quotes(text))


def _strip_wrapping_quotes(text: str) -> str:
    stripped = text.strip()

    if len(stripped) < 2:
        return stripped

    if stripped[0] == stripped[-1] and stripped[0] in {"'", '"'}:
        return stripped[1:-1].strip()

    return stripped


def _normalize_spacing(text: str) -> str:
    return " ".join(text.split())
