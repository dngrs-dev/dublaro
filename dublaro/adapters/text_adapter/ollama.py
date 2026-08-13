import json
import re
from collections.abc import Mapping
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from dublaro.adapters.text_adapter.base import (
    TextAdaptationOptions,
    TextAdapterResult,
    TextTimingRepairOptions,
)
from dublaro.schemas import Segment

DEFAULT_OLLAMA_MODEL = "llama3.1"
DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_OLLAMA_TIMEOUT_SECONDS = 30.0
DEFAULT_OLLAMA_PREFLIGHT_TIMEOUT_SECONDS = 5.0
DEFAULT_OLLAMA_TEMPERATURE = 0.2

_LABEL_PREFIX_RE = re.compile(
    r"^\s*"
    r"(?:shorter repaired translated text|"
    r"shorter repaired spoken text|"
    r"shorter repaired text|"
    r"repaired translated text|"
    r"repaired spoken text|"
    r"repaired text|"
    r"translated text|"
    r"spoken text|"
    r"adapted text|"
    r"repaired|"
    r"adapted|"
    r"output|"
    r"result)"
    r"\s*[:-]?\s+",
    re.IGNORECASE,
)

_CODE_FENCE_RE = re.compile(
    r"^\s*```(?:json)?\s*(.*?)\s*```\s*$",
    re.IGNORECASE | re.DOTALL,
)


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
        return self.adapt_segment_result(segment, options).text

    def adapt_segment_result(
        self,
        segment: Segment,
        options: TextAdaptationOptions,
    ) -> TextAdapterResult:
        translated_text = _normalize_spacing(
            segment.translated_text or segment.source_text
        )
        if not translated_text:
            return TextAdapterResult()

        result = self._generate_text_result(
            _build_prompt(segment, options, translated_text)
        )
        cleaned = _clean_llm_output(result.text)

        return TextAdapterResult(
            text=cleaned or translated_text,
            reason=result.reason,
        )

    def repair_segment_timing(
        self,
        segment: Segment,
        options: TextTimingRepairOptions,
    ) -> str:
        return self.repair_segment_timing_result(segment, options).text

    def repair_segment_timing_result(
        self,
        segment: Segment,
        options: TextTimingRepairOptions,
    ) -> TextAdapterResult:
        current_text = _normalize_spacing(
            segment.adapted_text or segment.translated_text or segment.source_text
        )
        if not current_text:
            return TextAdapterResult()

        translated_text = _normalize_spacing(segment.translated_text)
        result = self._generate_text_result(
            _build_timing_repair_prompt(
                segment,
                options,
                current_text=current_text,
                translated_text=translated_text,
            )
        )
        cleaned = _clean_llm_output(result.text)

        return TextAdapterResult(
            text=cleaned or current_text,
            reason=result.reason,
        )

    def _generate_text_result(self, prompt: str) -> TextAdapterResult:
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
            raise ValueError("Ollama response did not include adapted text.")

        structured = _parse_structured_text_result(response_text)
        if structured is not None:
            return structured

        cleaned = _clean_llm_output(response_text)
        if not cleaned:
            raise ValueError("Ollama response did not include adapted text.")

        return TextAdapterResult(text=cleaned)

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

Return valid JSON only. No markdown, no labels, no code fences.
Use exactly this shape: {{"text": "...", "reason": "..."}}.

Rules for "text":
- It must contain only the adapted {target_language} text.
- Preserve the meaning, questions, names, and numbers.
- Prefer shorter natural wording.
- Do not delete a whole sentence unless it is truly unavoidable.
- Do not add explanations, labels, quotes, or markdown inside "text".

Rules for "reason":
- Write a short English explanation of what changed.
- If nothing changed, explain why.

Source language: {source_language}
Target language: {target_language}
Segment duration: {segment.duration:.2f} seconds
Character budget: {budget_text}

Original source text:
{source_text}

Translated text:
{translated_text}

JSON response:"""


def _build_timing_repair_prompt(
    segment: Segment,
    options: TextTimingRepairOptions,
    *,
    current_text: str,
    translated_text: str,
) -> str:
    source_text = _normalize_spacing(segment.source_text)
    source_language = options.source_language or segment.source_language or "unknown"
    target_language = options.target_language or segment.target_language or "unknown"

    return f"""You repair dubbing text after TTS generated audio that is too long.

Return valid JSON only. No markdown, no labels, no code fences.
Use exactly this shape: {{"text": "...", "reason": "..."}}.

Goal:
- Rewrite the current {target_language} text so generated speech becomes shorter.
- Avoid video slowdown.
- Keep the result natural enough for dubbing.

Rules for "text":
- It must contain only the shorter repaired {target_language} text.
- Preserve the core meaning, questions, names, and numbers.
- Make it shorter and easier to speak.
- Do not add new meaning.
- Do not add explanations, labels, quotes, or markdown inside "text".

Rules for "reason":
- Write a short English explanation of why this rewrite should fit better.

Source language: {source_language}
Target language: {target_language}
Attempt: {options.attempt} of {options.max_attempts}
Segment duration: {options.target_duration_seconds:.2f} seconds
Current generated audio duration: {options.current_audio_duration_seconds:.2f} seconds
Target maximum audio duration: {options.max_audio_duration_seconds:.2f} seconds

Original source text:
{source_text}

Translated text:
{translated_text}

Current spoken text:
{current_text}

JSON response:"""


def _character_budget(
    segment: Segment,
    options: TextAdaptationOptions,
) -> int | None:
    if segment.duration <= 0:
        return None

    return max(8, int(segment.duration * options.max_chars_per_second))


def _parse_structured_text_result(raw_text: str) -> TextAdapterResult | None:
    json_text = _extract_json_object(raw_text)
    if json_text is None:
        return None

    try:
        parsed = json.loads(json_text)
    except json.JSONDecodeError as error:
        raise ValueError("Ollama returned invalid structured text response.") from error

    if not isinstance(parsed, dict):
        return None

    text = parsed.get("text")
    if not isinstance(text, str):
        raise ValueError('Ollama structured response did not include "text".')

    cleaned_text = _clean_llm_output(text)
    if not cleaned_text:
        raise ValueError('Ollama structured response included empty "text".')

    reason = parsed.get("reason")
    cleaned_reason = _clean_llm_reason(reason) if isinstance(reason, str) else None

    return TextAdapterResult(text=cleaned_text, reason=cleaned_reason or None)


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


def _clean_llm_reason(text: str) -> str:
    cleaned = _normalize_spacing(text)
    cleaned = _strip_wrapping_quotes(cleaned)
    return _normalize_spacing(cleaned)


def _clean_llm_output(text: str) -> str:
    cleaned = _normalize_spacing(text)
    cleaned = _strip_wrapping_quotes(cleaned)
    cleaned = _strip_label(cleaned)
    cleaned = _strip_wrapping_quotes(cleaned)

    return _normalize_spacing(cleaned)


def _strip_label(text: str) -> str:
    previous = text

    while True:
        stripped = _LABEL_PREFIX_RE.sub("", previous, count=1).strip()
        if stripped == previous:
            return stripped

        previous = stripped


def _strip_wrapping_quotes(text: str) -> str:
    stripped = text.strip()

    if len(stripped) < 2:
        return stripped

    if stripped[0] == stripped[-1] and stripped[0] in {"'", '"'}:
        return stripped[1:-1].strip()

    return stripped


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
