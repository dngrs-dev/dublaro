import json
import re

from dublaro.adapters.dubbing_script.base import (
    DubbingScriptOptions,
    DubbingScriptResult,
)
from dublaro.adapters.translation.ollama import (
    DEFAULT_OLLAMA_TRANSLATION_MODEL,
    DEFAULT_OLLAMA_TRANSLATION_TEMPERATURE,
    DEFAULT_OLLAMA_TRANSLATION_TIMEOUT_SECONDS,
    DEFAULT_OLLAMA_TRANSLATION_URL,
    OllamaTranslationAdapter,
)
from dublaro.schemas import Segment

_LABEL_PREFIX_RE = re.compile(
    r"^\s*(?:translated text|adapted text|spoken text|output|result)\s*[:-]?\s+",
    re.IGNORECASE,
)

_CODE_FENCE_RE = re.compile(
    r"^\s*```(?:json)?\s*(.*?)\s*```\s*$",
    re.IGNORECASE | re.DOTALL,
)


class OllamaDubbingScriptAdapter(OllamaTranslationAdapter):
    name = "ollama-dubbing-script"

    def __init__(
        self,
        *,
        model: str = DEFAULT_OLLAMA_TRANSLATION_MODEL,
        url: str = DEFAULT_OLLAMA_TRANSLATION_URL,
        timeout_seconds: float = DEFAULT_OLLAMA_TRANSLATION_TIMEOUT_SECONDS,
        temperature: float = DEFAULT_OLLAMA_TRANSLATION_TEMPERATURE,
    ) -> None:
        super().__init__(
            model=model,
            url=url,
            timeout_seconds=timeout_seconds,
            temperature=temperature,
        )

    def generate_segment_script(
        self,
        segment: Segment,
        options: DubbingScriptOptions,
    ) -> DubbingScriptResult:
        source_text = _normalize_spacing(segment.source_text)
        if not source_text:
            return DubbingScriptResult()

        payload: dict[str, object] = {
            "model": self.model,
            "prompt": _build_prompt(segment, options),
            "stream": False,
            "format": "json",
            "options": {"temperature": self.temperature},
        }

        response = self._request_json("/api/generate", method="POST", payload=payload)
        response_text = response.get("response")

        if not isinstance(response_text, str):
            raise ValueError("Ollama response did not include dubbing script text.")

        result = _parse_structured_result(response_text)
        if result is None:
            raise ValueError("Ollama returned invalid dubbing script JSON.")

        return result


def _build_prompt(segment: Segment, options: DubbingScriptOptions) -> str:
    source_language = (
        options.source_language or segment.source_language or "auto-detected"
    )
    target_language = options.target_language or segment.target_language or "unknown"
    budget = _character_budget(segment, options)
    budget_text = "no hard budget" if budget is None else f"{budget} characters"

    return f"""You create dubbing script text for translated video speech.

Return valid JSON only. No markdown, no labels, no code fences.
Use exactly this shape:
{{"translated_text": "...", "adapted_text": "...", "reason": "..."}}

Rules for "translated_text":
- Translate the source text into {target_language}.
- Preserve meaning, questions, names, numbers, and intent.
- Use natural spoken language.

Rules for "adapted_text":
- Rewrite the translation so it can be spoken naturally within the segment timing.
- Keep the core meaning.
- Prefer short natural wording.
- Do not delete a whole question or sentence unless unavoidable.
- Do not add explanations, labels, quotes, or markdown.

Rules for "reason":
- Write a short English explanation of the wording/timing choice.

Source language: {source_language}
Target language: {target_language}
Segment duration: {segment.duration:.2f} seconds
Adapted text budget: {budget_text}

Source text:
{_normalize_spacing(segment.source_text)}

JSON response:"""


def _parse_structured_result(raw_text: str) -> DubbingScriptResult | None:
    json_text = _extract_json_object(raw_text)
    if json_text is None:
        return None

    try:
        parsed = json.loads(json_text)
    except json.JSONDecodeError as error:
        raise ValueError(
            "Ollama returned invalid structured dubbing response."
        ) from error

    if not isinstance(parsed, dict):
        return None

    translated_text = parsed.get("translated_text")
    adapted_text = parsed.get("adapted_text")

    if not isinstance(translated_text, str):
        raise ValueError('Ollama dubbing response did not include "translated_text".')
    if not isinstance(adapted_text, str):
        raise ValueError('Ollama dubbing response did not include "adapted_text".')

    translated = _clean_text(translated_text)
    adapted = _clean_text(adapted_text)

    if not translated:
        raise ValueError('Ollama dubbing response included empty "translated_text".')
    if not adapted:
        raise ValueError('Ollama dubbing response included empty "adapted_text".')

    reason = parsed.get("reason")
    cleaned_reason = _clean_reason(reason) if isinstance(reason, str) else None

    return DubbingScriptResult(
        translated_text=translated,
        adapted_text=adapted,
        reason=cleaned_reason or None,
    )


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


def _clean_text(text: str) -> str:
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


def _character_budget(
    segment: Segment,
    options: DubbingScriptOptions,
) -> int | None:
    if segment.duration <= 0:
        return None

    return max(8, int(segment.duration * options.max_chars_per_second))


def _normalize_spacing(text: str) -> str:
    return " ".join(text.split())
