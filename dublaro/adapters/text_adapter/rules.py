import re

from dublaro.adapters.text_adapter.base import TextAdaptationOptions
from dublaro.schemas import Segment

_WHITESPACE_RE = re.compile(r"\s+")
_PARENTHESES_RE = re.compile(r"\([^)]*\)")

_LOW_VALUE_PATTERNS = (
    re.compile(r"\bactually\b", re.IGNORECASE),
    re.compile(r"\bbasically\b", re.IGNORECASE),
    re.compile(r"\breally\b", re.IGNORECASE),
    re.compile(r"\bjust\b", re.IGNORECASE),
    re.compile(r"\byou know\b", re.IGNORECASE),
    re.compile(r"\bi mean\b", re.IGNORECASE),
    re.compile(r"\bkind of\b", re.IGNORECASE),
    re.compile(r"\bsort of\b", re.IGNORECASE),
)


class RuleBasedTextAdapter:
    name = "rules"

    def adapt_segment(
        self,
        segment: Segment,
        options: TextAdaptationOptions,
    ) -> str:
        text = _normalize_spacing(segment.translated_text or segment.source_text)
        budget = _character_budget(segment, options)

        if budget is None or len(text) <= budget:
            return text

        shortened = _shorten_without_truncation(text)

        if len(shortened) <= budget:
            return shortened

        if options.preserve_meaning:
            return shortened

        return _trim_to_budget(shortened, budget)


def _character_budget(
    segment: Segment,
    options: TextAdaptationOptions,
) -> int | None:
    if segment.duration <= 0:
        return None

    return max(8, int(segment.duration * options.max_chars_per_second))


def _shorten_without_truncation(text: str) -> str:
    shortened = _remove_parenthetical_text(text)
    shortened = _remove_low_value_phrases(shortened)

    return shortened or text


def _remove_parenthetical_text(text: str) -> str:
    return _normalize_spacing(_PARENTHESES_RE.sub("", text))


def _remove_low_value_phrases(text: str) -> str:
    shortened = text

    for pattern in _LOW_VALUE_PATTERNS:
        shortened = pattern.sub("", shortened)

    return _normalize_spacing(shortened)


def _trim_to_budget(text: str, budget: int) -> str:
    if budget <= 0:
        return ""

    if len(text) <= budget:
        return text

    raw_cut = text[:budget]
    trimmed = raw_cut.rstrip()

    if not trimmed:
        return ""

    if raw_cut[-1].isspace():
        return trimmed

    boundary_index = max(trimmed.rfind(mark) for mark in (".", "!", "?", ",", ";", ":"))
    minimum_boundary_index = max(8, int(budget * 0.55))

    if boundary_index >= minimum_boundary_index:
        return trimmed[: boundary_index + 1].strip()

    space_index = trimmed.rfind(" ")

    if space_index >= minimum_boundary_index:
        return trimmed[:space_index].strip()

    return trimmed


def _normalize_spacing(text: str) -> str:
    normalized = _WHITESPACE_RE.sub(" ", text)
    normalized = re.sub(r"\s+([,.!?;:])", r"\1", normalized)
    normalized = re.sub(r"^[,;:]\s*", "", normalized.strip())
    return normalized
