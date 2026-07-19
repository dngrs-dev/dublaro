from dublaro.adapters.translation.base import TranslationOptions


class FakeTranslationAdapter:
    name = "fake-translation"

    def translate_text(self, text: str, options: TranslationOptions) -> str:
        if not text.strip():
            return ""

        return f"[{options.target_language}] {text}"
