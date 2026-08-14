import sys
from types import ModuleType

import pytest
from dublaro.adapters.translation.argos import ArgosTranslationAdapter
from dublaro.adapters.translation.base import TranslationOptions


def install_fake_argos_modules(
    monkeypatch: pytest.MonkeyPatch,
    *,
    translation_installed: bool = True,
) -> list[object]:
    calls: list[object] = []

    package_module = ModuleType("argostranslate.package")
    translate_module = ModuleType("argostranslate.translate")
    argos_module = ModuleType("argostranslate")

    class FakePackage:
        from_code = "en"
        to_code = "es"

        def download(self) -> str:
            calls.append("download")
            return "en_es.argosmodel"

    def update_package_index() -> None:
        calls.append("update_package_index")

    def get_available_packages() -> list[FakePackage]:
        calls.append("get_available_packages")
        return [FakePackage()]

    def install_from_path(path: str) -> None:
        calls.append(("install_from_path", path))

    def get_translation_from_codes(
        source_language: str, target_language: str
    ) -> object:
        calls.append(("get_translation_from_codes", source_language, target_language))
        if not translation_installed:
            raise RuntimeError("missing translation")
        return object()

    def translate(text: str, source_language: str, target_language: str) -> str:
        calls.append(("translate", text, source_language, target_language))
        return f"{source_language}->{target_language}: {text}"

    package_module.__dict__.update(
        {
            "update_package_index": update_package_index,
            "get_available_packages": get_available_packages,
            "install_from_path": install_from_path,
        }
    )

    translate_module.__dict__.update(
        {
            "get_translation_from_codes": get_translation_from_codes,
            "translate": translate,
        }
    )

    argos_module.__dict__.update(
        {
            "package": package_module,
            "translate": translate_module,
        }
    )

    monkeypatch.setitem(sys.modules, "argostranslate", argos_module)
    monkeypatch.setitem(sys.modules, "argostranslate.package", package_module)
    monkeypatch.setitem(sys.modules, "argostranslate.translate", translate_module)

    return calls


def test_argos_adapter_translates_text(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = install_fake_argos_modules(monkeypatch)

    adapter = ArgosTranslationAdapter()

    translated = adapter.translate_text(
        "Hello world",
        TranslationOptions(source_language="en", target_language="es"),
    )

    assert translated == "en->es: Hello world"
    assert ("translate", "Hello world", "en", "es") in calls


def test_argos_adapter_installs_missing_package(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = install_fake_argos_modules(monkeypatch, translation_installed=False)

    adapter = ArgosTranslationAdapter(auto_install=True)

    translated = adapter.translate_text(
        "Hello world",
        TranslationOptions(source_language="en", target_language="es"),
    )

    assert translated == "en->es: Hello world"
    assert "update_package_index" in calls
    assert "download" in calls
    assert ("install_from_path", "en_es.argosmodel") in calls


def test_argos_adapter_requires_source_language() -> None:
    adapter = ArgosTranslationAdapter()

    with pytest.raises(ValueError, match="source language"):
        adapter.translate_text(
            "Hello world",
            TranslationOptions(target_language="es"),
        )
