from typing import Any

from dublaro.adapters.translation.base import TranslationOptions


class ArgosTranslationAdapter:
    name = "argos"

    def __init__(self, *, auto_install: bool = False) -> None:
        self.auto_install = auto_install

    def translate_text(self, text: str, options: TranslationOptions) -> str:
        if not text.strip():
            return ""

        if not options.source_language:
            raise ValueError("Argos translation requires a source language.")

        package_module, translate_module = self._load_argos()

        if self.auto_install:
            self._install_package_if_missing(
                package_module,
                translate_module,
                options.source_language,
                options.target_language,
            )

        try:
            return translate_module.translate(
                text,
                options.source_language,
                options.target_language,
            )
        except Exception as error:
            raise RuntimeError(
                "Argos translation package is not installed for "
                f"{options.source_language}->{options.target_language}. "
                "Run again with --install-package or install the package manually."
            ) from error

    def _load_argos(self) -> tuple[Any, Any]:
        try:
            import argostranslate.package as package_module
            import argostranslate.translate as translate_module
        except ImportError as error:
            raise RuntimeError(
                "argostranslate is not installed. "
                'Install it with: pip install -e ".[translation]"'
            ) from error

        return package_module, translate_module

    def _install_package_if_missing(
        self,
        package_module: Any,
        translate_module: Any,
        source_language: str,
        target_language: str,
    ) -> None:
        if self._has_translation(
            translate_module,
            source_language,
            target_language,
        ):
            return

        package_module.update_package_index()
        available_packages = package_module.get_available_packages()

        package_to_install = next(
            (
                package
                for package in available_packages
                if package.from_code == source_language
                and package.to_code == target_language
            ),
            None,
        )

        if package_to_install is None:
            raise RuntimeError(
                f"No Argos translation package found for "
                f"{source_language}->{target_language}."
            )

        package_module.install_from_path(package_to_install.download())

    def _has_translation(
        self,
        translate_module: Any,
        source_language: str,
        target_language: str,
    ) -> bool:
        try:
            translate_module.get_translation_from_codes(
                source_language,
                target_language,
            )
        except Exception:
            return False

        return True
