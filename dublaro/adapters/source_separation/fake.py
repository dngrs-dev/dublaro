from pathlib import Path
from shutil import copyfile

from dublaro.adapters.source_separation.base import (
    SourceSeparationOptions,
    SourceSeparationResult,
)


class FakeSourceSeparationAdapter:
    name = "fake-source-separation"

    def separate_sources(
        self,
        audio_path: str | Path,
        *,
        background_output_path: str | Path,
        voice_output_path: str | Path,
        options: SourceSeparationOptions,
        overwrite: bool = False,
    ) -> SourceSeparationResult:
        source = Path(audio_path)
        background = Path(background_output_path)
        voice = Path(voice_output_path)

        _copy_audio(source, background, overwrite=overwrite)
        _copy_audio(source, voice, overwrite=overwrite)

        return SourceSeparationResult(
            background_audio_path=background,
            voice_audio_path=voice,
        )


def _copy_audio(source: Path, destination: Path, *, overwrite: bool) -> None:
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {destination}")

    destination.parent.mkdir(parents=True, exist_ok=True)

    if source.resolve(strict=False) == destination.resolve(strict=False):
        return

    copyfile(source, destination)
