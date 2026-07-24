import json
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

from dublaro.adapters.tts.base import SpeechSynthesisOptions
from dublaro.schemas import Segment


class CommandRunner(Protocol):
    def __call__(
        self,
        args: Sequence[str],
        *,
        input: str,
        text: bool,
        capture_output: bool,
        check: bool,
    ) -> subprocess.CompletedProcess[str]: ...


class PiperTtsAdapter:
    name = "piper"

    def __init__(
        self,
        model_path: str | Path,
        *,
        config_path: str | Path | None = None,
        executable: str = "piper",
        speaker: int | None = None,
        runner: CommandRunner = subprocess.run,
    ) -> None:
        self.model_path = Path(model_path)
        self.config_path = Path(config_path) if config_path is not None else None
        self.executable = executable
        self.speaker = speaker
        self._runner = runner

        if not self.model_path.exists():
            raise FileNotFoundError(f"Piper model does not exist: {self.model_path}")

        if self.config_path is not None and not self.config_path.exists():
            raise FileNotFoundError(f"Piper config does not exist: {self.config_path}")

        config_for_metadata = self.config_path or default_piper_config_path(
            self.model_path
        )
        self.model_sample_rate = read_piper_sample_rate(config_for_metadata)

    def synthesize_segment(
        self,
        segment: Segment,
        output_path: Path,
        options: SpeechSynthesisOptions,
    ) -> Path:
        text = _segment_text(segment).strip()

        if not text:
            raise ValueError(f"Cannot synthesize empty segment: {segment.id}")

        if (
            self.model_sample_rate is not None
            and self.model_sample_rate != options.sample_rate
        ):
            raise ValueError(
                "Piper model sample rate is "
                f"{self.model_sample_rate}, but Dublaro expected "
                f"{options.sample_rate}. Use --sample-rate or "
                "--speech-sample-rate with the model sample rate."
            )

        output_path.parent.mkdir(parents=True, exist_ok=True)

        command = [
            self.executable,
            "--model",
            str(self.model_path),
            "--output_file",
            str(output_path),
        ]

        if self.config_path is not None:
            command.extend(["--config", str(self.config_path)])

        if self.speaker is not None:
            command.extend(["--speaker", str(self.speaker)])

        try:
            result = self._runner(
                command,
                input=f"{text}\n",
                text=True,
                capture_output=True,
                check=False,
            )
        except FileNotFoundError as error:
            raise RuntimeError(
                "Piper executable not found. Install Piper or pass "
                "--piper-executable with the full path."
            ) from error

        if result.returncode != 0:
            details = (result.stderr or result.stdout or "").strip()
            suffix = f": {details}" if details else ""
            raise RuntimeError(f"Piper failed{suffix}")

        if not output_path.exists():
            raise RuntimeError(f"Piper did not create output file: {output_path}")

        return output_path


def _segment_text(segment: Segment) -> str:
    return segment.adapted_text or segment.translated_text or segment.source_text


def default_piper_config_path(model_path: str | Path) -> Path:
    return Path(f"{Path(model_path)}.json")


def read_piper_sample_rate(config_path: str | Path) -> int | None:
    config_file = Path(config_path)

    if not config_file.exists():
        return None

    try:
        data = json.loads(config_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid Piper config JSON: {config_file}") from error

    if not isinstance(data, dict):
        return None

    audio = data.get("audio")
    if not isinstance(audio, dict):
        return None

    sample_rate = audio.get("sample_rate")
    if sample_rate is None:
        return None

    if isinstance(sample_rate, int) and not isinstance(sample_rate, bool):
        return sample_rate

    raise ValueError(
        f"Piper config audio.sample_rate must be an integer: {config_file}"
    )
