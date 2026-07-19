import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pytest
from dublaro.adapters.tts.base import SpeechSynthesisOptions
from dublaro.adapters.tts.piper import PiperTtsAdapter
from dublaro.schemas import Segment


@dataclass
class PiperCall:
    args: list[str]
    input: str
    text: bool
    capture_output: bool
    check: bool


def test_piper_tts_adapter_calls_piper(tmp_path: Path) -> None:
    model_path = tmp_path / "voice.onnx"
    config_path = tmp_path / "voice.onnx.json"
    output_path = tmp_path / "speech.wav"

    model_path.write_bytes(b"fake model")
    config_path.write_text('{"audio": {"sample_rate": 22050}}', encoding="utf-8")

    calls: list[PiperCall] = []

    def fake_runner(
        args: Sequence[str],
        *,
        input: str,
        text: bool,
        capture_output: bool,
        check: bool,
    ) -> subprocess.CompletedProcess[str]:
        calls.append(
            PiperCall(
                args=list(args),
                input=input,
                text=text,
                capture_output=capture_output,
                check=check,
            )
        )
        output_path.write_bytes(b"fake wav")
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    adapter = PiperTtsAdapter(
        model_path,
        config_path=config_path,
        speaker=0,
        runner=fake_runner,
    )

    result = adapter.synthesize_segment(
        Segment(
            id="seg-0001",
            start=0.0,
            end=1.0,
            adapted_text="Czesc swiecie",
        ),
        output_path,
        options=SpeechSynthesisOptions(language="pl", sample_rate=22050),
    )

    assert result == output_path
    assert output_path.exists()
    assert calls[0].input == "Czesc swiecie\n"

    args = calls[0].args
    assert "--model" in args
    assert str(model_path) in args
    assert "--config" in args
    assert str(config_path) in args
    assert "--speaker" in args
    assert "0" in args


def test_piper_tts_adapter_rejects_sample_rate_mismatch(tmp_path: Path) -> None:
    model_path = tmp_path / "voice.onnx"
    config_path = tmp_path / "voice.onnx.json"

    model_path.write_bytes(b"fake model")
    config_path.write_text('{"audio": {"sample_rate": 22050}}', encoding="utf-8")

    adapter = PiperTtsAdapter(model_path, config_path=config_path)

    with pytest.raises(ValueError, match="sample rate"):
        adapter.synthesize_segment(
            Segment(id="seg-0001", start=0.0, end=1.0, adapted_text="Hello"),
            tmp_path / "speech.wav",
            options=SpeechSynthesisOptions(language="en", sample_rate=24000),
        )


def test_piper_tts_adapter_reports_missing_executable(tmp_path: Path) -> None:
    model_path = tmp_path / "voice.onnx"
    model_path.write_bytes(b"fake model")

    def fake_runner(
        *args: object, **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        raise FileNotFoundError

    adapter = PiperTtsAdapter(model_path, runner=fake_runner)

    with pytest.raises(RuntimeError, match="Piper executable not found"):
        adapter.synthesize_segment(
            Segment(id="seg-0001", start=0.0, end=1.0, adapted_text="Hello"),
            tmp_path / "speech.wav",
            options=SpeechSynthesisOptions(language="en", sample_rate=24000),
        )
