from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Literal

DubbingProgressStep = Literal[
    "extract_audio",
    "transcribe",
    "diarize",
    "translate",
    "adapt_text",
    "dubbing_script",
    "synthesize",
    "repair_timing",
    "fit_speech",
    "fit_video",
    "align_speech",
    "separate_background",
    "mix_original_audio",
    "normalize_audio",
    "export_video",
    "export_srt",
    "write_manifest",
]

DubbingProgressStatus = Literal["started", "finished", "failed", "skipped"]

DubbingProgressCallback = Callable[
    [DubbingProgressStep, DubbingProgressStatus, str],
    None,
]


@contextmanager
def progress_stage(
    callback: DubbingProgressCallback | None,
    step: DubbingProgressStep,
    message: str,
) -> Iterator[None]:
    if callback is None:
        yield
        return

    callback(step, "started", message)

    try:
        yield
    except Exception:
        callback(step, "failed", message)
        raise

    callback(step, "finished", message)


def progress_skipped(
    callback: DubbingProgressCallback | None,
    step: DubbingProgressStep,
    message: str,
) -> None:
    if callback is not None:
        callback(step, "skipped", message)
