from dataclasses import dataclass

from dublaro.pipeline.dub.options import (
    DubAdapters,
    DubArtifactPaths,
    DubOptions,
    DubPaths,
)
from dublaro.pipeline.dub.progress import DubbingProgressCallback


@dataclass(frozen=True)
class DubRunContext:
    paths: DubPaths
    options: DubOptions
    adapters: DubAdapters
    artifact_paths: DubArtifactPaths
    progress_callback: DubbingProgressCallback | None = None
