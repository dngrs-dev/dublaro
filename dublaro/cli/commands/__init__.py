import typer

from dublaro.cli.commands.adapt_text import adapt_text
from dublaro.cli.commands.align_speech import align_speech
from dublaro.cli.commands.batch import batch
from dublaro.cli.commands.check_timing import check_timing
from dublaro.cli.commands.doctor import doctor
from dublaro.cli.commands.dub import dub
from dublaro.cli.commands.export_srt import export_srt
from dublaro.cli.commands.export_video import export_video
from dublaro.cli.commands.extract_audio import extract_audio
from dublaro.cli.commands.fit_speech import fit_speech
from dublaro.cli.commands.inspect_workspace import inspect_workspace
from dublaro.cli.commands.mix_audio import mix_audio
from dublaro.cli.commands.preview_repairs import preview_repairs
from dublaro.cli.commands.preview_speakers import preview_speakers
from dublaro.cli.commands.preview_timing import preview_timing
from dublaro.cli.commands.preview_units import preview_units
from dublaro.cli.commands.preview_voices import preview_voices
from dublaro.cli.commands.synthesize import synthesize
from dublaro.cli.commands.transcribe import transcribe
from dublaro.cli.commands.translate import translate

__all__ = ["register_commands"]


def register_commands(app: typer.Typer) -> None:
    app.command("adapt-text")(adapt_text)
    app.command("align-speech")(align_speech)
    app.command("batch")(batch)
    app.command("check-timing")(check_timing)
    app.command("doctor")(doctor)
    app.command("dub")(dub)
    app.command("export-srt")(export_srt)
    app.command("export-video")(export_video)
    app.command("extract-audio")(extract_audio)
    app.command("fit-speech")(fit_speech)
    app.command("inspect-workspace")(inspect_workspace)
    app.command("mix-audio")(mix_audio)
    app.command("preview-repairs")(preview_repairs)
    app.command("preview-speakers")(preview_speakers)
    app.command("preview-timing")(preview_timing)
    app.command("preview-units")(preview_units)
    app.command("preview-voices")(preview_voices)
    app.command("synthesize")(synthesize)
    app.command("transcribe")(transcribe)
    app.command("translate")(translate)
