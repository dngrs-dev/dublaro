from typing import Annotated

import typer

from dublaro import __version__
from dublaro.cli.commands import register_commands
from dublaro.cli.rendering import (
    console,
)

app = typer.Typer(
    name="dublaro",
    help="Open-source AI dubbing tools.",
    no_args_is_help=True,
)


def version_callback(show_version: bool) -> None:
    if show_version:
        console.print(f"dublaro {__version__}")
        raise typer.Exit


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            callback=version_callback,
            help="Show Dublaro version.",
            is_eager=True,
        ),
    ] = False,
) -> None:
    pass


register_commands(app)
