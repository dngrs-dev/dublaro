from importlib import import_module

from typer.testing import CliRunner

cli = import_module("dublaro.cli.app")


runner = CliRunner()


def test_cli_shows_version() -> None:
    result = runner.invoke(cli.app, ["--version"])

    assert result.exit_code == 0
    assert "dublaro" in result.output
