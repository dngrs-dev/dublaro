from importlib import import_module
from pathlib import Path

from typer.testing import CliRunner

cli = import_module("dublaro.cli.app")


runner = CliRunner()


def test_inspect_workspace_command_shows_workspace_artifacts(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "lesson.audio.wav").write_bytes(b"audio")

    result = runner.invoke(cli.app, ["inspect-workspace", str(workspace)])

    assert result.exit_code == 0
    assert "Workspace" in result.output
    assert "Artifacts" in result.output
    assert "Workspace Artifacts" in result.output
