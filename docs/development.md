# Development

This page is for people changing Dublaro code.

If you only want to use Dublaro, start with [Getting Started](getting-started.md).

## Install For Development

Use Python 3.11 or newer.

Clone the project:

```powershell
git clone https://github.com/dngrs-dev/dublaro.git
cd dublaro
```

Create a virtual environment if you want an isolated dev environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install Dublaro with development tools:

```powershell
python -m pip install -e ".[dev]"
```

Install optional runtime extras as needed:

```powershell
python -m pip install -e ".[asr,translation,source-separation]"
python -m pip install -e ".[diarization]"
```

## Checks

Run all main checks:

```powershell
ruff check .
ruff format --check .
pytest
python -m build
```

Format code:

```powershell
ruff format .
```

## Test Layout

```text
tests/adapters/
tests/cli/
tests/pipeline/
```

Adapter tests should avoid downloading real models.

CLI tests should patch pipeline/service calls.

Pipeline tests should use fake adapters and small generated WAV files.

## Commit Style

Recommended examples:

```text
feat: add dub quality report command
fix: handle missing Piper config path
refactor: split dub preflight
docs: add user documentation
ci: add GitHub Actions checks
test: cover batch resume behavior
```

## Adding An Adapter

Add files under:

```text
dublaro/adapters/<domain>/
```

Typical structure:

```text
base.py
fake.py
provider.py
__init__.py
```

Then add:

- tests under `tests/adapters/<domain>/`
- factory support under `dublaro/cli/services/adapter_factories.py`
- config fields if needed
- preflight or doctor checks if needed
- docs update

## Adding A CLI Command

Add command file:

```text
dublaro/cli/commands/<command>.py
```

Register it in:

```text
dublaro/cli/commands/__init__.py
```

If the command prints complex output, add report data classes in:

```text
dublaro/cli/reports/
```

and rendering in:

```text
dublaro/cli/rendering.py
```

## CI

GitHub Actions runs:

```text
ruff check .
ruff format --check .
pytest
python -m build
```

CI intentionally avoids heavy optional model downloads.
