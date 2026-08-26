# Dublaro

<!-- [![CI](https://github.com/dngrs-dev/dublaro/actions/workflows/ci.yml/badge.svg)](https://github.com/dngrs-dev/dublaro/actions/workflows/ci.yml) -->
[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Dublaro is an open-source CLI pipeline for AI dubbing.

It turns a source video into a target-language dubbed video by combining speech recognition, translation, text adaptation, text-to-speech, timing alignment, background audio mixing, subtitles, and resumable workspace artifacts.

## Features

- Extract audio from video with FFmpeg
- Transcribe speech with Faster-Whisper
- Detect speakers with pyannote diarization
- Translate with Argos Translate or local Ollama models
- Adapt translated text for dubbing timing
- Generate speech with Piper
- Route different speakers to different voice profiles
- Repair overlong speech with an LLM
- Fit speech timing with speed-up and optional video slowdown
- Duck original audio or use separated background audio with Demucs
- Export SRT subtitles
- Embed soft or hard subtitles into video
- Resume failed or stopped runs
- Inspect workspaces and generate quality reports
- Batch dub folders of videos

## Status

Dublaro is early, but already usable as a local CLI tool.

Good fit:

- local dubbing experiments
- short and medium videos
- manual review/edit workflows
- open-source AI media pipeline development

Not yet the goal:

- one-click perfect production dubbing
- a GUI
- real-time dubbing
- commercial API integrations

## Quick Start

Use Python 3.11 or newer. Use any environment you prefer: an existing virtual environment, a new virtual environment, uv, pipx, or plain `pip`.

Clone the project:

```powershell
git clone https://github.com/dngrs-dev/dublaro.git
cd dublaro
```

Install Dublaro with the common first-run extras:

```powershell
python -m pip install -e ".[asr,translation]"
```

Copy the example config:

```powershell
Copy-Item dublaro.example.toml dublaro.toml
```

Edit `dublaro.toml`, especially your Piper model paths, then check your setup:

```powershell
dublaro doctor --config .\dublaro.toml
```

Run one dub:

```powershell
dublaro dub .\data\input\video.mp4 --config .\dublaro.toml --output-dir .\data\output
```

Inspect the result:

```powershell
dublaro report .\.dublaro\video
```

For speaker diarization, source separation, and LLM timing repair, install and enable the optional extras described in the docs.

## Documentation

- [Getting Started](docs/getting-started.md)
- [Configuration](docs/configuration.md)
- [Workflows](docs/workflows.md)
- [Commands](docs/commands.md)
- [Troubleshooting](docs/troubleshooting.md)
- [Architecture](docs/architecture.md)
- [Development](docs/development.md)
- [Example Config](dublaro.example.toml)

## Core Pipeline

```text
video
  -> extracted audio
  -> transcript
  -> diarized transcript
  -> translated transcript
  -> adapted dubbing text
  -> synthesized speech
  -> timing repair
  -> speech fitting
  -> optional video fitting
  -> mixed or separated background audio
  -> subtitles
  -> dubbed video
```

## License

MIT. See [LICENSE](LICENSE).
