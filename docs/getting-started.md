# Getting Started

This guide installs Dublaro and runs one local dubbing job.

## Requirements

Required:

- Python 3.11+
- FFmpeg on `PATH`
- Piper executable and at least one Piper voice model

Recommended for a first real run:

- Faster-Whisper for ASR
- Argos Translate for offline translation
- Piper for TTS

Optional advanced features:

- Ollama for LLM translation, text adaptation, and timing repair
- Demucs for source separation
- pyannote.audio for speaker diarization

## Python Environment

Dublaro does not require one specific environment manager.

Use whichever option fits your workflow:

- an existing virtual environment
- a new `venv`
- uv
- pipx
- plain `pip`

Check your Python version:

```powershell
python --version
```

It must be Python 3.11 or newer.

If your default `python` is older on Windows, use the Python launcher:

```powershell
py -3.11 --version
```

Then replace `python` with `py -3.11` in install commands.

## Install Dublaro

Clone the project:

```powershell
git clone https://github.com/dngrs-dev/dublaro.git
cd dublaro
```

Install the base package:

```powershell
python -m pip install -e .
```

For a practical first run, install ASR and translation extras:

```powershell
python -m pip install -e ".[asr,translation]"
```

Install optional source separation only if you want separated background audio:

```powershell
python -m pip install -e ".[source-separation]"
```

Install diarization only if you need automatic speaker detection:

```powershell
python -m pip install -e ".[diarization]"
```

Install development tools only if you will work on the code:

```powershell
python -m pip install -e ".[dev]"
```

## Optional: Create A Virtual Environment

If you want an isolated environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Then run the same install command:

```powershell
python -m pip install -e ".[asr,translation]"
```

## Install FFmpeg

Check if FFmpeg is available:

```powershell
ffmpeg -version
```

If not, install FFmpeg and make sure the executable is available in the same terminal where you run Dublaro.

## Install Piper Voices

Dublaro expects a Piper executable plus voice model files:

```text
models/piper/<voice>.onnx
models/piper/<voice>.onnx.json
```

The `.onnx.json` file is important because Dublaro can read the model sample rate from it.

## Optional: Install Ollama

Install Ollama if you want local LLM translation, text adaptation, or timing repair.

Pull a model:

```powershell
ollama pull llama3.1
```

Check that Ollama is running:

```powershell
ollama list
```

## Configure Dublaro

Copy the beginner-friendly example config:

```powershell
Copy-Item dublaro.example.toml dublaro.toml
```

Edit:

```text
dublaro.toml
```

At minimum, set:

```toml
[dub]
target_language = "pl"

[dub.tts]
backend = "piper"
piper_model_path = "models/piper/your-voice.onnx"
piper_config_path = "models/piper/your-voice.onnx.json"
```

The example config intentionally keeps advanced features disabled by default. Enable diarization, speaker voices, source separation, and LLM timing repair after the first basic dub works.

## Check Your Environment

```powershell
dublaro doctor --config .\dublaro.toml
```

Fix any errors before a long run.

## Preview Voices

```powershell
dublaro preview-voices --config .\dublaro.toml --text "Dzien dobry"
```

## Run A Dub

```powershell
dublaro dub .\data\input\video.mp4 --config .\dublaro.toml --output-dir .\data\output
```

Expected output:

```text
data/output/video.pl.dubbed.mp4
data/output/video.pl.dubbed.srt
.dublaro/video/
```

## Inspect Results

```powershell
dublaro inspect-workspace .\.dublaro\video
dublaro report .\.dublaro\video
```
