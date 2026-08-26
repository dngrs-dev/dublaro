# Troubleshooting

## Run Doctor First

```powershell
dublaro doctor --config .\dublaro.toml
```

This catches most environment and config problems before a long run.

## FFmpeg Missing

Error:

```text
ffmpeg executable was not found
```

Check:

```powershell
ffmpeg -version
```

Fix:

- install FFmpeg
- add it to `PATH`
- or pass `--ffmpeg C:\path\to\ffmpeg.exe`

## Output Already Exists

Use:

```powershell
--overwrite
```

or choose a different output path.

For failed runs, prefer:

```powershell
--resume
```

## Piper Model Missing

Check:

```toml
[dub.tts]
piper_model_path = "models/piper/voice.onnx"
piper_config_path = "models/piper/voice.onnx.json"
```

Remember: relative paths are resolved relative to the config file.

## Piper Sample Rate Error

Make sure the Piper `.onnx.json` exists next to the model.

Dublaro can auto-detect the sample rate from:

```text
voice.onnx.json
```

You can also set it manually:

```toml
[dub]
speech_sample_rate = 22050
```

## Ollama Connection Refused

Check:

```powershell
ollama list
```

If Ollama is not running, start it.

If the model is missing:

```powershell
ollama pull llama3.1
```

If it times out, try:

- a smaller model
- a longer timeout in config
- fewer timing repair attempts

## Argos Package Missing

Install translation extra:

```powershell
python -m pip install -e ".[translation]"
```

Run with package install:

```powershell
dublaro dub video.mp4 --to pl --translation argos --install-package
```

Or install the language package manually.

## Hugging Face Warnings

Set:

```powershell
$env:HF_TOKEN = "your-token"
```

This can improve downloads and is often required for gated pyannote models.

## Pyannote Model Access

Some pyannote models require:

- Hugging Face account
- accepted model terms
- `HF_TOKEN`

If diarization is not needed, disable it:

```toml
[dub.diarization]
enabled = false
```

## Demucs Is Slow

Demucs is expensive, especially on CPU.

Test source separation separately:

```powershell
dublaro separate-audio .\data\voice\sample.wav --backend demucs
```

Use this only when needed:

```toml
background_mode = "separated"
```

Otherwise use:

```toml
background_mode = "ducked"
```

## Subtitles Do Not Appear

For a separate SRT file:

```toml
[dub.srt]
export = true
embed = "none"
```

For selectable subtitles:

```toml
[dub.srt]
export = true
embed = "soft"
```

For burned-in subtitles:

```toml
[dub.srt]
export = true
embed = "hard"
```

## Timing Sounds Too Fast

Lower speed-up limits:

```toml
[dub.fit_speech]
max_speedup = 1.2
```

Enable timing repair:

```toml
[dub.timing_repair]
enabled = true
```

Avoid video slowdown unless you accept changed video timing:

```toml
[dub.fit_video]
enabled = false
```

## Check What Happened

```powershell
dublaro inspect-workspace .\.dublaro\video
dublaro report .\.dublaro\video
dublaro preview-timing .\.dublaro\video\video.pl.synthesized.json
dublaro preview-repairs .\.dublaro\video\video.pl.timing-repaired.json
```
