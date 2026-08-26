# Configuration

Dublaro uses TOML config files.

The recommended local config file is:

```text
dublaro.toml
```

This file is ignored by git because it usually contains machine-specific paths.

Use this tracked template:

```text
dublaro.example.toml
```

## Path Resolution

Relative paths in a config file are resolved relative to that config file.

If the config is here:

```text
C:\Projects\dublaro\dublaro.toml
```

then this:

```toml
piper_model_path = "models/piper/voice.onnx"
```

means:

```text
C:\Projects\dublaro\models\piper\voice.onnx
```

Because `dublaro.example.toml` lives in the project root, its paths are written for the project root. After copying it to `dublaro.toml`, the paths still point to the same places.

## Minimal Config

```toml
[dub]
target_language = "pl"
output_dir = "data/output"
workspace_dir = ".dublaro"

[dub.tts]
backend = "piper"
piper_model_path = "models/piper/pl.onnx"
piper_config_path = "models/piper/pl.onnx.json"
```

## Recommended Local Config

Start from:

```powershell
Copy-Item dublaro.example.toml dublaro.toml
```

Then edit:

- `target_language`
- Piper model paths
- ASR/translation/text adapter choices
- output/workspace paths
- advanced features after the first run works

The example config is intentionally conservative. It should help a new user get one successful dub before enabling heavier features.

## Beginner Defaults

The example config uses:

```text
ASR: faster-whisper
Translation: argos
Text adapter: rules
TTS: piper
Diarization: disabled
Timing repair: disabled
Source separation: disabled
Background mode: ducked
```

Enable advanced options only when the basic path works.

## Important Sections

### `[dub]`

Main run settings.

Common fields:

```toml
source_language = "en"
target_language = "pl"
output_dir = "data/output"
workspace_dir = ".dublaro"
text_workflow = "separate"
background_mode = "ducked"
preflight = true
overwrite = false
resume = false
ffmpeg_executable = "ffmpeg"
```

### `[dub.asr]`

Speech recognition.

```toml
[dub.asr]
backend = "faster-whisper"
model_size = "small"
device = "cpu"
compute_type = "int8"
```

Backends:

```text
fake
faster-whisper
```

### `[dub.diarization]`

Speaker detection.

Beginner default:

```toml
[dub.diarization]
enabled = false
backend = "fake"
```

Advanced pyannote example:

```toml
[dub.diarization]
enabled = true
backend = "pyannote"
model_id = "pyannote/speaker-diarization-community-1"
device = "cpu"
token_env_var = "HF_TOKEN"
```

Backends:

```text
fake
pyannote
```

### `[dub.translation]`

Translation.

```toml
[dub.translation]
backend = "argos"
install_package = false
group_segments = true
```

Backends:

```text
fake
argos
ollama
```

Ollama example:

```toml
[dub.translation]
backend = "ollama"
ollama_model = "llama3.1"
ollama_url = "http://localhost:11434"
ollama_timeout_seconds = 120.0
ollama_temperature = 0.2
```

### `[dub.text_adapter]`

Text adaptation for dubbing.

Beginner default:

```toml
[dub.text_adapter]
backend = "rules"
```

Ollama example:

```toml
[dub.text_adapter]
backend = "ollama"
ollama_model = "llama3.1"
ollama_url = "http://localhost:11434"
ollama_timeout_seconds = 120.0
ollama_temperature = 0.2
```

Backends:

```text
fake
rules
ollama
```

### `[dub.tts]`

Speech synthesis.

```toml
[dub.tts]
backend = "piper"
piper_model_path = "models/piper/pl.onnx"
piper_config_path = "models/piper/pl.onnx.json"
piper_executable = "piper"
```

Backends:

```text
fake
piper
```

### `[dub.timing_repair]`

LLM-based repair for overlong speech text.

Beginner default:

```toml
[dub.timing_repair]
enabled = false
```

Ollama-based repair example:

```toml
[dub.timing_repair]
enabled = true
max_attempts = 2
target_speedup = 1.15
```

This works best with:

```toml
[dub.text_adapter]
backend = "ollama"
```

### `[dub.fit_speech]`

Audio speed-up when synthesized speech is too long.

```toml
[dub.fit_speech]
enabled = true
max_speedup = 1.35
min_overrun_seconds = 0.05
```

### `[dub.fit_video]`

Optional video slowdown when speech cannot fit cleanly.

```toml
[dub.fit_video]
enabled = false
max_slowdown = 1.2
```

Use carefully. Video slowdown changes the final video timing.

### `[dub.source_separation]`

Background audio source separation.

Beginner default:

```toml
[dub.source_separation]
backend = "fake"
```

Demucs example:

```toml
[dub]
background_mode = "separated"

[dub.source_separation]
backend = "demucs"
demucs_executable = "demucs"
demucs_model = "htdemucs"
demucs_device = "cpu"
```

### `[dub.mix]`

Audio mixing.

```toml
[dub.mix]
enabled = true
original_audio_gain = 0.25
ducking_gain = 0.08
speech_gain = 1.0
ducking_margin_seconds = 0.15
ducking_fade_seconds = 0.05
```

### `[dub.srt]`

Subtitle export and embedding.

```toml
[dub.srt]
export = true
text_mode = "adapted"
embed = "none"
```

Embed modes:

```text
none
soft
hard
```

### `[voices."SPEAKER_00"]`

Per-speaker voice routing.

Only enable this after diarization works and you know the speaker IDs.

```toml
[voices."SPEAKER_00"]
display_name = "Speaker 1"
language = "pl"
tts_backend = "piper"
piper_model_path = "models/piper/pl-speaker-1.onnx"
piper_config_path = "models/piper/pl-speaker-1.onnx.json"
```

Speaker IDs usually come from diarization, for example:

```text
SPEAKER_00
SPEAKER_01
```
