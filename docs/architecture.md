# Architecture

Dublaro is organized around adapters, pipeline stages, CLI commands, and workspace artifacts.

## High-Level Flow

```text
input video
  -> extract audio
  -> transcribe
  -> diarize
  -> translate
  -> adapt text
  -> synthesize speech
  -> repair timing
  -> fit speech
  -> fit video
  -> build speech track
  -> mix audio
  -> normalize audio
  -> prepare subtitles
  -> export video
  -> write manifest
```

## Adapters

Adapters isolate model or tool providers.

```text
dublaro/adapters/asr/
dublaro/adapters/diarization/
dublaro/adapters/translation/
dublaro/adapters/text_adapter/
dublaro/adapters/dubbing_script/
dublaro/adapters/tts/
dublaro/adapters/source_separation/
```

Current examples:

```text
ASR: fake, faster-whisper
Diarization: fake, pyannote
Translation: fake, argos, ollama
Text adapter: fake, rules, ollama
Dubbing script: ollama
TTS: fake, piper
Source separation: fake, demucs
```

## Pipeline

Reusable pipeline logic lives in:

```text
dublaro/pipeline/
```

The full dub runner lives in:

```text
dublaro/pipeline/dub/
```

Important files:

```text
runner.py       full dub orchestration
options.py      user-facing dub options and artifact paths
artifacts.py    returned artifacts and run state
context.py      context passed through stages
results.py      stage result dataclasses
progress.py     progress callback types
```

Stage implementations:

```text
dublaro/pipeline/dub/stages/audio.py
dublaro/pipeline/dub/stages/text.py
dublaro/pipeline/dub/stages/speech.py
dublaro/pipeline/dub/stages/export.py
```

Preflight:

```text
dublaro/pipeline/dub/preflight/
```

## CLI

CLI commands live in:

```text
dublaro/cli/commands/
```

CLI services live in:

```text
dublaro/cli/services/
```

CLI report builders live in:

```text
dublaro/cli/reports/
```

Rendering lives in:

```text
dublaro/cli/rendering.py
```

This keeps command functions thin.

## Workspace

Dublaro writes intermediate artifacts to:

```text
.dublaro/<video-name>/
```

This makes long runs resumable and debuggable.

Common artifacts:

```text
<video>.audio.wav
<video>.<source-lang>.json
<video>.<target-lang>.translated.json
<video>.<target-lang>.adapted.json
<video>.<target-lang>.synthesized.json
<video>.<target-lang>.timing-repaired.json
<video>.<target-lang>.fitted.json
<video>.<target-lang>.video-fitted.json
<video>.<target-lang>.speech/
<video>.<target-lang>.fitted-speech/
<video>.<target-lang>.speech-track.wav
<video>.<target-lang>.mixed.wav
<video>.<target-lang>.normalized.wav
<video>.<target-lang>.manifest.json
```

## Resume And Checkpoints

The full dub runner can stop at checkpoints:

```text
audio
transcribed
diarized
translated
adapted
synthesized
timing-repaired
fitted
video-fitted
aligned
mixed
normalized
subtitles
exported
manifest
```

Examples:

```powershell
dublaro dub video.mp4 --config dublaro.toml --until adapted
dublaro dub video.mp4 --config dublaro.toml --start-from adapted --overwrite
```

## Manifest

Each full run can write:

```text
<video>.<target-lang>.manifest.json
```

The manifest records:

- input/output paths
- language pair
- selected adapters
- options
- generated artifact paths
- metadata
- speaker voice routing

Use:

```powershell
dublaro report .\.dublaro\video
```

to summarize it.
