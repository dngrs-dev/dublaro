# Commands

Use `--help` for the exact current options:

```powershell
dublaro --help
dublaro dub --help
```

## Environment

```powershell
dublaro doctor
```

Checks local tools and config.

## Full Pipeline

```powershell
dublaro dub VIDEO
```

Runs the full dubbing pipeline.

Common options:

```powershell
--config dublaro.toml
--to pl
--from en
--output-dir data/output
--overwrite
--resume
--until adapted
--start-from adapted
--background-mode ducked
--subtitle-embed soft
```

## Batch

```powershell
dublaro batch INPUT_DIR
```

Runs dubbing for multiple videos.

Common options:

```powershell
--to pl
--config dublaro.toml
--output-dir data/output
--dry-run
--overwrite
--resume
```

## Step Commands

```powershell
dublaro extract-audio
dublaro transcribe
dublaro translate
dublaro adapt-text
dublaro synthesize
dublaro align-speech
dublaro fit-speech
dublaro mix-audio
dublaro normalize-audio
dublaro export-srt
dublaro export-video
```

These are useful for debugging and building pipelines manually.

## Preview Commands

```powershell
dublaro preview-units
dublaro preview-speakers
dublaro preview-timing
dublaro preview-repairs
dublaro preview-voices
```

Use preview commands before a long run or after editing artifacts.

## Inspection Commands

```powershell
dublaro inspect-workspace WORKSPACE
dublaro report WORKSPACE
```

Use these after a run to understand what happened.

## Utility Commands

```powershell
dublaro separate-audio
dublaro check-timing
```

## Recommended First Commands

```powershell
dublaro doctor --config .\dublaro.toml
dublaro preview-voices --config .\dublaro.toml --text "Dzien dobry"
dublaro dub .\data\input\video.mp4 --config .\dublaro.toml --output-dir .\data\output
dublaro report .\.dublaro\video
```
