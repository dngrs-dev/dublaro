# Workflows

## Full Dub

```powershell
dublaro dub .\data\input\lesson.mp4 --config .\dublaro.toml --output-dir .\data\output
```

## Overwrite Existing Output

```powershell
dublaro dub .\data\input\lesson.mp4 --config .\dublaro.toml --output-dir .\data\output --overwrite
```

## Resume A Failed Run

```powershell
dublaro dub .\data\input\lesson.mp4 --config .\dublaro.toml --resume
```

Resume reuses existing workspace artifacts.

## Review And Edit Text

Stop after adapted text:

```powershell
dublaro dub .\data\input\lesson.mp4 --config .\dublaro.toml --until adapted --overwrite
```

Edit:

```text
.dublaro/lesson/lesson.pl.adapted.json
```

Continue from adapted text:

```powershell
dublaro dub .\data\input\lesson.mp4 --config .\dublaro.toml --start-from adapted --overwrite
```

Use:

```powershell
--resume --start-from adapted
```

only when you intentionally want to reuse downstream artifacts.

## Batch Dubbing

Dry run:

```powershell
dublaro batch .\data\input --to pl --config .\dublaro.toml --output-dir .\data\output --dry-run
```

Run:

```powershell
dublaro batch .\data\input --to pl --config .\dublaro.toml --output-dir .\data\output
```

## Speaker Voice Preview

```powershell
dublaro preview-speakers .\.dublaro\lesson\lesson.en.diarized.json --config .\dublaro.toml
dublaro preview-voices --config .\dublaro.toml --text "Dzien dobry"
```

## Timing Preview

```powershell
dublaro preview-timing .\.dublaro\lesson\lesson.pl.synthesized.json --max-speedup 1.35
```

## Timing Repair Preview

```powershell
dublaro preview-repairs .\.dublaro\lesson\lesson.pl.timing-repaired.json
```

## Workspace Inspection

```powershell
dublaro inspect-workspace .\.dublaro\lesson
```

## Quality Report

```powershell
dublaro report .\.dublaro\lesson
```

Use this after a run to inspect:

- missing artifacts
- timing issues
- timing repair decisions
- selected adapters
- speaker voice count
- source separation mode
- subtitle mode

## Separate Audio First

```powershell
dublaro extract-audio .\data\input\lesson.mp4 --output .\data\voice\lesson.wav
dublaro separate-audio .\data\voice\lesson.wav --backend demucs
```

This is useful before using:

```toml
background_mode = "separated"
```

## Export SRT From Transcript

```powershell
dublaro export-srt .\.dublaro\lesson\lesson.pl.adapted.json --output .\data\output\lesson.pl.srt
```

## Embed Subtitles

Soft subtitles are selectable:

```powershell
dublaro dub .\data\input\lesson.mp4 --config .\dublaro.toml --subtitle-embed soft
```

Hard subtitles are burned into frames:

```powershell
dublaro dub .\data\input\lesson.mp4 --config .\dublaro.toml --subtitle-embed hard
```
