# Changelog

All notable changes to Dublaro will be documented in this file.

## v0.1.0 - Initial Public Release

Dublaro v0.1.0 is the first public release of the project.

### Added

- Full local AI dubbing pipeline from video input to dubbed video output
- Audio extraction with FFmpeg
- Speech recognition support with Faster-Whisper
- Speaker diarization support with pyannote
- Translation support with Argos Translate and Ollama
- Text adaptation for dubbing timing
- Combined LLM dubbing text workflow
- Text timing repair for overlong synthesized speech
- Speech synthesis with Piper
- Per-speaker voice profiles
- Source separation support with Demucs
- Background audio modes for original, ducked, and separated audio
- Speech timing fit with audio speed-up
- Optional video slowdown for difficult timing cases
- SRT subtitle export
- Soft and hard subtitle embedding
- Batch dubbing command
- Resume, start-from, and until-checkpoint workflows
- Workspace inspection command
- Quality report command
- Doctor command for environment checks
- Preview commands for voices, speakers, timing, repairs, and translation units
- TOML config file support
- Example configs and documentation
- GitHub Actions CI for linting, formatting, tests, and package build

### Known Limitations

- Dublaro is still early and may need manual review for best results.
- Long videos can be slow, especially with diarization, source separation, or local LLMs.
- Output quality depends heavily on the selected ASR, translation, TTS, and voice models.
- Piper voices must be downloaded separately.
- Some pyannote models may require a Hugging Face token.
- There is no GUI yet.
- Real-time dubbing is not supported.
