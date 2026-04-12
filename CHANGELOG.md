# Changelog

## 0.1.0 — 2026-04-12

### Added
- **Three generation modes:** Clone, Custom Voice, and Voice Design — accessible via TUI mode picker and CLI `--mode` flag.
- **Custom Voice mode** — 9 preset speakers (Vivian, Serena, Uncle_Fu, Dylan, Eric, Ryan, Aiden, Ono_Anna, Sohee) with optional emotion/style `--instruct`.
- **Voice Design mode** — describe a voice in natural language via `--instruct` and generate speech in that style.
- Mode selector on both Generate and Batch tabs, kept in sync. Selected mode persists across sessions via `.tui_config.json`.
- Lazy model reload on mode switch — evicts the previous model's weights and loads the new one on next generate.
- CLI flags: `--mode {clone,custom_voice,voice_design}`, `--voice`, `--instruct`, `--lang-code`.
- `auto_confirm` parameter on `load_model()` for non-interactive environments (TUI model reload).
- `CUSTOM_VOICE_SPEAKERS` constant with upstream speaker metadata.

### Changed
- **Switched to 8-bit quantised models** (`Qwen3-TTS-12Hz-1.7B-*-8bit`). Roughly half the memory of bf16 (~1.5 GB vs ~3.4 GB) and slightly faster on Apple Silicon.
- Removed 0.6B fallback chain — single 1.7B 8-bit model per mode, no silent downgrade.
- `generate()` replaces `generate_cloned()` as the primary generation function. Dispatches per mode, returns `(audio, sr, peak_memory)`.
- `load_model(mode)` replaces `load_model_with_fallback()` as the primary loader.
- Language selector now uses upstream's 10 supported languages + "Auto" (was 14 languages with lowercase values).
- CLI `--language` renamed to `--lang-code` (old flag kept as hidden alias).

### Fixed
- **`language=` kwarg bug** — `generate_cloned()` passed `language=` to `model.generate()`, which silently dropped into `**kwargs` and was ignored. All call sites now correctly use `lang_code=`.
- `run_test()` referenced non-existent speaker "Chelsie"; changed to "Ryan".

### Deprecated
- `load_model_with_fallback()` and `generate_cloned()` — kept as thin shims for `ui.py` back-compat. Will be removed in a future version.

## 0.0.1 — 2026-03-10

Initial release: voice cloning TUI with batch generation, audio enhancement, and Gradio web UI.
