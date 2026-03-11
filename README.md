# Voice Clone — Qwen3-TTS on Apple Silicon

Local voice cloning pipeline using Qwen3-TTS via mlx-audio, optimised for M1/M2/M3/M4 Macs.

## Setup

```bash
# Create venv and install deps
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -r requirements.txt

# ffmpeg is required for audio preprocessing
brew install ffmpeg
```

## Interfaces

### Terminal UI (recommended)

```bash
python tui.py
```

Full-featured terminal interface with four tabs:

- **Create Voice** — select source audio from `ref_audio/`, optionally preprocess and enhance, save as a named voice profile to `voices/`
- **Generate** — pick a voice, type text, adjust language/speed/quality, generate speech, play/save/enhance the output
- **Batch** — split a text file into segments, generate each as a numbered WAV, optionally join them (see below)
- **Cleanup** — enhance audio files with AI denoising

Keyboard shortcuts: `Ctrl+G` generate, `Ctrl+P` play, `Ctrl+S` save, `Q` quit.

### Web UI

```bash
python ui.py
# Opens at http://127.0.0.1:7860
```

Gradio interface with reference audio upload (file or mic), language/speed controls.

### CLI

```bash
# Basic usage
python clone_voice.py --text "Hello world" --ref_audio ref_audio/ref_audio.wav

# From a text file
python clone_voice.py --text script.txt --ref_audio ref_audio/ref_audio.wav

# With options
python clone_voice.py \
  --text "Hello world" \
  --ref_audio ref_audio/ref_audio.wav \
  --ref_text "Transcript of the reference audio" \
  --output output/my_clip.wav \
  --language English \
  --speed 1.0

# Quick test
python clone_voice.py --test --ref_audio ref_audio/ref_audio.wav --text test
```

## Voice Management

Voices are stored in `voices/{name}/` with `ref_audio.wav` and `meta.json`.

**Create a voice via TUI:**

1. Place source audio in `ref_audio/`
2. Open the "Create Voice" tab
3. Select source audio, enter a name, optionally add a reference transcript
4. Toggle **Preprocess** (cleans & trims via ffmpeg) and/or **Enhance** (AI denoising)
5. Click "Create Voice"

**Manually:**

```bash
mkdir -p voices/my-voice
./prep_reference.sh your_audio.wav voices/my-voice/ref_audio.wav
echo '{"name": "my-voice", "transcript": ""}' > voices/my-voice/meta.json
```

## Audio Preprocessing

```bash
./prep_reference.sh <input_audio> [output_path]
```

Pipeline: mono 24kHz conversion, highpass 80Hz, lowpass 8kHz, noise reduction, compression, silence trimming. Files >10s are auto-trimmed to the loudest 10s segment.

## Audio Enhancement

AI audio enhancement powered by [Resemble Enhance](https://github.com/resemble-ai/resemble-enhance) (UNet denoiser + diffusion enhancer).

**Two use cases:**

- **Voice creation** — enhance reference audio at 24kHz (matches TTS model input). Toggle "Enhance" in the Create Voice tab.
- **Post-generation** — enhance TTS output at 44.1kHz (Resemble's native rate, better quality). Click "Enhance" in the Generate tab after generating.

Post-generation enhancement saves to `output/generate-{timestamp}_enhanced.wav` alongside the original 24kHz file. Play switches to the enhanced version automatically.

Enhancement parameters (via `enhance_audio.py`):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `nfe` | 32 | Diffusion function evaluations (1-128) |
| `solver` | midpoint | ODE solver (midpoint/rk4/euler) |
| `lambd` | 0.5 | Denoise strength (0.0-1.0) |
| `tau` | 0.5 | Prior temperature (0.0-1.0) |
| `denoise_only` | False | Skip diffusion enhancer for speed |

## Quality Settings

Available in the TUI's "Generate" tab under the collapsible "Quality Settings":

| Parameter | Default | Description |
|-----------|---------|-------------|
| Temperature | 0.9 | Sampling temperature |
| Top-K | 50 | Top-K sampling |
| Top-P | 1.0 | Nucleus sampling |
| Rep Penalty | 1.05 | Repetition penalty |

## Batch Generation

The model loses coherence on longer texts. The Batch tab solves this by splitting a source file into segments, generating each as a separate numbered WAV, and optionally joining them. This lets you re-generate individual problem segments without redoing everything.

**Usage:**

1. Open the "Batch" tab in the TUI
2. Select a voice and enter the path to a `.txt` or `.md` file
3. Choose a segmentation strategy and adjust max chars if needed
4. Click "Start Batch" — segments are saved as `001.wav`, `002.wav`, etc.
5. Click "Join All" to concatenate all segments into `joined.wav`

**Segmentation strategies:**

| Strategy | Description |
|----------|-------------|
| Paragraphs | Split on blank lines, sub-split long paragraphs at sentence boundaries |
| Sentences | Split entire text at sentence boundaries |
| Fixed chars | Split at character limit respecting word boundaries |

Markdown files (`.md`) are automatically cleaned to plain text before segmentation — headers, bold/italic, links, code blocks, lists, and HTML tags are stripped.

Each batch run writes a `manifest.txt` listing segment numbers, durations, and text previews. You can cancel mid-batch and re-run individual segments by replacing specific numbered WAVs. The join step inserts configurable silence gaps (default 300ms) between segments.

## Models

- **Default:** `mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16` (~3.4GB)
- **Fallback:** `mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16` (~1.2GB)

Models download automatically on first run to `~/.cache/huggingface/`.

## Project Structure

```
clone_voice.py      — CLI and core generation logic
tui.py              — Terminal UI (Textual)
ui.py               — Web UI (Gradio)
enhance_audio.py    — AI audio enhancement (Resemble Enhance)
prep_reference.sh   — ffmpeg preprocessing pipeline
ref_audio/          — source audio files
voices/             — created voice profiles (ref_audio.wav + meta.json)
output/             — generated audio files
```

## Notes

- **mlx-audio version:** Qwen3-TTS support requires v0.4.0+ (installed from Git main, not PyPI 0.2.x)
- **Memory:** The 1.7B model uses ~6-8GB. On 16GB M1, this leaves headroom for the OS.
- **Speed:** Expect ~2-3x realtime factor on M1 (i.e., a 10s clip takes 20-30s to generate).
- **Tokenizer warning:** The "incorrect regex pattern" warning from transformers is cosmetic and doesn't affect output quality.
- **Long text:** Automatically chunked at sentence boundaries (~200 chars per chunk) and concatenated with 150ms silence gaps. For best results on long texts, use the Batch tab for per-segment control.
- **Languages:** English, Chinese, Japanese, Korean, French, German, Spanish, Italian, Portuguese, Russian, Arabic, Thai, Indonesian, Vietnamese.
