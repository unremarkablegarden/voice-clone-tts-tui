# Voice Clone — Qwen3-TTS on Apple Silicon

Local text-to-speech pipeline using Qwen3-TTS via mlx-audio, optimised for M1/M2/M3/M4 Macs. Three modes: voice cloning from a reference recording, custom voice with preset speakers and emotion/style instructions, and voice design from a text description.

## Setup

```bash
# Create venv and install deps
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -r requirements.txt

# ffmpeg is required for audio preprocessing
brew install ffmpeg
```

## Models

Three 1.7B 8-bit MLX models, one per mode. Downloaded automatically on first use to `~/.cache/huggingface/` (~2.9 GB each).

| Mode | Model | Use case |
|------|-------|----------|
| Clone | `mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit` | Match a voice from a 3–10s reference recording |
| Custom Voice | `mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit` | 9 preset speakers + optional emotion/style instruct |
| Voice Design | `mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit` | Describe a voice in natural language |

8-bit quantisation uses roughly half the memory of bf16 (~1.5 GB vs ~3.4 GB resident) and is slightly faster on Apple Silicon due to lower memory bandwidth pressure.

## Interfaces

### Terminal UI (recommended)

```bash
python tui.py
```

Full-featured terminal interface with four tabs:

- **Create Voice** — select source audio from `ref_audio/`, optionally preprocess and enhance, save as a named voice profile to `voices/`
- **Generate** — pick a mode (Clone / Custom Voice / Voice Design), configure mode-specific inputs, type text, adjust language/speed/quality, generate speech, play/save/enhance the output
- **Batch** — same mode picker, split a text file into segments, generate each as a numbered WAV, optionally join them
- **Cleanup** — enhance audio files with AI denoising, with configurable enhancement settings

A **Mode** selector at the top of the Generate and Batch tabs switches between the three modes. Switching mode evicts the current model and loads the new one on next generate. The selected mode is persisted across sessions.

**Mode-specific inputs:**

| Mode | Inputs shown |
|------|-------------|
| Clone | Voice dropdown (from `voices/`) |
| Custom Voice | Speaker dropdown (9 presets with descriptions) + optional Instruct text field |
| Voice Design | Voice Description text field (required) |

**Custom Voice speakers** (from upstream Qwen3-TTS):

| Speaker | Description | Native Language |
|---------|-------------|----------------|
| Vivian | Bright, slightly edgy young female | Chinese |
| Serena | Warm, gentle young female | Chinese |
| Uncle_Fu | Seasoned male, low mellow timbre | Chinese |
| Dylan | Youthful Beijing male, clear natural | Chinese (Beijing) |
| Eric | Lively Chengdu male, slightly husky | Chinese (Sichuan) |
| Ryan | Dynamic male, strong rhythmic drive | English |
| Aiden | Sunny American male, clear midrange | English |
| Ono_Anna | Playful Japanese female, light nimble | Japanese |
| Sohee | Warm Korean female, rich emotion | Korean |

A system sound (Glass.aiff) plays when generation or enhancement finishes. Playback state (Play/Stop buttons) resets automatically when audio finishes playing.

Keyboard shortcuts: `Ctrl+G` generate, `Ctrl+P` play, `Ctrl+S` save, `Q` quit.

### Web UI

```bash
python ui.py
# Opens at http://127.0.0.1:7860
```

Gradio interface — clone mode only, with reference audio upload (file or mic), language/speed controls.

### CLI

```bash
# Clone mode (default)
python clone_voice.py --mode clone \
  --text "Hello world" \
  --ref_audio ref_audio/ref_audio.wav \
  --ref_text "Transcript of the reference"

# Custom Voice mode
python clone_voice.py --mode custom_voice \
  --text "Hello world" \
  --voice Ryan \
  --instruct "Very happy and excited"

# Voice Design mode
python clone_voice.py --mode voice_design \
  --text "Hello world" \
  --instruct "Calm British narrator, deep and measured"

# From a text file, with options
python clone_voice.py --mode clone \
  --text script.txt \
  --ref_audio ref_audio/ref_audio.wav \
  --output output/my_clip.wav \
  --lang-code English \
  --speed 1.0

# Quick test (needs CustomVoice model)
python clone_voice.py --mode custom_voice --test --text test
```

## Voice Management

Voices are stored in `voices/{name}/` with `ref_audio.wav` and `meta.json`. Only used in Clone mode.

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

Enhancement parameters are configurable in the TUI's Cleanup tab under a collapsible "Enhancement Settings" panel, or via `enhance_audio.py` CLI flags:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `nfe` | 32 | Diffusion function evaluations (1-128) |
| `solver` | midpoint | ODE solver (midpoint/rk4/euler) |
| `lambd` | 0.5 | Denoise strength (0.0-1.0) |
| `tau` | 0.5 | Prior temperature (0.0-1.0) |
| `denoise_only` | False | Skip diffusion enhancer for speed |

## Quality Settings

Available in the TUI's "Generate" and "Batch" tabs under the collapsible "Quality Settings":

| Parameter | Default | Description |
|-----------|---------|-------------|
| Temperature | 0.9 | Sampling temperature |
| Top-K | 50 | Top-K sampling |
| Top-P | 1.0 | Nucleus sampling |
| Rep Penalty | 1.05 | Repetition penalty |

## Batch Generation

The model loses coherence on longer texts. The Batch tab solves this by splitting a source file into segments, generating each as a separate numbered WAV, and optionally joining them. This lets you re-generate individual problem segments without redoing everything. All three modes are available in Batch.

**Usage:**

1. Open the "Batch" tab in the TUI
2. Select a mode and configure its inputs (voice / speaker / description)
3. Enter the path to a `.txt` or `.md` file
4. Choose a segmentation strategy and adjust max chars if needed
5. Click "Start Batch" — segments are saved as `001.wav`, `002.wav`, etc.
6. Click "Join All" to concatenate all segments into `joined.wav`

**Segmentation strategies:**

| Strategy | Description |
|----------|-------------|
| Paragraphs | Split on blank lines, sub-split long paragraphs at sentence boundaries |
| Sentences | Split entire text at sentence boundaries |
| Fixed chars | Split at character limit respecting word boundaries |

Markdown files (`.md`) are automatically cleaned to plain text before segmentation — headers, bold/italic, links, code blocks, lists, and HTML tags are stripped.

Each batch run writes a `manifest.txt` listing segment numbers, durations, and text previews. You can cancel mid-batch and re-run individual segments by replacing specific numbered WAVs. The join step inserts configurable silence gaps (default 300ms) between segments.

## Project Structure

```
clone_voice.py      — CLI and core generation logic (mode-aware)
tui.py              — Terminal UI (Textual)
ui.py               — Web UI (Gradio, clone mode only)
enhance_audio.py    — AI audio enhancement (Resemble Enhance)
prep_reference.sh   — ffmpeg preprocessing pipeline
run.sh              — launch script (loads venv, starts TUI)
ref_audio/          — source audio files
voices/             — created voice profiles (ref_audio.wav + meta.json)
output/             — generated audio files
```

## Notes

- **mlx-audio version:** Qwen3-TTS support requires v0.4.0+ (installed from Git main, not PyPI 0.2.x)
- **Memory:** 8-bit 1.7B models use ~1.5 GB resident. Only one model is loaded at a time — switching modes evicts and reloads.
- **Speed:** Expect ~2-3x realtime factor on M1 (i.e., a 10s clip takes 20-30s to generate).
- **Tokenizer warning:** The "incorrect regex pattern" warning from transformers is cosmetic and doesn't affect output quality.
- **Long text:** Automatically chunked at sentence boundaries (~200 chars per chunk) and concatenated with 150ms silence gaps. For best results on long texts, use the Batch tab for per-segment control.
- **Languages:** Auto (detect), English, Chinese, Japanese, Korean, French, German, Spanish, Italian, Portuguese, Russian.
- **Instruction control:** Only supported on 1.7B CustomVoice and 1.7B VoiceDesign models. The Clone (Base) model does not respond to `instruct`.
