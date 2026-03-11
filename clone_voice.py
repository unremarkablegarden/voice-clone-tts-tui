#!/usr/bin/env python3
"""Voice cloning using Qwen3-TTS via mlx-audio on Apple Silicon."""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf


# Model preference order: 1.7B bf16 → 0.6B bf16
MODELS = [
    "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16",
    "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16",
]

SAMPLE_RATE = 24000
MAX_CHUNK_CHARS = 200  # Characters per chunk for long text


def parse_args():
    parser = argparse.ArgumentParser(description="Voice cloning with Qwen3-TTS (MLX)")
    parser.add_argument(
        "--text",
        required=True,
        help="Text to synthesize, or path to a .txt file",
    )
    parser.add_argument(
        "--ref_audio",
        required=True,
        help="Path to reference audio WAV file",
    )
    parser.add_argument(
        "--ref_text",
        default="",
        help="Transcript of the reference audio (improves quality)",
    )
    parser.add_argument(
        "--output",
        default="output/output.wav",
        help="Output WAV path (default: output/output.wav)",
    )
    parser.add_argument(
        "--language",
        default="English",
        help="Language for synthesis (default: English)",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Speech speed multiplier (default: 1.0)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override model ID (default: auto-select)",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run a quick test with a built-in voice (no ref_audio needed)",
    )
    return parser.parse_args()


def get_text(text_arg: str) -> str:
    """Return text content — reads from file if path ends with .txt."""
    if text_arg.endswith(".txt") and os.path.isfile(text_arg):
        return Path(text_arg).read_text(encoding="utf-8").strip()
    return text_arg


def chunk_text(text: str, max_chars: int = MAX_CHUNK_CHARS) -> list[str]:
    """Split long text into chunks at sentence boundaries."""
    if len(text) <= max_chars:
        return [text]

    chunks = []
    current = ""

    # Split on sentence-ending punctuation
    sentences = []
    buf = ""
    for char in text:
        buf += char
        if char in ".!?;":
            sentences.append(buf.strip())
            buf = ""
    if buf.strip():
        sentences.append(buf.strip())

    for sentence in sentences:
        if len(current) + len(sentence) + 1 > max_chars and current:
            chunks.append(current.strip())
            current = sentence
        else:
            current = f"{current} {sentence}" if current else sentence

    if current.strip():
        chunks.append(current.strip())

    return chunks if chunks else [text]


def clean_markdown(text: str) -> str:
    """Strip markdown formatting to plain text. Idempotent on plain text."""
    import re
    # Code blocks (fenced)
    text = re.sub(r"```[\s\S]*?```", "", text)
    # Inline code
    text = re.sub(r"`([^`]+)`", r"\1", text)
    # Images
    text = re.sub(r"!\[([^\]]*)\]\([^)]+\)", r"\1", text)
    # Links
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    # Headers
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    # Bold/italic
    text = re.sub(r"\*{1,3}([^*]+)\*{1,3}", r"\1", text)
    text = re.sub(r"_{1,3}([^_]+)_{1,3}", r"\1", text)
    # Strikethrough
    text = re.sub(r"~~([^~]+)~~", r"\1", text)
    # Blockquotes
    text = re.sub(r"^>\s?", "", text, flags=re.MULTILINE)
    # Unordered list markers
    text = re.sub(r"^[\s]*[-*+]\s+", "", text, flags=re.MULTILINE)
    # Ordered list markers
    text = re.sub(r"^[\s]*\d+\.\s+", "", text, flags=re.MULTILINE)
    # Horizontal rules
    text = re.sub(r"^[-*_]{3,}\s*$", "", text, flags=re.MULTILINE)
    # HTML tags
    text = re.sub(r"<[^>]+>", "", text)
    # Collapse multiple blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_into_segments(
    text: str, strategy: str = "paragraphs", max_chars: int = MAX_CHUNK_CHARS
) -> list[str]:
    """Split text into segments using the chosen strategy.

    Strategies:
      - paragraphs: split on blank lines, sub-split long paragraphs via chunk_text()
      - sentences: feed entire text through chunk_text()
      - fixed: split at max_chars respecting word boundaries
    """
    if strategy == "paragraphs":
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        segments = []
        for para in paragraphs:
            if len(para) <= max_chars:
                segments.append(para)
            else:
                segments.extend(chunk_text(para, max_chars))
        return segments if segments else [text]

    elif strategy == "sentences":
        return chunk_text(text, max_chars)

    elif strategy == "fixed":
        segments = []
        remaining = text
        while remaining:
            if len(remaining) <= max_chars:
                segments.append(remaining.strip())
                break
            # Find last space within limit
            cut = remaining[:max_chars].rfind(" ")
            if cut <= 0:
                cut = max_chars
            segments.append(remaining[:cut].strip())
            remaining = remaining[cut:].lstrip()
        return [s for s in segments if s]

    else:
        return chunk_text(text, max_chars)


def _is_model_cached(model_id: str) -> bool:
    """Check if a model is already in the HuggingFace cache."""
    try:
        from huggingface_hub import try_to_load_from_cache
        result = try_to_load_from_cache(model_id, "config.json")
        return result is not None
    except Exception:
        return False


def _get_model_size(model_id: str) -> str | None:
    """Fetch total download size of a model from HuggingFace Hub."""
    try:
        from huggingface_hub import model_info
        info = model_info(model_id, files_metadata=True)
        total = sum(s.size for s in info.siblings if s.size)
        if total > 0:
            return f"{total / (1024 ** 3):.1f} GB"
    except Exception:
        pass
    return None


def _confirm_download(model_id: str) -> bool:
    """Ask user to confirm model download, showing size if available."""
    size = _get_model_size(model_id)
    size_str = f" ({size})" if size else ""
    try:
        answer = input(f"Model {model_id}{size_str} not found locally. Download? [Y/n] ")
        return answer.strip().lower() in ("", "y", "yes")
    except (EOFError, KeyboardInterrupt):
        return False


def load_model_with_fallback(model_override=None):
    """Try loading models in preference order, prompting before download."""
    from mlx_audio.tts.utils import load_model

    models = [model_override] if model_override else MODELS

    for model_id in models:
        if not _is_model_cached(model_id):
            if not _confirm_download(model_id):
                print(f"Skipping {model_id}")
                continue

        print(f"Loading model: {model_id}")
        try:
            transformers_logger = logging.getLogger("transformers")
            prev_level = transformers_logger.level
            transformers_logger.setLevel(logging.ERROR)
            try:
                model = load_model(model_id)
            finally:
                transformers_logger.setLevel(prev_level)

            # Workaround: newer transformers breaks fix_mistral_regex kwarg
            # in post_load_hook, so the tokenizer may silently fail to load.
            if getattr(model, "tokenizer", None) is None:
                from transformers import AutoTokenizer
                from huggingface_hub import snapshot_download
                model_path = snapshot_download(model_id)
                model.tokenizer = AutoTokenizer.from_pretrained(model_path)
                print("  Loaded tokenizer (workaround)")

            print(f"Model loaded successfully: {model_id}")
            return model, model_id
        except Exception as e:
            error_msg = str(e).lower()
            if "memory" in error_msg or "oom" in error_msg or "alloc" in error_msg:
                print(f"Out of memory with {model_id}, trying smaller model...")
                continue
            raise

    print("Error: Could not load any model. Try the 0.6B model explicitly:", file=sys.stderr)
    print("  --model mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16", file=sys.stderr)
    sys.exit(1)


def run_test(model):
    """Quick test with a built-in voice."""
    print("\n=== Quick Test (built-in voice) ===")
    test_text = "Hello! This is a quick test of the voice synthesis system running on Apple Silicon."

    start = time.time()
    results = list(model.generate(
        text=test_text,
        voice="Chelsie",
        language="English",
    ))
    elapsed = time.time() - start

    if not results:
        print("Error: No audio generated in test.", file=sys.stderr)
        return False

    audio = np.array(results[0].audio)
    sr = getattr(results[0], "sample_rate", SAMPLE_RATE)
    duration = len(audio) / sr
    rtf = elapsed / duration if duration > 0 else float("inf")

    os.makedirs("output", exist_ok=True)
    sf.write("output/test_output.wav", audio, sr)

    print(f"Audio duration:  {duration:.2f}s")
    print(f"Generation time: {elapsed:.2f}s")
    print(f"Realtime factor: {rtf:.2f}x (< 1.0 = faster than realtime)")
    print(f"Output saved:    output/test_output.wav")
    return True


def generate_cloned(model, text, ref_audio, ref_text, language, speed):
    """Generate speech using voice cloning."""
    kwargs = dict(
        text=text,
        ref_audio=ref_audio,
        language=language,
    )
    if ref_text:
        kwargs["ref_text"] = ref_text
    if speed != 1.0:
        kwargs["speed"] = speed

    results = list(model.generate(**kwargs))
    if not results:
        return None, 0

    audio = np.array(results[0].audio)
    sr = getattr(results[0], "sample_rate", SAMPLE_RATE)
    return audio, sr


def main():
    args = parse_args()

    # Load model
    model, model_id = load_model_with_fallback(args.model)

    # Quick test mode
    if args.test:
        success = run_test(model)
        sys.exit(0 if success else 1)

    # Validate ref audio
    if not os.path.isfile(args.ref_audio):
        print(f"Error: Reference audio not found: {args.ref_audio}", file=sys.stderr)
        print("Run prep_reference.sh first to create a reference clip.", file=sys.stderr)
        sys.exit(1)

    # Get text
    text = get_text(args.text)
    if not text:
        print("Error: No text provided.", file=sys.stderr)
        sys.exit(1)

    print(f"\nModel:     {model_id}")
    print(f"Ref audio: {args.ref_audio}")
    print(f"Language:  {args.language}")
    print(f"Speed:     {args.speed}")
    print(f"Text:      {text[:80]}{'...' if len(text) > 80 else ''}")

    # Chunk text for long inputs
    chunks = chunk_text(text)
    print(f"Chunks:    {len(chunks)}")

    all_audio = []
    total_start = time.time()

    for i, chunk in enumerate(chunks, 1):
        print(f"\n--- Generating chunk {i}/{len(chunks)} ({len(chunk)} chars) ---")
        chunk_start = time.time()

        try:
            audio, sr = generate_cloned(
                model, chunk, args.ref_audio, args.ref_text,
                args.language, args.speed,
            )
        except Exception as e:
            print(f"Error generating chunk {i}: {e}", file=sys.stderr)
            if "memory" in str(e).lower():
                print("Hint: Try the 0.6B model with --model mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16", file=sys.stderr)
            sys.exit(1)

        if audio is None:
            print(f"Warning: No audio for chunk {i}, skipping.", file=sys.stderr)
            continue

        chunk_elapsed = time.time() - chunk_start
        chunk_dur = len(audio) / sr
        print(f"  Duration: {chunk_dur:.2f}s, Time: {chunk_elapsed:.2f}s")
        all_audio.append(audio)

    if not all_audio:
        print("Error: No audio generated.", file=sys.stderr)
        sys.exit(1)

    # Concatenate chunks with small silence gap
    silence = np.zeros(int(sr * 0.15))  # 150ms gap between chunks
    combined = []
    for i, seg in enumerate(all_audio):
        combined.append(seg)
        if i < len(all_audio) - 1:
            combined.append(silence)
    final_audio = np.concatenate(combined)

    # Save output
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    sf.write(args.output, final_audio, sr)

    total_elapsed = time.time() - total_start
    total_duration = len(final_audio) / sr
    rtf = total_elapsed / total_duration if total_duration > 0 else float("inf")

    print(f"\n=== Results ===")
    print(f"Audio duration:  {total_duration:.2f}s")
    print(f"Generation time: {total_elapsed:.2f}s")
    print(f"Realtime factor: {rtf:.2f}x")
    print(f"Output saved:    {args.output}")


if __name__ == "__main__":
    main()
