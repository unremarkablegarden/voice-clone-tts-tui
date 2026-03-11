#!/usr/bin/env python3
"""Minimal Gradio web UI for voice cloning with Qwen3-TTS."""

import os
import subprocess
import tempfile
import time

import gradio as gr
import numpy as np
import soundfile as sf

from clone_voice import chunk_text, generate_cloned, load_model_with_fallback

# Load model once at startup
print("Loading TTS model...")
MODEL, MODEL_ID = load_model_with_fallback()
print(f"Ready: {MODEL_ID}")

LANGUAGES = [
    "English", "Chinese", "Japanese", "Korean", "French",
    "German", "Spanish", "Italian", "Portuguese", "Russian",
    "Arabic", "Thai", "Indonesian", "Vietnamese",
]


def preprocess_audio(audio_path: str) -> str:
    """Run prep_reference.sh on uploaded audio."""
    if not audio_path:
        return None

    script = os.path.join(os.path.dirname(__file__), "prep_reference.sh")
    output_path = os.path.join(
        os.path.dirname(__file__), "ref_audio", "ref_audio_ui.wav"
    )

    result = subprocess.run(
        [script, audio_path, output_path],
        capture_output=True,
        text=True,
        cwd=os.path.dirname(__file__),
    )

    if result.returncode != 0:
        raise gr.Error(f"Preprocessing failed: {result.stderr}")

    return output_path


def generate(
    ref_audio_path: str,
    ref_text: str,
    text: str,
    language: str,
    speed: float,
):
    """Generate cloned speech."""
    if not ref_audio_path:
        raise gr.Error("Please upload a reference audio file.")
    if not text.strip():
        raise gr.Error("Please enter text to synthesize.")

    # Clear any currently playing audio before starting
    yield "Starting...", None

    # Preprocess reference audio
    status = "Preprocessing reference audio..."
    yield status, None

    try:
        processed_path = preprocess_audio(ref_audio_path)
    except Exception as e:
        raise gr.Error(f"Audio preprocessing failed: {e}")

    # Generate speech
    chunks = chunk_text(text.strip())
    all_audio = []
    sr = 24000

    total_start = time.time()

    for i, chunk in enumerate(chunks, 1):
        yield f"Generating chunk {i}/{len(chunks)}...", None

        audio, sr = generate_cloned(
            MODEL, chunk, processed_path, ref_text.strip(),
            language, speed,
        )
        if audio is not None:
            all_audio.append(audio)

    if not all_audio:
        raise gr.Error("No audio was generated. Try different text or reference audio.")

    # Concatenate with crossfade to avoid clicks at chunk boundaries
    crossfade_samples = int(sr * 0.05)  # 50ms crossfade
    gap_samples = int(sr * 0.10)  # 100ms silence between chunks
    gap = np.zeros(gap_samples)

    final_audio = all_audio[0]
    for seg in all_audio[1:]:
        # Crossfade: fade out tail of previous, fade in head of next
        fade_len = min(crossfade_samples, len(final_audio), len(seg))
        fade_out = np.linspace(1.0, 0.0, fade_len)
        fade_in = np.linspace(0.0, 1.0, fade_len)
        final_audio[-fade_len:] *= fade_out
        seg = seg.copy()
        seg[:fade_len] *= fade_in
        # Overlap-add the crossfade region
        overlap = final_audio[-fade_len:] + seg[:fade_len]
        final_audio = np.concatenate([
            final_audio[:-fade_len], overlap, gap, seg[fade_len:]
        ])

    # Save
    os.makedirs(os.path.join(os.path.dirname(__file__), "output"), exist_ok=True)
    output_path = os.path.join(os.path.dirname(__file__), "output", "ui_output.wav")
    sf.write(output_path, final_audio, sr)

    elapsed = time.time() - total_start
    duration = len(final_audio) / sr
    rtf = elapsed / duration if duration > 0 else 0

    status = f"Done! Duration: {duration:.1f}s | Generated in {elapsed:.1f}s | RTF: {rtf:.2f}x"
    yield status, output_path


STOP_AUDIO_JS = """
() => {
    document.querySelectorAll('audio').forEach(a => {
        a.pause();
        a.currentTime = 0;
    });
}
"""


def build_ui():
    with gr.Blocks(title="Voice Clone — Qwen3-TTS") as app:
        gr.Markdown("# Voice Clone — Qwen3-TTS on Apple Silicon")
        gr.Markdown(f"**Model:** `{MODEL_ID}`")

        with gr.Row():
            with gr.Column():
                ref_audio = gr.Audio(
                    label="Reference Audio",
                    type="filepath",
                    sources=["upload", "microphone"],
                )
                ref_text = gr.Textbox(
                    label="Reference transcript (optional, improves quality)",
                    placeholder="What is said in the reference audio...",
                    lines=2,
                )
                text_input = gr.Textbox(
                    label="Text to synthesize",
                    placeholder="Enter the text you want spoken in the cloned voice...",
                    lines=15,
                )
                with gr.Row():
                    language = gr.Dropdown(
                        choices=LANGUAGES,
                        value="English",
                        label="Language",
                    )
                    speed = gr.Slider(
                        minimum=0.5,
                        maximum=2.0,
                        value=1.0,
                        step=0.1,
                        label="Speed",
                    )
                with gr.Row():
                    generate_btn = gr.Button("Generate", variant="primary")
                    clear_btn = gr.Button("Clear")

            with gr.Column():
                status_box = gr.Textbox(label="Status", interactive=False)
                output_audio = gr.Audio(label="Generated Audio", type="filepath")

        # Stop any playing audio, then run generation
        generate_btn.click(
            fn=None,
            inputs=None,
            outputs=None,
            js=STOP_AUDIO_JS,
        ).then(
            fn=generate,
            inputs=[ref_audio, ref_text, text_input, language, speed],
            outputs=[status_box, output_audio],
        )

        clear_btn.click(
            fn=lambda: "",
            inputs=None,
            outputs=[text_input],
        )

    return app


if __name__ == "__main__":
    app = build_ui()
    app.launch(server_name="127.0.0.1", server_port=7860)
