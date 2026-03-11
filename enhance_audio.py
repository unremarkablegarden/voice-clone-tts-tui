#!/usr/bin/env python3
"""AI audio enhancement using Resemble Enhance (UNet denoiser + diffusion enhancer)."""

import argparse
import sys
from pathlib import Path
from typing import Callable


def enhance_audio(
    input_path: str,
    output_path: str | None = None,
    denoise_only: bool = False,
    nfe: int = 32,
    solver: str = "midpoint",
    lambd: float = 0.5,
    tau: float = 0.5,
    target_sr: int = 44100,
    device: str | None = None,
    on_progress: Callable[[str], None] | None = None,
) -> str:
    """Enhance audio using Resemble Enhance.

    Args:
        input_path: Path to input audio file.
        output_path: Path for output. Defaults to {stem}_enhanced.wav.
        denoise_only: If True, skip diffusion enhancer (faster).
        nfe: Number of function evaluations for diffusion (1-128).
        solver: ODE solver: midpoint, rk4, or euler.
        lambd: Denoise strength (0.0-1.0).
        tau: Prior temperature (0.0-1.0).
        target_sr: Output sample rate (Resemble outputs 44.1kHz).
        device: Force device (mps/cpu). Auto-detects if None.
        on_progress: Optional callback for status updates.

    Returns:
        Path to the enhanced output file.
    """
    import torch
    import torchaudio
    import soundfile as sf
    import numpy as np

    def log(msg: str) -> None:
        if on_progress:
            on_progress(msg)
        else:
            print(msg)

    # Resolve output path
    inp = Path(input_path)
    if output_path is None:
        output_path = str(inp.with_name(f"{inp.stem}_enhanced.wav"))

    # Force CPU — MPS produces near-silent output due to numerical issues in the diffusion model
    device = "cpu"
    log(f"Using device: {device}")

    # Load audio via soundfile (avoids torchaudio backend issues)
    log("Loading audio...")
    audio_np, orig_sr = sf.read(input_path, dtype="float32")
    if audio_np.ndim > 1:
        audio_np = audio_np.mean(axis=1)
    dwav = torch.from_numpy(audio_np)

    # Import resemble enhance inference
    from resemble_enhance.enhancer.inference import denoise, enhance

    # Run enhancement
    if denoise_only:
        log("Running AI denoiser...")
        enhanced, new_sr = denoise(dwav, orig_sr, device)
    else:
        log(f"Running AI enhancer (nfe={nfe}, solver={solver})...")
        enhanced, new_sr = enhance(
            dwav, orig_sr, device,
            nfe=nfe,
            solver=solver,
            lambd=lambd,
            tau=tau,
        )

    # Resample to target SR if needed
    if new_sr != target_sr:
        log(f"Resampling {new_sr}Hz → {target_sr}Hz...")
        enhanced = torchaudio.functional.resample(enhanced, new_sr, target_sr)

    # Save output
    log(f"Saving to {output_path}")
    audio_np = enhanced.cpu().numpy()
    sf.write(output_path, audio_np, target_sr)

    log("Enhancement complete.")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="AI audio enhancement with Resemble Enhance")
    parser.add_argument("input", help="Input audio file")
    parser.add_argument("output", nargs="?", default=None, help="Output audio file (default: {stem}_enhanced.wav)")
    parser.add_argument("--denoise-only", action="store_true", help="Fast mode: denoiser only, skip diffusion enhancer")
    parser.add_argument("--nfe", type=int, default=32, help="Quality steps for diffusion (1-128, default: 32)")
    parser.add_argument("--solver", default="midpoint", choices=["midpoint", "rk4", "euler"], help="ODE solver (default: midpoint)")
    parser.add_argument("--lambd", type=float, default=0.5, help="Denoise strength 0-1 (default: 0.5)")
    parser.add_argument("--tau", type=float, default=0.5, help="Prior temperature 0-1 (default: 0.5)")
    parser.add_argument("--device", default=None, choices=["mps", "cpu"], help="Force device (default: auto)")
    args = parser.parse_args()

    if not Path(args.input).is_file():
        print(f"Error: Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    try:
        out = enhance_audio(
            input_path=args.input,
            output_path=args.output,
            denoise_only=args.denoise_only,
            nfe=args.nfe,
            solver=args.solver,
            lambd=args.lambd,
            tau=args.tau,
            device=args.device,
        )
        print(f"Output: {out}")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
