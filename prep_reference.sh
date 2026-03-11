#!/usr/bin/env bash
# prep_reference.sh — Preprocess audio for voice cloning reference
# Usage: ./prep_reference.sh <input_audio> [output_path]

set -euo pipefail

INPUT="${1:?Usage: $0 <input_audio> [output_path]}"
OUTPUT="${2:-ref_audio/ref_audio.wav}"

if [ ! -f "$INPUT" ]; then
    echo "Error: Input file '$INPUT' not found." >&2
    exit 1
fi

if ! command -v ffmpeg &>/dev/null; then
    echo "Error: ffmpeg not found. Install with: brew install ffmpeg" >&2
    exit 1
fi

# Ensure output directory exists
mkdir -p "$(dirname "$OUTPUT")"

TMPDIR_WORK="$(mktemp -d)"
trap 'rm -rf "$TMPDIR_WORK"' EXIT

STEP1="$TMPDIR_WORK/step1_cleaned.wav"
STEP2="$TMPDIR_WORK/step2_trimmed.wav"

echo "==> Processing: $INPUT"

# Step 1: Convert to mono 24kHz 16-bit, apply filters + noise reduction + compression
ffmpeg -y -i "$INPUT" \
    -af "
        aformat=sample_fmts=s16:sample_rates=24000:channel_layouts=mono,
        highpass=f=80,
        lowpass=f=8000,
        afftdn=nf=-23,
        acompressor=threshold=-21dB:ratio=2.5:attack=4:release=60:makeup=2dB
    " \
    -ar 24000 -ac 1 -sample_fmt s16 \
    "$STEP1" 2>/dev/null

echo "==> Filters applied"

# Step 2: Trim silence from start and end (threshold -40dB, min silence 0.3s)
ffmpeg -y -i "$STEP1" \
    -af "silenceremove=start_periods=1:start_silence=0.1:start_threshold=-40dB,areverse,silenceremove=start_periods=1:start_silence=0.1:start_threshold=-40dB,areverse" \
    -ar 24000 -ac 1 -sample_fmt s16 \
    "$STEP2" 2>/dev/null

echo "==> Silence trimmed"

# Step 3: Check duration — if >10s, extract the loudest 10s segment
DURATION=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$STEP2")
DURATION_INT=${DURATION%.*}

if [ "$DURATION_INT" -gt 10 ]; then
    echo "==> Clip is ${DURATION_INT}s — selecting best 10s segment"

    # Find the loudest 10-second window by scanning with 1s steps
    BEST_START=0
    BEST_VOLUME=-999

    MAX_START=$((DURATION_INT - 10))
    for START in $(seq 0 1 "$MAX_START"); do
        VOL=$(ffmpeg -ss "$START" -t 10 -i "$STEP2" \
            -af "volumedetect" -f null /dev/null 2>&1 \
            | grep "mean_volume" | sed 's/.*mean_volume: //' | sed 's/ dB//')

        if [ -n "$VOL" ]; then
            # Compare as floating point (higher = louder, less negative)
            IS_BETTER=$(python3 -c "print(1 if float('${VOL}') > float('${BEST_VOLUME}') else 0)")
            if [ "$IS_BETTER" = "1" ]; then
                BEST_VOLUME="$VOL"
                BEST_START="$START"
            fi
        fi
    done

    echo "==> Best segment starts at ${BEST_START}s (mean volume: ${BEST_VOLUME}dB)"

    ffmpeg -y -ss "$BEST_START" -t 10 -i "$STEP2" \
        -ar 24000 -ac 1 -sample_fmt s16 \
        "$OUTPUT" 2>/dev/null
else
    cp "$STEP2" "$OUTPUT"
fi

# Print final info
FINAL_DUR=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$OUTPUT")
echo "==> Output: $OUTPUT (${FINAL_DUR}s)"
echo "==> Done!"
