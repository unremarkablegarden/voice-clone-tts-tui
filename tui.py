#!/usr/bin/env python3
"""Terminal UI for voice cloning with Qwen3-TTS on Apple Silicon."""

import json
import os
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf
from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import (
    Button,
    Checkbox,
    Collapsible,
    Footer,
    Header,
    Input,
    Label,
    RichLog,
    Select,
    Static,
    TabbedContent,
    TabPane,
    TextArea,
)

from clone_voice import (
    SAMPLE_RATE,
    chunk_text,
    clean_markdown,
    load_model_with_fallback,
    split_into_segments,
)

LANGUAGES = [
    ("English", "english"),
    ("Chinese", "chinese"),
    ("Japanese", "japanese"),
    ("Korean", "korean"),
    ("French", "french"),
    ("German", "german"),
    ("Spanish", "spanish"),
    ("Italian", "italian"),
    ("Portuguese", "portuguese"),
    ("Russian", "russian"),
    ("Arabic", "arabic"),
    ("Thai", "thai"),
    ("Indonesian", "indonesian"),
    ("Vietnamese", "vietnamese"),
]

REF_AUDIO_DIR = Path(__file__).parent / "ref_audio"
VOICES_DIR = Path(__file__).parent / "voices"

OUTPUT_DIR = Path(__file__).parent / "output"
AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac"}
CONFIG_PATH = Path(__file__).parent / ".tui_config.json"


def _load_config() -> dict:
    try:
        return json.loads(CONFIG_PATH.read_text()) if CONFIG_PATH.is_file() else {}
    except (json.JSONDecodeError, OSError):
        return {}


def _save_config(config: dict) -> None:
    try:
        CONFIG_PATH.write_text(json.dumps(config, indent=2))
    except OSError:
        pass


def scan_ref_audio() -> list[tuple[str, str]]:
    """Return (display_name, full_path) pairs for audio files in ref_audio/."""
    if not REF_AUDIO_DIR.is_dir():
        return []
    files = sorted(
        f for f in REF_AUDIO_DIR.iterdir()
        if f.is_file() and f.suffix.lower() in AUDIO_EXTENSIONS
    )
    return [(f.name, str(f)) for f in files]


def scan_voices() -> list[tuple[str, str]]:
    """Scan voices/ subdirs for meta.json, return (name, dir_path) pairs."""
    if not VOICES_DIR.is_dir():
        return []
    voices = []
    for d in sorted(VOICES_DIR.iterdir()):
        meta_path = d / "meta.json"
        if d.is_dir() and meta_path.is_file():
            try:
                meta = json.loads(meta_path.read_text())
                voices.append((meta.get("name", d.name), str(d)))
            except (json.JSONDecodeError, OSError):
                voices.append((d.name, str(d)))
    return voices


def scan_audio_files() -> list[tuple[str, str]]:
    """Find audio files in output/ and ref_audio/ available for enhancement."""
    results = []
    for scan_dir, label in [(OUTPUT_DIR, "output"), (REF_AUDIO_DIR, "ref_audio")]:
        if not scan_dir.is_dir():
            continue
        for f in sorted(scan_dir.iterdir()):
            if not f.is_file() or f.suffix.lower() not in AUDIO_EXTENSIONS:
                continue
            size_kb = f.stat().st_size / 1024
            if size_kb >= 1024:
                size_str = f"{size_kb / 1024:.1f} MB"
            else:
                size_str = f"{size_kb:.0f} KB"
            results.append((f"{label}/{f.name} ({size_str})", str(f)))
    return results


def load_voice_meta(voice_dir: str | Path) -> dict:
    """Read and parse meta.json from a voice directory."""
    meta_path = Path(voice_dir) / "meta.json"
    if meta_path.is_file():
        return json.loads(meta_path.read_text())
    return {}


class VoiceCloneApp(App):
    """Voice Clone TUI — Qwen3-TTS on Apple Silicon."""

    TITLE = "Voice Clone — Qwen3-TTS"

    CSS = """
    TabbedContent {
        height: 1fr;
    }
    TabPane {
        height: 1fr;
    }
    TabPane > Horizontal {
        height: 1fr;
    }
    #create-panel {
        width: 1fr;
        padding: 1 2;
        border-right: solid $accent;
        align: left top;
    }
    #create-log {
        width: 1fr;
        padding: 1 2;
        height: 1fr;
        min-height: 10;
        border: solid $surface-lighten-2;
    }
    #gen-left-panel {
        width: 1fr;
        padding: 1 2;
        border-right: solid $accent;
    }
    #gen-fields {
        height: auto;
    }
    #gen-btns {
        dock: bottom;
        height: auto;
        layout: horizontal;
    }
    #gen-btns Button {
        width: 1fr;
    }
    #gen-right-panel {
        width: 1fr;
        padding: 1 2;
    }
    #gen-log {
        height: 1fr;
        min-height: 10;
        border: solid $surface-lighten-2;
    }
    #stats {
        height: auto;
        max-height: 8;
        border: solid $surface-lighten-2;
        padding: 0 1;
        margin-top: 1;
    }
    #audio-controls {
        height: auto;
        margin-top: 1;
        layout: horizontal;
    }
    #audio-controls Button {
        margin-right: 1;
    }
    #create-voice-btn {
        margin-top: 1;
        width: 100%;
    }
    .field-label {
        color: $text-muted;
    }
    #source-audio-select, #voice-select {
        width: 100%;
    }
    #ref-text {
        height: 10;
    }
    #synth-text {
        height: 20;
    }
    #speed-lang-row {
        height: auto;
    }
    #speed-lang-row > * {
        width: 1fr;
        margin-right: 1;
    }
    .quality-row {
        layout: horizontal;
        height: auto;
        margin-bottom: 1;
    }
    .quality-row Label {
        width: 16;
        content-align: left middle;
    }
    .quality-row Input {
        width: 1fr;
    }
    #cleanup-panel {
        width: 1fr;
        padding: 1 2;
        border-right: solid $accent;
    }
    #cleanup-log {
        width: 1fr;
        padding: 1 2;
        height: 1fr;
        min-height: 10;
        border: solid $surface-lighten-2;
    }
    #cleanup-btns {
        height: auto;
        margin-top: 1;
    }
    #cleanup-btns Button {
        margin-right: 1;
    }
    #batch-left-panel {
        width: 1fr;
        padding: 1 2;
        border-right: solid $accent;
    }
    #batch-fields {
        height: auto;
    }
    #batch-btns {
        dock: bottom;
        height: auto;
        layout: horizontal;
    }
    #batch-btns Button {
        width: 1fr;
    }
    #batch-right-panel {
        width: 1fr;
        padding: 1 2;
    }
    #batch-log {
        height: 1fr;
        min-height: 10;
        border: solid $surface-lighten-2;
    }
    #batch-stats {
        height: auto;
        max-height: 8;
        border: solid $surface-lighten-2;
        padding: 0 1;
        margin-top: 1;
    }
    #batch-speed-lang-row {
        height: auto;
    }
    #batch-speed-lang-row > * {
        width: 1fr;
        margin-right: 1;
    }
    """

    BINDINGS = [
        Binding("ctrl+g", "generate", "Generate"),
        Binding("ctrl+p", "play", "Play"),
        Binding("ctrl+s", "save_as", "Save As"),
        Binding("ctrl+t", "change_theme", "Theme"),
        Binding("q", "quit", "Quit", show=True),
    ]

    def __init__(self, model, model_id: str):
        super().__init__()
        self.model = model
        self.model_id = model_id
        self.generated_audio = None
        self.generated_sr = SAMPLE_RATE
        self.output_path = None
        self.batch_output_dir: Path | None = None
        self._batch_cancelled = False

    def compose(self) -> ComposeResult:
        yield Header()

        with TabbedContent("Create Voice", "Generate", "Batch", "Cleanup"):
            with TabPane("Create Voice", id="create-tab"):
                with Horizontal():
                    with Vertical(id="create-panel"):
                        yield Label("Source Audio", classes="field-label")
                        yield Select(
                            scan_ref_audio(),
                            prompt="Select source audio",
                            id="source-audio-select",
                        )
                        yield Checkbox("Preprocess (clean & trim)", value=True, id="preprocess-check")
                        yield Checkbox("Enhance (AI cleanup)", value=False, id="enhance-check")

                        yield Label("Reference Transcript (optional)", classes="field-label")
                        yield TextArea(id="ref-text")

                        yield Label("Voice Name", classes="field-label")
                        yield Input(placeholder="e.g. adam-curtis", id="voice-name-input")

                        yield Button(
                            "Create Voice", id="create-voice-btn", variant="primary"
                        )

                    yield RichLog(id="create-log", highlight=True, markup=True)

            with TabPane("Generate", id="generate-tab"):
                with Horizontal():
                    with Vertical(id="gen-left-panel"):
                        with Vertical(id="gen-fields"):
                            yield Label("Voice", classes="field-label")
                            yield Select(
                                scan_voices(),
                                prompt="Select a voice",
                                id="voice-select",
                            )

                            yield Label("Text to Synthesize", classes="field-label")
                            yield TextArea(id="synth-text")

                            with Horizontal(id="speed-lang-row"):
                                with Vertical():
                                    yield Label("Language")
                                    yield Select(LANGUAGES, value="english", id="language-select")
                                with Vertical():
                                    yield Label("Speed")
                                    yield Input("1.0", id="speed-input")

                            with Collapsible(title="Quality Settings"):
                                with Horizontal(classes="quality-row"):
                                    yield Label("Temperature")
                                    yield Input("0.9", id="q-temperature")
                                with Horizontal(classes="quality-row"):
                                    yield Label("Top-K")
                                    yield Input("50", id="q-top-k")
                                with Horizontal(classes="quality-row"):
                                    yield Label("Top-P")
                                    yield Input("1.0", id="q-top-p")
                                with Horizontal(classes="quality-row"):
                                    yield Label("Rep Penalty")
                                    yield Input("1.05", id="q-rep-penalty")

                        with Horizontal(id="gen-btns"):
                            yield Button(
                                "Generate", id="generate-btn", variant="primary", disabled=True
                            )
                            yield Button("Clear", id="clear-btn")

                    with Vertical(id="gen-right-panel"):
                        yield RichLog(id="gen-log", highlight=True, markup=True)
                        yield Static("", id="stats")
                        with Horizontal(id="audio-controls"):
                            yield Button("Play", id="play-btn", variant="success", disabled=True)
                            yield Button("Stop", id="stop-btn", variant="error", disabled=True)
                            yield Button("Save As", id="save-btn", variant="default", disabled=True)
                            yield Button("Enhance", id="enhance-btn", variant="warning", disabled=True)

            with TabPane("Batch", id="batch-tab"):
                with Horizontal():
                    with Vertical(id="batch-left-panel"):
                        with Vertical(id="batch-fields"):
                            yield Label("Voice", classes="field-label")
                            yield Select(
                                scan_voices(),
                                prompt="Select a voice",
                                id="batch-voice-select",
                            )

                            yield Label("Source File (.txt / .md)", classes="field-label")
                            yield Input(placeholder="path/to/source.txt", id="batch-source-input")

                            yield Label("Output Directory", classes="field-label")
                            yield Input(placeholder="auto: output/batch-TIMESTAMP/", id="batch-output-input")

                            with Horizontal(id="batch-speed-lang-row"):
                                with Vertical():
                                    yield Label("Segmentation")
                                    yield Select(
                                        [
                                            ("Paragraphs", "paragraphs"),
                                            ("Sentences", "sentences"),
                                            ("Fixed chars", "fixed"),
                                        ],
                                        value="paragraphs",
                                        id="batch-strategy-select",
                                    )
                                with Vertical():
                                    yield Label("Max chars")
                                    yield Input("200", id="batch-max-chars")

                            with Horizontal(id="batch-speed-lang-row"):
                                with Vertical():
                                    yield Label("Language")
                                    yield Select(LANGUAGES, value="english", id="batch-language-select")
                                with Vertical():
                                    yield Label("Speed")
                                    yield Input("1.0", id="batch-speed-input")

                            with Collapsible(title="Quality Settings"):
                                with Horizontal(classes="quality-row"):
                                    yield Label("Temperature")
                                    yield Input("0.9", id="bq-temperature")
                                with Horizontal(classes="quality-row"):
                                    yield Label("Top-K")
                                    yield Input("50", id="bq-top-k")
                                with Horizontal(classes="quality-row"):
                                    yield Label("Top-P")
                                    yield Input("1.0", id="bq-top-p")
                                with Horizontal(classes="quality-row"):
                                    yield Label("Rep Penalty")
                                    yield Input("1.05", id="bq-rep-penalty")

                            with Collapsible(title="Join Settings"):
                                with Horizontal(classes="quality-row"):
                                    yield Label("Silence gap (ms)")
                                    yield Input("300", id="batch-silence-gap")

                        with Horizontal(id="batch-btns"):
                            yield Button("Start Batch", id="batch-start-btn", variant="primary")
                            yield Button("Cancel", id="batch-cancel-btn", variant="error", disabled=True)
                            yield Button("Join All", id="batch-join-btn", variant="success", disabled=True)

                    with Vertical(id="batch-right-panel"):
                        yield RichLog(id="batch-log", highlight=True, markup=True)
                        yield Static("", id="batch-stats")

            with TabPane("Cleanup", id="cleanup-tab"):
                with Horizontal():
                    with Vertical(id="cleanup-panel"):
                        yield Label("Audio files", classes="field-label")
                        yield Select(
                            scan_audio_files(),
                            prompt="Select a file",
                            id="cleanup-select",
                        )
                        with Horizontal(id="cleanup-btns"):
                            yield Button("Enhance Selected", id="cleanup-enhance-btn", variant="warning")
                            yield Button("Enhance All", id="cleanup-enhance-all-btn", variant="warning")
                            yield Button("Refresh", id="cleanup-refresh-btn", variant="default")
                    yield RichLog(id="cleanup-log", highlight=True, markup=True)

        yield Footer()

    def on_mount(self) -> None:
        saved_theme = _load_config().get("theme")
        if saved_theme and saved_theme in self.available_themes:
            self.theme = saved_theme
        self.sub_title = self.model_id
        self.gen_log_message(f"[green]Model loaded: {self.model_id}[/green]")
        self.query_one("#generate-btn", Button).disabled = False

    def watch_theme(self, new_theme: str) -> None:
        config = _load_config()
        if config.get("theme") != new_theme:
            config["theme"] = new_theme
            _save_config(config)

    def create_log_message(self, msg: str) -> None:
        self.query_one("#create-log", RichLog).write(msg)

    def gen_log_message(self, msg: str) -> None:
        self.query_one("#gen-log", RichLog).write(msg)

    def cleanup_log_message(self, msg: str) -> None:
        self.query_one("#cleanup-log", RichLog).write(msg)

    def batch_log_message(self, msg: str) -> None:
        self.query_one("#batch-log", RichLog).write(msg)

    def _refresh_batch_voices(self) -> None:
        sel = self.query_one("#batch-voice-select", Select)
        sel.set_options(scan_voices())

    def _refresh_voices(self) -> None:
        """Rescan voices/ and update the voice selector."""
        sel = self.query_one("#voice-select", Select)
        sel.set_options(scan_voices())

    def on_button_pressed(self, event: Button.Pressed) -> None:
        match event.button.id:
            case "generate-btn":
                self.action_generate()
            case "create-voice-btn":
                self.create_voice()
            case "play-btn":
                self.action_play()
            case "stop-btn":
                self.stop_audio()
            case "save-btn":
                self.action_save_as()
            case "enhance-btn":
                self.enhance_output()
            case "cleanup-enhance-btn":
                self.cleanup_enhance_selected()
            case "cleanup-enhance-all-btn":
                self.cleanup_enhance_all()
            case "clear-btn":
                self.query_one("#synth-text", TextArea).clear()
            case "cleanup-refresh-btn":
                self._refresh_cleanup()
            case "batch-start-btn":
                self.start_batch()
            case "batch-cancel-btn":
                self._batch_cancelled = True
                self.batch_log_message("[yellow]Cancelling after current segment...[/yellow]")
            case "batch-join-btn":
                self.join_batch()

    # ── Create Voice ─────────────────────────────────────────────

    def create_voice(self) -> None:
        source = self.query_one("#source-audio-select", Select).value
        if source is Select.BLANK:
            self.create_log_message("[red]Please select a source audio file[/red]")
            return

        name = self.query_one("#voice-name-input", Input).value.strip()
        if not name:
            self.create_log_message("[red]Please enter a voice name[/red]")
            return

        voice_dir = VOICES_DIR / name
        if voice_dir.exists():
            self.create_log_message(f"[red]Voice '{name}' already exists[/red]")
            return

        preprocess = self.query_one("#preprocess-check", Checkbox).value
        enhance = self.query_one("#enhance-check", Checkbox).value
        ref_text = self.query_one("#ref-text", TextArea).text.strip()

        self.run_create_voice(str(source), name, preprocess, enhance, ref_text)

    @work(thread=True)
    def run_create_voice(self, source: str, name: str, preprocess: bool, enhance: bool, ref_text: str) -> None:
        voice_dir = VOICES_DIR / name
        voice_dir.mkdir(parents=True, exist_ok=True)
        out_audio = voice_dir / "ref_audio.wav"

        if preprocess:
            script = Path(__file__).parent / "prep_reference.sh"
            self.call_from_thread(
                self.create_log_message,
                f"[bold]Preprocessing: {Path(source).name}...[/bold]",
            )
            try:
                result = subprocess.run(
                    [str(script), source, str(out_audio)],
                    capture_output=True,
                    text=True,
                    cwd=str(Path(__file__).parent),
                )
                if result.returncode != 0:
                    self.call_from_thread(
                        self.create_log_message,
                        f"[red]Preprocess failed: {result.stderr.strip()}[/red]",
                    )
                    shutil.rmtree(voice_dir, ignore_errors=True)
                    return
            except Exception as e:
                self.call_from_thread(
                    self.create_log_message,
                    f"[red]Preprocess error: {e}[/red]",
                )
                shutil.rmtree(voice_dir, ignore_errors=True)
                return
        else:
            self.call_from_thread(
                self.create_log_message,
                f"[bold]Copying: {Path(source).name}...[/bold]",
            )
            shutil.copy2(source, out_audio)

        # AI enhancement step
        enhanced = False
        if enhance:
            self.call_from_thread(
                self.create_log_message,
                "[bold]Enhancing audio (AI cleanup)...[/bold]",
            )
            try:
                from enhance_audio import enhance_audio as run_enhance

                def on_progress(msg: str) -> None:
                    self.call_from_thread(self.create_log_message, f"  {msg}")

                run_enhance(
                    input_path=str(out_audio),
                    output_path=str(out_audio),
                    on_progress=on_progress,
                )
                enhanced = True
                self.call_from_thread(
                    self.create_log_message,
                    "[green]Enhancement complete[/green]",
                )
            except Exception as e:
                self.call_from_thread(
                    self.create_log_message,
                    f"[yellow]Enhancement failed (using unenhanced audio): {e}[/yellow]",
                )

        # Write meta.json
        meta = {
            "name": name,
            "transcript": ref_text,
            "enhanced": enhanced,
            "created": datetime.now(timezone.utc).isoformat(),
        }
        (voice_dir / "meta.json").write_text(json.dumps(meta, indent=2))

        self.call_from_thread(
            self.create_log_message,
            f"[bold green]Voice \"{name}\" created[/bold green]",
        )
        self.call_from_thread(self._refresh_voices)
        self.call_from_thread(self._refresh_batch_voices)

    # ── Generate ─────────────────────────────────────────────────

    def action_generate(self) -> None:
        if self.model is None:
            self.gen_log_message("[red]Model not loaded yet[/red]")
            return

        voice_dir = self.query_one("#voice-select", Select).value
        if voice_dir is Select.BLANK:
            self.gen_log_message("[red]Please select a voice[/red]")
            return

        ref_audio = str(Path(voice_dir) / "ref_audio.wav")
        if not os.path.isfile(ref_audio):
            self.gen_log_message(f"[red]Reference audio not found: {ref_audio}[/red]")
            return

        text = self.query_one("#synth-text", TextArea).text.strip()
        if not text:
            self.gen_log_message("[red]Please enter text to synthesize[/red]")
            return

        meta = load_voice_meta(voice_dir)
        ref_text = meta.get("transcript", "")

        lang = self.query_one("#language-select", Select).value

        try:
            speed = float(self.query_one("#speed-input", Input).value)
        except ValueError:
            speed = 1.0

        try:
            temperature = float(self.query_one("#q-temperature", Input).value)
            top_k = int(self.query_one("#q-top-k", Input).value)
            top_p = float(self.query_one("#q-top-p", Input).value)
            rep_penalty = float(self.query_one("#q-rep-penalty", Input).value)
        except ValueError:
            self.gen_log_message("[red]Invalid quality settings, using defaults[/red]")
            temperature, top_k, top_p, rep_penalty = 0.9, 50, 1.0, 1.05

        self.query_one("#generate-btn", Button).disabled = True
        self.run_generation(
            ref_audio, ref_text, text, lang, speed,
            temperature, top_k, top_p, rep_penalty,
        )

    @work(thread=True)
    def run_generation(
        self,
        ref_audio: str,
        ref_text: str,
        text: str,
        lang: str,
        speed: float,
        temperature: float,
        top_k: int,
        top_p: float,
        rep_penalty: float,
    ) -> None:
        chunks = chunk_text(text)
        total = len(chunks)
        self.call_from_thread(
            self.gen_log_message,
            f"[bold]Generating {total} chunk{'s' if total > 1 else ''}...[/bold]",
        )

        all_audio = []
        sr = SAMPLE_RATE
        total_start = time.time()
        peak_mem = 0.0

        for i, chunk in enumerate(chunks, 1):
            self.call_from_thread(
                self.gen_log_message,
                f"  Chunk {i}/{total} ({len(chunk)} chars)...",
            )

            try:
                kwargs = dict(
                    text=chunk,
                    ref_audio=ref_audio,
                    lang_code=lang,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    repetition_penalty=rep_penalty,
                )
                if ref_text:
                    kwargs["ref_text"] = ref_text
                if speed != 1.0:
                    kwargs["speed"] = speed

                results = list(self.model.generate(**kwargs))
                if not results:
                    self.call_from_thread(
                        self.gen_log_message,
                        f"  [yellow]No audio for chunk {i}, skipping[/yellow]",
                    )
                    continue

                result = results[0]
                audio = np.array(result.audio)
                sr = getattr(result, "sample_rate", SAMPLE_RATE)
                all_audio.append(audio)

                chunk_dur = len(audio) / sr
                mem = getattr(result, "peak_memory_usage", 0.0)
                peak_mem = max(peak_mem, mem)
                self.call_from_thread(
                    self.gen_log_message,
                    f"  [green]Chunk {i} done[/green] — {chunk_dur:.1f}s audio",
                )

            except Exception as e:
                err = str(e)
                self.call_from_thread(
                    self.gen_log_message,
                    f"  [red]Error chunk {i}: {err}[/red]",
                )
                if "memory" in err.lower():
                    self.call_from_thread(
                        self.gen_log_message,
                        "[red]Out of memory — try the 0.6B model[/red]",
                    )
                break

        if not all_audio:
            self.call_from_thread(self.gen_log_message, "[red]No audio generated[/red]")
            self.call_from_thread(self._generation_done, False)
            return

        # Concatenate with 150ms silence gaps
        silence = np.zeros(int(sr * 0.15))
        combined = []
        for j, seg in enumerate(all_audio):
            combined.append(seg)
            if j < len(all_audio) - 1:
                combined.append(silence)
        final_audio = np.concatenate(combined)

        # Save output
        os.makedirs("output", exist_ok=True)
        ts = time.strftime("%Y-%m-%d_%H.%M.%S")
        self.output_path = Path(f"output/generate-{ts}.wav")
        sf.write(str(self.output_path), final_audio, sr)

        self.generated_audio = final_audio
        self.generated_sr = sr

        elapsed = time.time() - total_start
        duration = len(final_audio) / sr
        rtf = elapsed / duration if duration > 0 else 0
        mem_gb = peak_mem / (1024 ** 3) if peak_mem > 0 else 0

        stats = (
            f"Duration: {duration:.1f}s  |  "
            f"Gen time: {elapsed:.1f}s  |  "
            f"RTF: {rtf:.2f}x"
        )
        if mem_gb > 0:
            stats += f"  |  Memory: {mem_gb:.1f} GB"

        self.call_from_thread(
            self.gen_log_message,
            f"[bold green]Done![/bold green] Saved to {self.output_path}",
        )
        self.call_from_thread(self._update_stats, stats)
        self.call_from_thread(self._generation_done, True)

    def _update_stats(self, stats: str) -> None:
        self.query_one("#stats", Static).update(stats)

    def _generation_done(self, success: bool) -> None:
        self.query_one("#generate-btn", Button).disabled = False
        if success:
            self.query_one("#play-btn", Button).disabled = False
            self.query_one("#stop-btn", Button).disabled = False
            self.query_one("#save-btn", Button).disabled = False
            self.query_one("#enhance-btn", Button).disabled = False

    # ── Enhance Output ────────────────────────────────────────────

    def enhance_output(self) -> None:
        if self.output_path is None or not self.output_path.is_file():
            self.gen_log_message("[red]No generated audio to enhance[/red]")
            return
        self.query_one("#enhance-btn", Button).disabled = True
        self.run_enhance_output()

    @work(thread=True)
    def run_enhance_output(self) -> None:
        self.call_from_thread(
            self.gen_log_message,
            "[bold]Enhancing output audio (44.1kHz)...[/bold]",
        )
        try:
            from enhance_audio import enhance_audio as run_enhance

            stem = self.output_path.stem
            enhanced_path = self.output_path.with_name(f"{stem}_enhanced.wav")

            def on_progress(msg: str) -> None:
                self.call_from_thread(self.gen_log_message, f"  {msg}")

            run_enhance(
                input_path=str(self.output_path),
                output_path=str(enhanced_path),
                target_sr=44100,
                on_progress=on_progress,
            )

            # Load enhanced audio for playback
            audio, sr = sf.read(str(enhanced_path))
            self.generated_audio = audio
            self.generated_sr = sr

            self.call_from_thread(
                self.gen_log_message,
                f"[bold green]Enhanced![/bold green] Saved to {enhanced_path}",
            )
        except Exception as e:
            self.call_from_thread(
                self.gen_log_message,
                f"[yellow]Enhancement failed (original audio kept): {e}[/yellow]",
            )
        finally:
            self.call_from_thread(
                lambda: setattr(self.query_one("#enhance-btn", Button), "disabled", False)
            )

    # ── Batch ────────────────────────────────────────────────────

    def start_batch(self) -> None:
        voice_dir = self.query_one("#batch-voice-select", Select).value
        if voice_dir is Select.BLANK:
            self.batch_log_message("[red]Please select a voice[/red]")
            return

        ref_audio = str(Path(voice_dir) / "ref_audio.wav")
        if not os.path.isfile(ref_audio):
            self.batch_log_message(f"[red]Reference audio not found: {ref_audio}[/red]")
            return

        source_path = self.query_one("#batch-source-input", Input).value.strip()
        if not source_path or not os.path.isfile(source_path):
            self.batch_log_message("[red]Please enter a valid source file path[/red]")
            return

        # Read source file
        text = Path(source_path).read_text(encoding="utf-8").strip()
        if not text:
            self.batch_log_message("[red]Source file is empty[/red]")
            return

        # Clean markdown if .md
        if source_path.lower().endswith(".md"):
            text = clean_markdown(text)
            self.batch_log_message("[dim]Cleaned markdown formatting[/dim]")

        # Output directory
        output_input = self.query_one("#batch-output-input", Input).value.strip()
        if output_input:
            out_dir = Path(output_input)
        else:
            ts = time.strftime("%Y-%m-%d_%H.%M.%S")
            out_dir = OUTPUT_DIR / f"batch-{ts}"

        # Segmentation
        strategy = self.query_one("#batch-strategy-select", Select).value
        try:
            max_chars = int(self.query_one("#batch-max-chars", Input).value)
        except ValueError:
            max_chars = 200

        segments = split_into_segments(text, strategy, max_chars)
        if not segments:
            self.batch_log_message("[red]No segments produced[/red]")
            return

        meta = load_voice_meta(voice_dir)
        ref_text = meta.get("transcript", "")
        lang = self.query_one("#batch-language-select", Select).value

        try:
            speed = float(self.query_one("#batch-speed-input", Input).value)
        except ValueError:
            speed = 1.0

        try:
            temperature = float(self.query_one("#bq-temperature", Input).value)
            top_k = int(self.query_one("#bq-top-k", Input).value)
            top_p = float(self.query_one("#bq-top-p", Input).value)
            rep_penalty = float(self.query_one("#bq-rep-penalty", Input).value)
        except ValueError:
            temperature, top_k, top_p, rep_penalty = 0.9, 50, 1.0, 1.05

        self.batch_output_dir = out_dir
        self._batch_cancelled = False
        self.query_one("#batch-start-btn", Button).disabled = True
        self.query_one("#batch-cancel-btn", Button).disabled = False

        self.run_batch(
            ref_audio, ref_text, segments, out_dir, lang, speed,
            temperature, top_k, top_p, rep_penalty,
        )

    @work(thread=True)
    def run_batch(
        self,
        ref_audio: str,
        ref_text: str,
        segments: list[str],
        out_dir: Path,
        lang: str,
        speed: float,
        temperature: float,
        top_k: int,
        top_p: float,
        rep_penalty: float,
    ) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        total = len(segments)
        self.call_from_thread(
            self.batch_log_message,
            f"[bold]Batch: {total} segments → {out_dir}[/bold]",
        )

        manifest_lines = []
        succeeded = 0
        failed = 0
        total_start = time.time()

        for i, segment in enumerate(segments, 1):
            if self._batch_cancelled:
                self.call_from_thread(
                    self.batch_log_message,
                    f"[yellow]Cancelled at segment {i}/{total}[/yellow]",
                )
                break

            self.call_from_thread(
                self.batch_log_message,
                f"  Segment {i}/{total} ({len(segment)} chars)...",
            )

            try:
                kwargs = dict(
                    text=segment,
                    ref_audio=ref_audio,
                    lang_code=lang,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    repetition_penalty=rep_penalty,
                )
                if ref_text:
                    kwargs["ref_text"] = ref_text
                if speed != 1.0:
                    kwargs["speed"] = speed

                results = list(self.model.generate(**kwargs))
                if not results:
                    self.call_from_thread(
                        self.batch_log_message,
                        f"  [yellow]No audio for segment {i}, skipping[/yellow]",
                    )
                    failed += 1
                    manifest_lines.append(f"{i:03d}\tFAILED\t{segment[:60]}")
                    continue

                result = results[0]
                audio = np.array(result.audio)
                sr = getattr(result, "sample_rate", SAMPLE_RATE)

                wav_name = f"{i:03d}.wav"
                sf.write(str(out_dir / wav_name), audio, sr)

                dur = len(audio) / sr
                succeeded += 1
                manifest_lines.append(f"{i:03d}\t{dur:.1f}s\t{segment[:60]}")
                self.call_from_thread(
                    self.batch_log_message,
                    f"  [green]Segment {i} done[/green] — {dur:.1f}s → {wav_name}",
                )

            except Exception as e:
                failed += 1
                manifest_lines.append(f"{i:03d}\tERROR\t{segment[:60]}")
                self.call_from_thread(
                    self.batch_log_message,
                    f"  [red]Error segment {i}: {e}[/red]",
                )

        # Write manifest
        manifest_path = out_dir / "manifest.txt"
        manifest_path.write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")

        elapsed = time.time() - total_start
        stats = f"Done: {succeeded} ok, {failed} failed | Time: {elapsed:.1f}s"
        self.call_from_thread(
            self.batch_log_message,
            f"[bold green]Batch complete![/bold green] {stats}",
        )
        self.call_from_thread(
            self.query_one("#batch-stats", Static).update, stats,
        )
        self.call_from_thread(self._batch_done)

    def _batch_done(self) -> None:
        self.query_one("#batch-start-btn", Button).disabled = False
        self.query_one("#batch-cancel-btn", Button).disabled = True
        self.query_one("#batch-join-btn", Button).disabled = False

    def join_batch(self) -> None:
        if self.batch_output_dir is None or not self.batch_output_dir.is_dir():
            self.batch_log_message("[red]No batch output directory found[/red]")
            return
        try:
            silence_ms = int(self.query_one("#batch-silence-gap", Input).value)
        except ValueError:
            silence_ms = 300
        self.query_one("#batch-join-btn", Button).disabled = True
        self.run_join_batch(self.batch_output_dir, silence_ms)

    @work(thread=True)
    def run_join_batch(self, out_dir: Path, silence_ms: int) -> None:
        wav_files = sorted(out_dir.glob("[0-9][0-9][0-9].wav"))
        if not wav_files:
            self.call_from_thread(
                self.batch_log_message,
                "[red]No numbered WAV files found in output directory[/red]",
            )
            self.call_from_thread(
                lambda: setattr(self.query_one("#batch-join-btn", Button), "disabled", False)
            )
            return

        self.call_from_thread(
            self.batch_log_message,
            f"[bold]Joining {len(wav_files)} files with {silence_ms}ms gaps...[/bold]",
        )

        combined = []
        sr = SAMPLE_RATE
        for wf in wav_files:
            audio, file_sr = sf.read(str(wf))
            sr = file_sr
            if combined:
                silence = np.zeros(int(sr * silence_ms / 1000))
                combined.append(silence)
            combined.append(audio)

        if not combined:
            self.call_from_thread(
                self.batch_log_message,
                "[red]No audio data to join[/red]",
            )
            self.call_from_thread(
                lambda: setattr(self.query_one("#batch-join-btn", Button), "disabled", False)
            )
            return

        joined = np.concatenate(combined)
        joined_path = out_dir / "joined.wav"
        sf.write(str(joined_path), joined, int(sr))

        duration = len(joined) / sr
        self.call_from_thread(
            self.batch_log_message,
            f"[bold green]Joined![/bold green] {duration:.1f}s → {joined_path}",
        )
        self.call_from_thread(
            lambda: setattr(self.query_one("#batch-join-btn", Button), "disabled", False)
        )

    # ── Cleanup ──────────────────────────────────────────────────

    def _refresh_cleanup(self) -> None:
        sel = self.query_one("#cleanup-select", Select)
        sel.set_options(scan_audio_files())

    def cleanup_enhance_selected(self) -> None:
        selected = self.query_one("#cleanup-select", Select).value
        if selected is Select.BLANK:
            self.cleanup_log_message("[red]Please select a file[/red]")
            return
        self.run_cleanup_enhance([selected])

    def cleanup_enhance_all(self) -> None:
        files = [path for _, path in scan_audio_files()]
        if not files:
            self.cleanup_log_message("[yellow]No unenhanced files found[/yellow]")
            return
        self.run_cleanup_enhance(files)

    @work(thread=True)
    def run_cleanup_enhance(self, paths: list[str]) -> None:
        self.call_from_thread(
            self.cleanup_log_message,
            f"[bold]Enhancing {len(paths)} file{'s' if len(paths) > 1 else ''}...[/bold]",
        )
        try:
            from enhance_audio import enhance_audio as run_enhance
        except ImportError as e:
            self.call_from_thread(
                self.cleanup_log_message,
                f"[red]Cannot import enhance_audio: {e}[/red]",
            )
            return

        for i, path in enumerate(paths, 1):
            p = Path(path)
            enhanced_path = p.with_name(f"{p.stem}_enhanced.wav")
            # 44100 for output files, 24000 for ref_audio
            target_sr = 44100 if "output" in p.parts else 24000

            self.call_from_thread(
                self.cleanup_log_message,
                f"  [{i}/{len(paths)}] {p.parent.name}/{p.name}...",
            )
            try:
                def on_progress(msg: str) -> None:
                    self.call_from_thread(self.cleanup_log_message, f"    {msg}")

                run_enhance(
                    input_path=str(p),
                    output_path=str(enhanced_path),
                    target_sr=target_sr,
                    on_progress=on_progress,
                )
                self.call_from_thread(
                    self.cleanup_log_message,
                    f"  [green]Done → {enhanced_path.name}[/green]",
                )
            except Exception as e:
                self.call_from_thread(
                    self.cleanup_log_message,
                    f"  [red]Failed: {e}[/red]",
                )

        self.call_from_thread(
            self.cleanup_log_message,
            "[bold green]All done![/bold green]",
        )
        self.call_from_thread(self._refresh_cleanup)

    # ── Playback ─────────────────────────────────────────────────

    def action_play(self) -> None:
        if self.generated_audio is None:
            self.gen_log_message("[yellow]No audio to play — generate first[/yellow]")
            return
        sd.stop()
        audio = np.ascontiguousarray(self.generated_audio, dtype=np.float32)
        sd.play(audio, self.generated_sr, blocksize=2048)
        self.gen_log_message("Playing audio...")

    def stop_audio(self) -> None:
        sd.stop()
        self.gen_log_message("Playback stopped")

    def action_save_as(self) -> None:
        if self.generated_audio is None:
            self.gen_log_message("[yellow]No audio to save[/yellow]")
            return
        ts = time.strftime("%Y-%m-%d_%H.%M.%S")
        save_path = Path(f"output/generate-{ts}.wav")
        os.makedirs("output", exist_ok=True)
        sf.write(str(save_path), self.generated_audio, self.generated_sr)
        self.gen_log_message(f"[green]Saved to {save_path}[/green]")


if __name__ == "__main__":
    print("Loading TTS model...")
    model, model_id = load_model_with_fallback()
    print(f"Ready: {model_id}")
    app = VoiceCloneApp(model, model_id)
    app.run()
