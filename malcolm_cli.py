"""Clinician terminal HUD, Bluetooth TWS voice Push-to-Talk, and bedside watcher for Malcolm."""

from __future__ import annotations

import os
import socket
import tempfile
import time
from typing import Any

import numpy as np
import sounddevice as sd
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from scipy.io import wavfile

from local_rag import rag_agent
from state_manager import state_manager
from telemetry_provider import get_telemetry_provider

console = Console()
telemetry_provider = get_telemetry_provider()


def get_lan_ip() -> str:
    """Resolve the local LAN IPv4 address for mobile phone access to Bedside UI."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("10.255.255.255", 1))
            return sock.getsockname()[0]
    except OSError:
        return "127.0.0.1"


def render_dashboard(
    state: dict[str, Any],
    telemetry: dict[str, Any],
    mic_label: str,
    last_engine_status: str,
) -> None:
    """Render the Rich clinical terminal HUD."""
    lan_ip = get_lan_ip()
    bpm = telemetry.get("bpm", state.get("current_bpm", 90))
    wave_state = telemetry.get("wave_state", state.get("wave_state", "Awake / Normal Sinus Rhythm"))
    hw_mode = telemetry.get("mode", "SIMULATED [Arduino AD8232 Stub Ready]")
    audit_count = len(state.get("audit_log") or [])
    stage = state.get("conversation_stage", "idle")
    nurse_dispatched = state.get("nurse_dispatched", False)
    check_in_active = state.get("check_in_active", False)
    symptoms = state.get("patient_symptoms") or []

    grid = Table.grid(expand=True, padding=(0, 2))
    grid.add_column(style="bold cyan", width=22)
    grid.add_column(style="white")

    grid.add_row(
        "Patient Index:",
        f"{state.get('patient_id', '1429')} | {state.get('name', 'Mika Hämäläinen')} "
        f"({state.get('wing', 'Wing A')}, {state.get('room', 'Room 153')} — {state.get('condition', 'Dialysis')})",
    )
    grid.add_row(
        "Live BPM:",
        f"[bold green]{bpm} BPM[/bold green] (89-92 fluctuating) | {wave_state}",
    )
    grid.add_row("Hardware Mode:", f"[yellow]{hw_mode}[/yellow]")
    grid.add_row(
        "PII Redaction Status:",
        "[bold green]ACTIVE[/bold green] ([PATIENT_1429] / [LOCATION_REDACTED])",
    )
    grid.add_row("AI Engine:", f"[magenta]{last_engine_status}[/magenta]")
    grid.add_row(
        "Audit Log Count:",
        f"[bold white]{audit_count}[/bold white] | Stage: [cyan]{stage}[/cyan] | "
        f"Check-In: {'[green]ACTIVE[/green]' if check_in_active else 'OFF'} | "
        f"Nurse Dispatched: {'[bold green]YES[/bold green]' if nurse_dispatched else 'NO'}",
    )
    grid.add_row(
        "Logged Symptoms:",
        ", ".join(symptoms) if symptoms else "[dim]None reported[/dim]",
    )
    grid.add_row("Audio Input:", mic_label)
    grid.add_row(
        "Bedside Web UI:",
        f"http://{lan_ip}:5000  (Local: http://127.0.0.1:5000)",
    )

    panel = Panel(
        grid,
        title="[bold white]MALCOLM — CLINICAL TRIAGE COPILOT HUD[/bold white]",
        subtitle="[dim][Enter] Speak  |  Type command  |  [A] Audit Log  |  [R] Reset Demo  |  [Q] Quit[/dim]",
        border_style="bright_blue",
    )
    console.print(panel)


def select_audio_input_device() -> tuple[int | None, str, bool]:
    """
    List available audio input devices and let the clinician pick a Bluetooth TWS mic
    or choose text-only mode.

    Returns (device_id, display_label, text_only_mode).
    """
    try:
        devices = sd.query_devices()
    except Exception as exc:
        console.print(f"[yellow]Audio subsystem unavailable ({exc}). Using text input mode.[/yellow]")
        return None, "Text-Only Fallback (No Audio Device)", True

    input_devices: list[tuple[int, dict[str, Any]]] = []
    for idx, dev in enumerate(devices):
        if dev.get("max_input_channels", 0) > 0:
            input_devices.append((idx, dev))

    if not input_devices:
        console.print("[yellow]No microphone input devices detected. Using text input mode.[/yellow]")
        return None, "Text-Only Fallback (No Microphone)", True

    table = Table(title="Available Audio Input Devices (Bluetooth TWS / Laptop Mic)", border_style="cyan")
    table.add_column("ID", style="bold yellow", justify="right")
    table.add_column("Device Name", style="white")
    table.add_column("Channels", justify="right")
    table.add_column("Default Sample Rate", justify="right", style="green")

    default_in = sd.default.device[0] if isinstance(sd.default.device, (list, tuple)) else sd.default.device
    for idx, dev in input_devices:
        marker = " *" if idx == default_in else ""
        table.add_row(
            str(idx),
            f"{dev['name']}{marker}",
            str(dev["max_input_channels"]),
            f"{int(dev['default_samplerate'])} Hz",
        )

    console.print(table)
    choice = console.input(
        "[bold cyan]Select input device ID[/bold cyan] ([Enter] for system default, [T] for text-only): "
    ).strip()

    if choice.lower() == "t":
        return None, "Text-Only Mode (Microphone Disabled)", True

    if choice == "":
        try:
            dev_info = sd.query_devices(kind="input")
            dev_idx = default_in if isinstance(default_in, int) and default_in >= 0 else None
            sr = int(dev_info["default_samplerate"])
            return dev_idx, f"System Default ({dev_info['name']} @ {sr} Hz)", False
        except Exception:
            return None, "Text-Only Fallback (Default Mic Unavailable)", True

    try:
        selected_id = int(choice)
        dev_info = sd.query_devices(selected_id, "input")
        sr = int(dev_info["default_samplerate"])
        return selected_id, f"[{selected_id}] {dev_info['name']} @ {sr} Hz", False
    except Exception as exc:
        console.print(f"[yellow]Invalid selection ({exc}); falling back to text-only mode.[/yellow]")
        return None, "Text-Only Fallback", True


class LowSpecWhisperTranscriber:
    """Lazy-loading faster-whisper tiny.en int8 CPU transcriber + dynamic-rate recorder."""

    def __init__(self, device_id: int | None) -> None:
        self.device_id = device_id
        self._model: Any = None

    def _ensure_model(self) -> Any:
        if self._model is None:
            from faster_whisper import WhisperModel

            console.print("[dim]Loading faster-whisper (tiny.en, cpu, int8)...[/dim]")
            self._model = WhisperModel("tiny.en", device="cpu", compute_type="int8")
        return self._model

    def _query_samplerate(self) -> int:
        """Query the selected input device's native sample rate (supports 8kHz/16kHz TWS HFP)."""
        if self.device_id is not None:
            info = sd.query_devices(self.device_id, "input")
        else:
            info = sd.query_devices(kind="input")
        return int(info["default_samplerate"])

    def record_and_transcribe(self) -> str:
        """Record audio between two [Enter] keystrokes and transcribe locally via faster-whisper."""
        samplerate = self._query_samplerate()
        chunks: list[np.ndarray] = []

        def _audio_callback(indata: np.ndarray, _frames: int, _time_info: Any, status: Any) -> None:
            if status:
                pass
            chunks.append(indata.copy())

        console.print(
            f"[bold red]● RECORDING[/bold red] ({samplerate} Hz) — Speak now, then press [bold][Enter][/bold] to stop..."
        )
        with sd.InputStream(
            device=self.device_id,
            samplerate=samplerate,
            channels=1,
            dtype="float32",
            callback=_audio_callback,
        ):
            input()

        if not chunks:
            return ""

        audio_float = np.concatenate(chunks, axis=0).flatten()
        if audio_float.size < int(samplerate * 0.2):
            console.print("[yellow]Recording too short; please try again or type your message.[/yellow]")
            return ""

        pcm_int16 = np.clip(audio_float * 32767.0, -32768.0, 32767.0).astype(np.int16)
        tmp_path = ""
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name
            wavfile.write(tmp_path, samplerate, pcm_int16)

            model = self._ensure_model()
            console.print("[dim]Transcribing locally on CPU (int8)...[/dim]")
            segments, _info = model.transcribe(tmp_path, beam_size=1, vad_filter=True)
            transcript = " ".join(seg.text.strip() for seg in segments).strip()
            return transcript
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass


def print_audit_log(state: dict[str, Any]) -> None:
    """Render the immutable compliance audit_log table."""
    entries = state.get("audit_log") or []
    table = Table(title=f"Immutable Compliance Audit Log ({len(entries)} entries)", border_style="green")
    table.add_column("Timestamp (UTC)", style="dim", width=21)
    table.add_column("Actor", style="cyan")
    table.add_column("Action", style="yellow")
    table.add_column("PII Shield", style="green", justify="center")
    table.add_column("Engine", style="magenta")
    table.add_column("Details", style="white")

    if not entries:
        console.print("[dim]Audit log is currently empty.[/dim]")
        return

    for entry in entries:
        table.add_row(
            str(entry.get("timestamp", "")),
            str(entry.get("actor", "")),
            str(entry.get("action", "")),
            "YES" if entry.get("pii_redacted", True) else "NO",
            str(entry.get("engine", "")),
            str(entry.get("details", "")),
        )
    console.print(table)


def watch_bedside_checkin() -> str | None:
    """
    Poll patient_1429_state.json while check_in_active is True, streaming bedside messages
    in real time until clinician_report_ready flips to True.
    """
    seen_count = len(state_manager.load_state().get("bedside_messages") or [])
    console.print(
        Panel(
            "[cyan]Bedside Web UI session is live.[/cyan] Waiting for patient response...\n"
            "[dim]Press Ctrl+C if you need to return to the clinician prompt early.[/dim]",
            border_style="cyan",
        )
    )

    try:
        while True:
            state = state_manager.load_state()
            messages = state.get("bedside_messages") or []
            if len(messages) > seen_count:
                for msg in messages[seen_count:]:
                    sender = msg.get("sender", "unknown")
                    label = "[bold blue]Patient (Bedside)[/bold blue]" if sender == "patient" else "[bold green]Malcolm (Bedside)[/bold green]"
                    console.print(f"  {label}: {msg.get('text', '')}")
                seen_count = len(messages)

            if state.get("clinician_report_ready"):
                return rag_agent.consume_clinician_report()

            if not state.get("check_in_active") and not state.get("clinician_report_ready"):
                return None

            time.sleep(1.0)
    except KeyboardInterrupt:
        console.print("\n[yellow]Paused bedside watch; returning to clinician prompt.[/yellow]")
        return None


def main() -> None:
    """Run the Malcolm Clinician CLI loop."""
    console.clear()
    device_id, mic_label, text_only = select_audio_input_device()
    transcriber = None if text_only else LowSpecWhisperTranscriber(device_id)
    last_engine_status = rag_agent.get_engine_label()

    while True:
        state = state_manager.load_state()
        telemetry = telemetry_provider.get_reading()
        render_dashboard(state, telemetry, mic_label, last_engine_status)

        # Auto-deliver pending clinician report if bedside finished while idle
        if state.get("clinician_report_ready"):
            report = rag_agent.consume_clinician_report()
            if report:
                console.print(Panel(f"[bold green]Malcolm:[/bold green] {report}", border_style="green"))
                continue

        # If check_in_active is already True, resume watching seamlessly
        if state.get("check_in_active"):
            report = watch_bedside_checkin()
            if report:
                console.print(Panel(f"[bold green]Malcolm:[/bold green] {report}", border_style="green"))
            continue

        prompt_hint = (
            "[bold cyan]Clinician[/bold cyan] (type message, [R]eset, [A]udit, [Q]uit): "
            if text_only
            else "[bold cyan]Clinician[/bold cyan] ([Enter]=Speak, or type message / [R]eset / [A]udit / [Q]uit): "
        )
        try:
            raw_input = console.input(prompt_hint).strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Shutting down Malcolm CLI.[/dim]")
            break

        lowered = raw_input.lower()
        if lowered in {"q", "quit", "exit"}:
            console.print("[dim]Malcolm standing down. Goodbye.[/dim]")
            break

        if lowered == "r":
            state_manager.reset_state()
            last_engine_status = rag_agent.get_engine_label()
            console.print("[bold yellow]Demo state reset to 'idle' for Patient 1429.[/bold yellow]\n")
            continue

        if lowered == "a":
            print_audit_log(state_manager.load_state())
            continue

        if raw_input == "":
            if text_only or transcriber is None:
                console.print("[yellow]Text-only mode is active. Type a clinician message to continue.[/yellow]")
                continue
            try:
                clinician_text = transcriber.record_and_transcribe()
            except Exception as exc:
                console.print(f"[red]Voice capture failed ({exc}). Type your command instead.[/red]")
                continue
            if not clinician_text:
                console.print("[yellow]No speech detected. Try again or type your command.[/yellow]")
                continue
            console.print(f"[bold white]You (STT):[/bold white] {clinician_text}")
        else:
            clinician_text = raw_input

        result = rag_agent.process_clinician_turn(clinician_text)
        last_engine_status = result.get("engine", last_engine_status)

        console.print(
            Panel(
                f"[bold green]Malcolm:[/bold green] {result.get('reply', '')}",
                subtitle=f"[dim]Engine: {last_engine_status} | Stage: {result.get('stage')}[/dim]",
                border_style="green",
            )
        )

        if result.get("check_in_active"):
            if result.get("bedside_greeting"):
                console.print(
                    f"  [bold green]Malcolm (Bedside Initial Push):[/bold green] {result['bedside_greeting']}"
                )
            report = watch_bedside_checkin()
            if report:
                console.print(
                    Panel(
                        f"[bold green]Malcolm:[/bold green] {report}",
                        title="[bold white]BEDSIDE CHECK-IN COMPLETE[/bold white]",
                        border_style="green",
                    )
                )


if __name__ == "__main__":
    main()
