"""Bedside Web UI and Clinician React UI API server for Malcolm."""

from __future__ import annotations

import asyncio
from collections import deque
import ctypes
from datetime import datetime, timedelta
import io
import json
import os
import re
import socket
import sys
import tempfile
import textwrap
import threading
import time
import wave
from typing import Any, Callable

import numpy as np
from dotenv import load_dotenv
from flask import Flask, Response, jsonify, render_template, request, send_file

from local_rag import (
    build_timetable_events,
    build_ward_hud_data,
    rag_agent,
    render_timetable_html_css,
)
from state_manager import state_manager
from telemetry_provider import get_telemetry_provider

try:
    import winsound

    _WINSOUND_AVAILABLE = True
except ImportError:
    winsound = None  # type: ignore[assignment]
    _WINSOUND_AVAILABLE = False

_winsound_enabled: bool = os.getenv("WINSOUND_ENABLED", "true").strip().lower() not in {"0", "false", "no", "off"}
if "--no-winsound" in sys.argv:
    _winsound_enabled = False


def set_winsound_enabled(enabled: bool) -> bool:
    """Enable or disable terminal winsound speaker playback at runtime."""
    global _winsound_enabled
    _winsound_enabled = bool(enabled)
    if not _winsound_enabled and _WINSOUND_AVAILABLE and winsound is not None:
        try:
            winsound.PlaySound(None, 0)
        except Exception:
            pass
    return _winsound_enabled

try:
    import openpyxl
    from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
    from openpyxl.utils import get_column_letter

    _OPENPYXL_AVAILABLE = True
except ImportError:
    openpyxl = None  # type: ignore[assignment]
    _OPENPYXL_AVAILABLE = False

try:
    from PIL import Image, ImageDraw, ImageFont

    _PIL_AVAILABLE = True
except ImportError:
    Image = None  # type: ignore[assignment]
    ImageDraw = None  # type: ignore[assignment]
    ImageFont = None  # type: ignore[assignment]
    _PIL_AVAILABLE = False

load_dotenv()

try:
    import edge_tts

    _EDGE_TTS_AVAILABLE = True
except ImportError:
    edge_tts = None  # type: ignore[assignment]
    _EDGE_TTS_AVAILABLE = False

try:
    import miniaudio

    _MINIAUDIO_AVAILABLE = True
except ImportError:
    miniaudio = None  # type: ignore[assignment]
    _MINIAUDIO_AVAILABLE = False

try:
    import speech_recognition as sr

    _SR_AVAILABLE = True
except ImportError:
    sr = None  # type: ignore[assignment]
    _SR_AVAILABLE = False

try:
    import sherpa_onnx

    _SHERPA_AVAILABLE = True
except ImportError:
    sherpa_onnx = None  # type: ignore[assignment]
    _SHERPA_AVAILABLE = False

app = Flask(__name__)
telemetry_provider = get_telemetry_provider()

TTS_VOICE: str = os.getenv("TTS_VOICE", "en-GB-RyanNeural").strip() or "en-GB-RyanNeural"
_SHERPA_MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "sherpa-zipformer-en")
_WHISPER_MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "whisper-base-en")
_sherpa_recognizer: Any = None
_sherpa_lock = threading.Lock()
_whisper_model: Any = None
_whisper_lock = threading.Lock()
_TTS_CACHE: dict[str, tuple[bytes, str]] = {}
_TTS_CACHE_LOCK = threading.Lock()
_TTS_MAX_CACHE_SIZE = 128


def get_lan_ip() -> str:
    """Resolve the machine's primary local LAN IPv4 address for mobile phone access."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("10.255.255.255", 1))
            return sock.getsockname()[0]
    except OSError:
        return "127.0.0.1"


def _build_bridge_data(state: dict[str, Any]) -> dict[str, Any]:
    """Build the { bpm, transcript, clinicianTranscript, patient, customPane, wardHud, calendarEvents, pageAlert, serverTimeIso } payload."""
    panel = state.get("patient_panel")
    if not isinstance(panel, dict) or not panel:
        panel = rag_agent.synthesize_patient_panel(state)
    custom_pane = state.get("custom_pane")
    if not isinstance(custom_pane, dict) or not custom_pane:
        custom_pane = {"kind": "patient", "title": "", "subtitle": "", "filterWing": "", "generatedHtml": ""}
    now = datetime.now()
    return {
        "bpm": state.get("current_bpm", 90),
        "transcript": state.get("last_transcript", "Listening for clinician request..."),
        "clinicianTranscript": state.get("last_clinician_input", ""),
        "replySeq": state_manager.get_reply_seq(),
        "alertLevel": state.get("alert_level", "normal"),
        "doctorPaged": bool(state.get("doctor_paged", False)),
        "crashCartActive": bool(state.get("crash_cart_active", False)),
        "emergencyMessage": state.get("emergency_message", ""),
        "bedsideMessages": state.get("bedside_messages", []),
        "patient": panel,
        "customPane": custom_pane,
        "wardHud": build_ward_hud_data(),
        "calendarEvents": state_manager.get_calendar_events(),
        "pageAlert": state_manager.get_last_page_alert(),
        "exportRequest": state.get("export_request"),
        "serverTimeIso": now.isoformat(),
        "serverTimeDisplay": now.strftime("%I:%M:%S %p").lstrip("0"),
    }


def _build_unified_state_payload(state: dict[str, Any]) -> dict[str, Any]:
    """Build a deterministic JSON response compatible with both React UI and Bedside Web UI."""
    payload = dict(state)
    payload["state"] = state.get("ui_mode", "idle")
    payload["data"] = _build_bridge_data(state)
    payload["active_patient_id"] = state.get("patient_id", state_manager.get_active_patient_id())
    payload["patients_list"] = state_manager.list_patients()
    payload["ai_engine"] = rag_agent.get_engine_label()
    payload["telemetry_mode"] = telemetry_provider.get_mode_label()
    return payload


def _split_speech_sentences(text: str) -> list[str]:
    """Split a Malcolm reply into crisp sentence chunks for zero-wait pipelined TTS."""
    cleaned = _clean_text_for_speech(text)
    if not cleaned:
        return []
    parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", cleaned) if p.strip()]
    return parts if parts else [cleaned]


def _prewarm_reply_sentences_async(reply_text: str) -> None:
    """Immediately warm TTS cache for each sentence of a reply in a background thread."""
    if not _EDGE_TTS_AVAILABLE or not reply_text:
        return
    sentences = _split_speech_sentences(reply_text)
    if not sentences:
        return

    def _worker() -> None:
        for s in sentences:
            try:
                _synthesize_jarvis_audio(s, voice=TTS_VOICE)
            except Exception:
                break

    threading.Thread(target=_worker, daemon=True).start()


def _run_clinician_message(message: str) -> dict[str, Any]:
    """Handle a clinician utterance or reset command and return a unified bridge payload."""
    _ensure_pager_daemon_running()
    if message.lower() in {"reset", "reset demo", "reset state"}:
        fresh = state_manager.reset_state()
        rag_agent._clinician_history.clear()
        return {
            "ok": True,
            "reply": fresh["last_transcript"],
            "engine": rag_agent.get_engine_label(),
            "stage": fresh["conversation_stage"],
            "check_in_active": False,
            "active_patient_id": fresh["patient_id"],
            "patients_list": state_manager.list_patients(),
            "state": fresh["ui_mode"],
            "data": _build_bridge_data(fresh),
            "raw_state": fresh,
        }

    result = rag_agent.process_clinician_turn(message)
    updated = result.get("state") or state_manager.load_state()
    reply_text = str(result.get("reply", ""))
    rescored_transcript = str(updated.get("last_clinician_input") or message)
    _prewarm_reply_sentences_async(reply_text)
    return {
        "ok": True,
        "transcript": rescored_transcript,
        "reply": reply_text,
        "engine": result.get("engine", rag_agent.get_engine_label()),
        "stage": result.get("stage", updated.get("conversation_stage", "idle")),
        "check_in_active": bool(result.get("check_in_active", updated.get("check_in_active"))),
        "active_patient_id": updated.get("patient_id", state_manager.get_active_patient_id()),
        "patients_list": state_manager.list_patients(),
        "state": updated.get("ui_mode", "idle"),
        "data": _build_bridge_data(updated),
        "raw_state": updated,
    }


def _clean_text_for_speech(text: str) -> str:
    """Normalize punctuation gently while keeping natural conversational prosody intact."""
    cleaned = text.strip()
    if not cleaned:
        return ""
    cleaned = re.sub(r"(\d+)\s*[–—-]\s*(\d+)", r"\1 to \2", cleaned)
    cleaned = cleaned.replace("—", ", ").replace("–", ", ")
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def _store_tts_cache(cache_key: str, audio_bytes: bytes, mimetype: str) -> None:
    """Cache synthesized TTS audio with a bounded dictionary."""
    if not audio_bytes:
        return
    with _TTS_CACHE_LOCK:
        if len(_TTS_CACHE) >= _TTS_MAX_CACHE_SIZE and cache_key not in _TTS_CACHE:
            _TTS_CACHE.pop(next(iter(_TTS_CACHE)))
        _TTS_CACHE[cache_key] = (audio_bytes, mimetype)


def _pcm16_to_wav_bytes(samples: np.ndarray, sample_rate: int) -> bytes:
    """Encode a 1D int16 numpy array into a mono WAV byte buffer."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(samples.astype(np.int16, copy=False).tobytes())
    return buf.getvalue()


def _strip_audio_pauses_wav(mp3_bytes: bytes, sample_rate: int = 24000) -> bytes:
    """
    Decode synthesized MP3 audio into 24kHz mono PCM, trim only leading/trailing dead air,
    and preserve natural conversational pauses & quiet unvoiced consonants.
    """
    if not _MINIAUDIO_AVAILABLE or miniaudio is None:
        return mp3_bytes

    decoded = miniaudio.decode(
        mp3_bytes,
        output_format=miniaudio.SampleFormat.SIGNED16,
        nchannels=1,
        sample_rate=sample_rate,
    )
    samples = np.frombuffer(decoded.samples, dtype=np.int16)
    if samples.size == 0:
        return mp3_bytes

    frame_len = int(sample_rate * 0.02)  # 20ms per frame (480 samples at 24kHz)
    num_frames = (samples.size + frame_len - 1) // frame_len
    padded = np.zeros(num_frames * frame_len, dtype=np.int16)
    padded[: samples.size] = samples
    frames = padded.reshape(num_frames, frame_len)

    rms = np.sqrt(np.mean(frames.astype(np.float32) ** 2, axis=1))
    speech = rms > 35.0
    if not np.any(speech):
        return _pcm16_to_wav_bytes(samples, sample_rate)

    # Expand active mask slightly so soft unvoiced consonant attacks/decays are never clipped
    active = speech.copy()
    for shift in (1, 2):
        active[:-shift] |= speech[shift:]
        active[shift:] |= speech[:-shift]

    indices = np.flatnonzero(active)
    start_idx = max(0, int(indices[0]) - 2)
    end_idx = min(num_frames, int(indices[-1]) + 4)

    kept_indices: list[int] = []
    silent_run = 0
    for idx in range(start_idx, end_idx):
        if active[idx]:
            kept_indices.append(idx)
            silent_run = 0
        else:
            silent_run += 1
            if silent_run <= 8:  # Preserve up to 160ms of crisp Jarvis clause pause
                kept_indices.append(idx)

    trimmed = np.concatenate([frames[i] for i in kept_indices])
    return _pcm16_to_wav_bytes(trimmed, sample_rate)


def _synthesize_jarvis_audio(text: str, voice: str = TTS_VOICE) -> tuple[bytes, str]:
    """Synthesize brisk Jarvis-cadence British neural audio with dead air stripped."""
    if not _EDGE_TTS_AVAILABLE or edge_tts is None:
        raise RuntimeError("edge-tts is not installed")

    cleaned = _clean_text_for_speech(text)
    if not cleaned:
        raise RuntimeError("Text is empty after normalization")

    cache_key = f"v4|{voice}|{cleaned}"
    with _TTS_CACHE_LOCK:
        cached = _TTS_CACHE.get(cache_key)
    if cached:
        return cached

    async def _generate() -> bytes:
        communicate = edge_tts.Communicate(cleaned, voice=voice, rate="+12%", pitch="+0Hz")
        buf = io.BytesIO()
        async for chunk in communicate.stream():
            if chunk.get("type") == "audio":
                buf.write(chunk["data"])
        return buf.getvalue()

    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)
        mp3_bytes = loop.run_until_complete(_generate())
        if not mp3_bytes:
            raise RuntimeError("Empty audio buffer from edge-tts")
    finally:
        loop.close()

    if _MINIAUDIO_AVAILABLE:
        audio_bytes = _strip_audio_pauses_wav(mp3_bytes, sample_rate=24000)
        mimetype = "audio/wav"
    else:
        audio_bytes = mp3_bytes
        mimetype = "audio/mpeg"

    _store_tts_cache(cache_key, audio_bytes, mimetype)
    return audio_bytes, mimetype


def _synthesize_jarvis_mp3(text: str, voice: str = TTS_VOICE) -> bytes:
    """Compatibility wrapper returning synthesized Jarvis TTS audio bytes."""
    audio_bytes, _ = _synthesize_jarvis_audio(text, voice=voice)
    return audio_bytes


def _prewarm_tts_cache() -> None:
    """Warm up Gemini TLS connection pool, Sherpa-ONNX streaming transducer, device list, and core TTS phrases."""
    try:
        rag_agent.warm_connection()
    except Exception:
        pass
    try:
        WindowsWaveInRecorder._ranked_input_devices(_host_recorder._get_winmm())
    except Exception:
        pass
    try:
        _get_sherpa_recognizer()
    except Exception:
        pass

    if not _EDGE_TTS_AVAILABLE:
        return

    for phrase in ("Listening for clinician request...", "At your Service"):
        try:
            _synthesize_jarvis_audio(phrase, voice=TTS_VOICE)
        except Exception:
            break


def _get_sherpa_recognizer() -> Any:
    """Thread-safe lazy loader for Sherpa-ONNX Online Streaming Zipformer Transducer (int8 CPU)."""
    global _sherpa_recognizer
    if not _SHERPA_AVAILABLE or sherpa_onnx is None:
        return None
    with _sherpa_lock:
        if _sherpa_recognizer is None:
            encoder_path = os.path.join(_SHERPA_MODEL_DIR, "encoder-epoch-99-avg-1-chunk-16-left-128.int8.onnx")
            decoder_path = os.path.join(_SHERPA_MODEL_DIR, "decoder-epoch-99-avg-1-chunk-16-left-128.onnx")
            joiner_path = os.path.join(_SHERPA_MODEL_DIR, "joiner-epoch-99-avg-1-chunk-16-left-128.int8.onnx")
            tokens_path = os.path.join(_SHERPA_MODEL_DIR, "tokens.txt")
            if not all(os.path.exists(p) for p in (encoder_path, decoder_path, joiner_path, tokens_path)):
                return None
            _sherpa_recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
                tokens=tokens_path,
                encoder=encoder_path,
                decoder=decoder_path,
                joiner=joiner_path,
                num_threads=4,
                sample_rate=16000,
                feature_dim=80,
                enable_endpoint_detection=True,
                rule1_min_trailing_silence=2.2,
                rule2_min_trailing_silence=0.85,
                rule3_min_utterance_length=25.0,
                decoding_method="modified_beam_search",
                max_active_paths=4,
            )
        return _sherpa_recognizer


def _get_whisper_model() -> Any:
    """Thread-safe lazy loader for local faster-whisper base.en/tiny.en int8 CPU model."""
    global _whisper_model
    with _whisper_lock:
        if _whisper_model is None:
            from faster_whisper import WhisperModel

            model_id = _WHISPER_MODEL_DIR if os.path.exists(_WHISPER_MODEL_DIR) else "tiny.en"
            _whisper_model = WhisperModel(model_id, device="cpu", compute_type="int8", cpu_threads=4)
        return _whisper_model


class _WAVEINCAPSW(ctypes.Structure):
    _fields_ = [
        ("wMid", ctypes.c_ushort),
        ("wPid", ctypes.c_ushort),
        ("vDriverVersion", ctypes.c_uint32),
        ("szPname", ctypes.c_wchar * 32),
        ("dwFormats", ctypes.c_uint32),
        ("wChannels", ctypes.c_ushort),
        ("wReserved1", ctypes.c_ushort),
    ]


class _WAVEFORMATEX(ctypes.Structure):
    _fields_ = [
        ("wFormatTag", ctypes.c_ushort),
        ("nChannels", ctypes.c_ushort),
        ("nSamplesPerSec", ctypes.c_uint32),
        ("nAvgBytesPerSec", ctypes.c_uint32),
        ("nBlockAlign", ctypes.c_ushort),
        ("wBitsPerSample", ctypes.c_ushort),
        ("cbSize", ctypes.c_ushort),
    ]


class _WAVEHDR(ctypes.Structure):
    _fields_ = [
        ("lpData", ctypes.c_void_p),
        ("dwBufferLength", ctypes.c_uint32),
        ("dwBytesRecorded", ctypes.c_uint32),
        ("dwUser", ctypes.c_size_t),
        ("dwFlags", ctypes.c_uint32),
        ("dwLoops", ctypes.c_uint32),
        ("lpNext", ctypes.c_void_p),
        ("reserved", ctypes.c_size_t),
    ]


class WindowsWaveInRecorder:
    """Simultaneous multi-device Windows winmm.dll recorder with parallel clause-chunked STT."""

    _WHDR_DONE = 0x00000001
    _BUFFER_BYTES = 3200  # 100ms per buffer at 16kHz mono 16-bit PCM (zero dropout)
    _NUM_BUFFERS = 16
    _winmm_configured = False
    _winmm_lock = threading.Lock()

    def __init__(self, sample_rate: int = 16000) -> None:
        self.sample_rate = sample_rate
        self._lock = threading.Lock()
        self._chunks_lock = threading.Lock()
        self._recording = False
        self._stop_event = threading.Event()
        self._threads: list[threading.Thread] = []
        self._transcriber_thread: threading.Thread | None = None
        self._device_chunks: dict[str, list[bytes]] = {}
        self._device_peak_rms: dict[str, float] = {}
        self._opened_devices: list[tuple[str, int]] = []
        self._device_name = ""
        self._record_start_time = 0.0
        self._last_speech_time = 0.0
        self.current_rms = 0.0
        self.max_rms = 0.0
        self.live_transcript = ""
        self.final_payload: dict[str, Any] | None = None
        self.speech_detected = False
        self.auto_stopped = False
        self._status_error: str | None = None
        self._on_final_callback: Callable[[str], dict[str, Any]] | None = None
        self._clause_byte_offset = 0
        self._clause_slots: list[list[str]] = []
        self._clause_threads: list[threading.Thread] = []

    @classmethod
    def _get_winmm(cls) -> Any:
        winmm = ctypes.windll.winmm
        with cls._winmm_lock:
            if not cls._winmm_configured:
                winmm.waveInOpen.argtypes = [
                    ctypes.POINTER(ctypes.c_void_p),
                    ctypes.c_uint32,
                    ctypes.POINTER(_WAVEFORMATEX),
                    ctypes.c_size_t,
                    ctypes.c_size_t,
                    ctypes.c_uint32,
                ]
                winmm.waveInOpen.restype = ctypes.c_uint32
                for fn in (winmm.waveInPrepareHeader, winmm.waveInUnprepareHeader, winmm.waveInAddBuffer):
                    fn.argtypes = [ctypes.c_void_p, ctypes.POINTER(_WAVEHDR), ctypes.c_uint32]
                    fn.restype = ctypes.c_uint32
                for fn in (winmm.waveInStart, winmm.waveInStop, winmm.waveInReset, winmm.waveInClose):
                    fn.argtypes = [ctypes.c_void_p]
                    fn.restype = ctypes.c_uint32
                cls._winmm_configured = True
        return winmm

    @staticmethod
    def _score_device_name(name: str) -> int:
        lower = name.lower()
        if any(k in lower for k in ("camo", "virtual", "vb-audio", "obs", "droidcam", "iriun")):
            return -200
        score = 50
        if any(k in lower for k in ("headset", "buds", "airpods", "bluetooth", "usb")):
            score += 150
        if any(k in lower for k in ("microphone array", "realtek", "amd", "intel")):
            score += 100
        return score

    _cached_devices: list[tuple[int, str, int]] | None = None

    @classmethod
    def _ranked_input_devices(cls, winmm: Any) -> list[tuple[int, str, int]]:
        if cls._cached_devices is not None:
            return list(cls._cached_devices)
        count = int(winmm.waveInGetNumDevs())
        devices: list[tuple[int, str, int]] = []
        for idx in range(count):
            caps = _WAVEINCAPSW()
            if winmm.waveInGetDevCapsW(idx, ctypes.byref(caps), ctypes.sizeof(caps)) == 0:
                name = caps.szPname.strip() or f"Input {idx}"
                score = cls._score_device_name(name)
                if score >= 0:
                    devices.append((idx, name, score))
        devices.sort(key=lambda item: (-item[2], item[0]))
        if devices:
            cls._cached_devices = list(devices)
        return devices

    def _capture_device_loop(
        self,
        dev_id: int,
        dev_name: str,
        score: int,
        open_done: threading.Event,
        open_results: list[tuple[bool, str, int]],
    ) -> None:
        winmm = self._get_winmm()
        fmt = _WAVEFORMATEX(
            wFormatTag=1,
            nChannels=1,
            nSamplesPerSec=self.sample_rate,
            nAvgBytesPerSec=self.sample_rate * 2,
            nBlockAlign=2,
            wBitsPerSample=16,
            cbSize=0,
        )

        h_wave_in = ctypes.c_void_p()
        res = winmm.waveInOpen(ctypes.byref(h_wave_in), dev_id, ctypes.byref(fmt), 0, 0, 0)
        if res != 0 or not h_wave_in.value:
            with self._chunks_lock:
                open_results.append((False, f"err {int(res)}", score))
            open_done.set()
            return

        buffers: list[ctypes.Array[ctypes.c_char]] = []
        headers: list[_WAVEHDR] = []
        hdr_size = ctypes.sizeof(_WAVEHDR)
        try:
            for _ in range(self._NUM_BUFFERS):
                buf = ctypes.create_string_buffer(self._BUFFER_BYTES)
                hdr = _WAVEHDR(
                    lpData=ctypes.cast(buf, ctypes.c_void_p).value,
                    dwBufferLength=self._BUFFER_BYTES,
                )
                winmm.waveInPrepareHeader(h_wave_in, ctypes.byref(hdr), hdr_size)
                winmm.waveInAddBuffer(h_wave_in, ctypes.byref(hdr), hdr_size)
                buffers.append(buf)
                headers.append(hdr)

            start_res = winmm.waveInStart(h_wave_in)
            if start_res != 0:
                with self._chunks_lock:
                    open_results.append((False, f"start err {int(start_res)}", score))
                open_done.set()
                return

            with self._chunks_lock:
                self._opened_devices.append((dev_name, score))
                self._device_chunks[dev_name] = []
                self._device_peak_rms[dev_name] = 0.0
                open_results.append((True, dev_name, score))
            open_done.set()

            noise_rms: float | None = None
            while not self._stop_event.is_set():
                for idx, hdr in enumerate(headers):
                    if hdr.dwFlags & self._WHDR_DONE:
                        if hdr.dwBytesRecorded > 0:
                            chunk = ctypes.string_at(hdr.lpData, hdr.dwBytesRecorded)
                            samples = np.frombuffer(chunk, dtype=np.int16).astype(np.float32)
                            if samples.size:
                                centered = samples - float(np.mean(samples))
                                rms = float(np.sqrt(np.mean(centered * centered)))
                            else:
                                rms = 0.0
                            with self._chunks_lock:
                                self._device_chunks[dev_name].append(chunk)
                                self.current_rms = rms
                                if rms > self.max_rms:
                                    self.max_rms = rms
                                dev_peak = self._device_peak_rms.get(dev_name, 0.0)
                                if rms > dev_peak:
                                    self._device_peak_rms[dev_name] = rms
                                    dev_peak = rms
                            if rms >= 1.0:
                                if noise_rms is None:
                                    noise_rms = min(rms, 120.0)
                                elif rms < noise_rms:
                                    noise_rms = 0.65 * noise_rms + 0.35 * rms
                                elif rms < noise_rms * 1.6:
                                    noise_rms = 0.92 * noise_rms + 0.08 * rms
                            threshold = max(22.0, float(noise_rms or 12.0) * 2.1, dev_peak * 0.28)
                            now = time.time()
                            if rms > threshold:
                                self.speech_detected = True
                                self._last_speech_time = now
                        winmm.waveInUnprepareHeader(h_wave_in, ctypes.byref(hdr), hdr_size)
                        hdr.dwFlags = 0
                        hdr.dwBytesRecorded = 0
                        hdr.lpData = ctypes.cast(buffers[idx], ctypes.c_void_p).value
                        winmm.waveInPrepareHeader(h_wave_in, ctypes.byref(hdr), hdr_size)
                        winmm.waveInAddBuffer(h_wave_in, ctypes.byref(hdr), hdr_size)
                time.sleep(0.005)

            winmm.waveInStop(h_wave_in)
            winmm.waveInReset(h_wave_in)
            for hdr in headers:
                if hdr.dwBytesRecorded > 0:
                    chunk = ctypes.string_at(hdr.lpData, hdr.dwBytesRecorded)
                    with self._chunks_lock:
                        self._device_chunks[dev_name].append(chunk)
                winmm.waveInUnprepareHeader(h_wave_in, ctypes.byref(hdr), hdr_size)
        finally:
            winmm.waveInClose(h_wave_in)

    def _get_best_pcm(self) -> tuple[bytes, str]:
        """Return PCM bytes from whichever opened hardware mic captured the strongest signal."""
        with self._chunks_lock:
            if not self._opened_devices:
                return b"", ""
            ranked = sorted(
                self._opened_devices,
                key=lambda item: (
                    round(self._device_peak_rms.get(item[0], 0.0), 3),
                    item[1],
                ),
                reverse=True,
            )
            selected_chunks: list[bytes] = []
            selected_dev = ranked[0][0]
            for dev_name, _score in ranked:
                chunks = self._device_chunks.get(dev_name)
                if chunks:
                    selected_chunks = list(chunks)
                    selected_dev = dev_name
                    break
        return (b"".join(selected_chunks) if selected_chunks else b""), selected_dev

    def _get_best_device_new_chunks(self, dev_chunk_idx: dict[str, int]) -> tuple[bytes, str, float]:
        """Return newly captured PCM chunks for the strongest active mic."""
        with self._chunks_lock:
            if not self._opened_devices:
                return b"", "", 0.0
            ranked = sorted(
                self._opened_devices,
                key=lambda item: (
                    round(self._device_peak_rms.get(item[0], 0.0), 3),
                    item[1],
                ),
                reverse=True,
            )
            best_dev = ranked[0][0]
            new_slice: list[bytes] = []
            for dev_name, _score in ranked:
                chunks = self._device_chunks.get(dev_name)
                if chunks:
                    best_dev = dev_name
                    prev_idx = dev_chunk_idx.get(dev_name, 0)
                    if len(chunks) > prev_idx:
                        new_slice = chunks[prev_idx:]
                        dev_chunk_idx[dev_name] = len(chunks)
                    break
            dev_peak = self._device_peak_rms.get(best_dev, 0.0)
        return (b"".join(new_slice) if new_slice else b""), best_dev, dev_peak

    def _live_transcribe_loop(self) -> None:
        """True frame-by-frame ~60ms streaming ASR via Sherpa-ONNX + Clause-Chunked Parallel Google STT."""
        sherpa_rec = _get_sherpa_recognizer()
        dev_streams: dict[str, Any] = {}
        dev_chunk_idx: dict[str, int] = {}
        stable_ticks = 0
        last_speech_transcribed = ""
        last_state_sync = 0.0
        should_finalize = False

        spec_speech_ts = 0.0
        spec_thread: threading.Thread | None = None
        spec_box: list[str] = []

        while not self._stop_event.is_set():
            if self._stop_event.wait(0.06):
                break
            now = time.time()
            silence_elapsed = (now - self._last_speech_time) if self._last_speech_time > 0 else 0.0

            if sherpa_rec is not None:
                new_bytes, best_dev, dev_peak = self._get_best_device_new_chunks(dev_chunk_idx)

                best_partial = ""
                endpoint_hit = False
                if best_dev:
                    if new_bytes:
                        if best_dev not in dev_streams:
                            dev_streams[best_dev] = sherpa_rec.create_stream()
                        stream = dev_streams[best_dev]
                        samples_f32 = np.frombuffer(new_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                        if dev_peak > 20.0:
                            gain = min(6.0, max(1.5, 12000.0 / max(dev_peak, 1500.0)))
                            samples_f32 = np.clip(samples_f32 * gain, -0.95, 0.95)

                        stream.accept_waveform(self.sample_rate, samples_f32)
                        while sherpa_rec.is_ready(stream):
                            sherpa_rec.decode_stream(stream)

                    stream = dev_streams.get(best_dev)
                    if stream is not None:
                        raw_res = sherpa_rec.get_result(stream).strip()
                        if raw_res:
                            best_partial = _normalize_clinical_stt(raw_res)
                            if sherpa_rec.is_endpoint(stream):
                                endpoint_hit = True

                if best_partial and best_partial != self.live_transcript:
                    self.live_transcript = best_partial
                    stable_ticks = 0
                    last_speech_transcribed = best_partial
                    self.speech_detected = True
                    self._last_speech_time = now
                    silence_elapsed = 0.0
                    if now - last_state_sync >= 0.18:
                        last_state_sync = now
                        state_manager.update_state({"last_clinician_input": best_partial})
                elif best_partial and best_partial == last_speech_transcribed:
                    stable_ticks += 1

                # Clause-chunked background Google STT on micro-pauses (>=0.28s) during long speech (>=3.2s clause)
                if (
                    self.speech_detected
                    and silence_elapsed >= 0.28
                    and self._last_speech_time > 0
                    and spec_speech_ts != self._last_speech_time
                ):
                    spec_speech_ts = self._last_speech_time
                    full_pcm, _ = self._get_best_pcm()
                    uncommitted_len = len(full_pcm) - self._clause_byte_offset
                    if uncommitted_len >= int(self.sample_rate * 2 * 3.2):
                        clause_pcm = full_pcm[self._clause_byte_offset :]
                        self._clause_byte_offset = len(full_pcm)
                        slot: list[str] = []
                        self._clause_slots.append(slot)

                        def _clause_worker(audio_bytes: bytes, sr_hz: int, out_slot: list[str]) -> None:
                            res = _transcribe_google_pcm(audio_bytes, sr_hz, timeout_sec=1.9)
                            if res:
                                out_slot.append(res)

                        c_th = threading.Thread(
                            target=_clause_worker,
                            args=(clause_pcm, self.sample_rate, slot),
                            daemon=True,
                        )
                        self._clause_threads.append(c_th)
                        c_th.start()
                    elif self._clause_byte_offset == 0 and len(full_pcm) >= int(self.sample_rate * 2 * 0.30):
                        spec_box = []
                        box_ref = spec_box

                        def _spec_worker(audio_bytes: bytes, sr_hz: int, out_box: list[str]) -> None:
                            res = _transcribe_google_pcm(audio_bytes, sr_hz, timeout_sec=1.55)
                            if res:
                                out_box.append(res)

                        spec_thread = threading.Thread(
                            target=_spec_worker,
                            args=(full_pcm, self.sample_rate, box_ref),
                            daemon=True,
                        )
                        spec_thread.start()

                if (
                    self.speech_detected
                    and bool(self.live_transcript.strip())
                    and (now - self._record_start_time >= 0.85)
                    and (
                        (endpoint_hit and silence_elapsed > 0.85)
                        or silence_elapsed > 1.05
                        or (stable_ticks >= 12 and silence_elapsed > 0.80)
                    )
                ):
                    should_finalize = True
                    self._stop_event.set()
                    break
            else:
                if (
                    self.speech_detected
                    and (now - self._record_start_time >= 0.85)
                    and silence_elapsed > 1.0
                ):
                    should_finalize = True
                    self._stop_event.set()
                    break
                pcm, _best_dev = self._get_best_pcm()
                if len(pcm) < int(self.sample_rate * 2 * 0.30):
                    continue
                partial = _transcribe_pcm_bytes(pcm, self.sample_rate)
                if partial and partial != self.live_transcript:
                    self.live_transcript = partial
                    self.speech_detected = True
                    self._last_speech_time = now

        if should_finalize:
            for th in list(self._threads):
                if th.is_alive():
                    th.join(timeout=0.8)
            pcm, _best_dev = self._get_best_pcm()
            try:
                sherpa_final = self.live_transcript.strip()
                if sherpa_rec is not None and dev_streams:
                    tail = np.zeros(int(0.32 * self.sample_rate), dtype=np.float32)
                    for stream in dev_streams.values():
                        try:
                            stream.accept_waveform(self.sample_rate, tail)
                            stream.input_finished()
                            while sherpa_rec.is_ready(stream):
                                sherpa_rec.decode_stream(stream)
                            flushed = _normalize_clinical_stt(sherpa_rec.get_result(stream).strip())
                            if len(flushed) >= len(sherpa_final):
                                sherpa_final = flushed
                        except Exception:
                            pass

                google_final = self._assemble_google_clauses(pcm, spec_thread, spec_speech_ts, spec_box)
                final_text = _select_best_final_transcript(google_final, sherpa_final, pcm, self.sample_rate)

                if final_text:
                    self.live_transcript = final_text
                    if self._on_final_callback is not None:
                        payload = self._on_final_callback(final_text)
                        rescored_display = str(payload.get("transcript") or final_text)
                        payload["transcript"] = rescored_display
                        self.live_transcript = rescored_display
                        self.final_payload = payload
                else:
                    self._status_error = "No speech detected in audio."
            except Exception as exc:
                self._status_error = f"STT failed: {exc}"
            finally:
                self.auto_stopped = True
                self._recording = False

    def _assemble_google_clauses(
        self,
        pcm: bytes,
        spec_thread: threading.Thread | None = None,
        spec_speech_ts: float = 0.0,
        spec_box: list[str] | None = None,
    ) -> str:
        """Combine background-transcribed Google STT clauses with the final tail clause."""
        if self._clause_byte_offset > 0:
            tail_pcm = pcm[self._clause_byte_offset :]
            tail_box: list[str] = []
            tail_th: threading.Thread | None = None
            if len(tail_pcm) >= int(self.sample_rate * 2 * 0.35):
                def _tail_worker() -> None:
                    res = _transcribe_google_pcm(tail_pcm, self.sample_rate, timeout_sec=1.6)
                    if res:
                        tail_box.append(res)

                tail_th = threading.Thread(target=_tail_worker, daemon=True)
                tail_th.start()
            for c_th in list(self._clause_threads):
                if c_th.is_alive():
                    c_th.join(timeout=0.9)
            if tail_th is not None:
                tail_th.join(timeout=1.5)
            assembled = [s[0] for s in self._clause_slots if s]
            if tail_box:
                assembled.append(tail_box[0])
            if assembled:
                return _normalize_clinical_stt(" ".join(assembled))

        if spec_thread is not None and spec_speech_ts == self._last_speech_time and spec_box is not None:
            spec_thread.join(timeout=0.85)
            if spec_box:
                return spec_box[0]
        if pcm:
            return _transcribe_google_pcm(pcm, self.sample_rate, timeout_sec=1.8)
        return ""

    def start(self, on_final: Callable[[str], dict[str, Any]] | None = None) -> str:
        """Open all Windows hardware microphones in parallel (~50ms) and start real-time streaming STT."""
        self.stop_and_discard()
        winmm = self._get_winmm()
        candidates = self._ranked_input_devices(winmm)
        if not candidates:
            candidates = [(0xFFFFFFFF, "Windows Default Mapper", 0)]

        with self._lock:
            with self._chunks_lock:
                self._device_chunks = {}
                self._device_peak_rms = {}
                self._opened_devices = []
            self._device_name = ""
            self._status_error = None
            self.current_rms = 0.0
            self.max_rms = 0.0
            self.live_transcript = ""
            self.final_payload = None
            self.speech_detected = False
            self.auto_stopped = False
            self._last_speech_time = 0.0
            self._record_start_time = time.time()
            self._on_final_callback = on_final
            self._clause_byte_offset = 0
            self._clause_slots = []
            self._clause_threads = []
            self._stop_event.clear()
            self._recording = True
            self._threads = []

            open_results: list[tuple[bool, str, int]] = []
            events: list[threading.Event] = []
            for dev_id, dev_name, score in candidates:
                ev = threading.Event()
                events.append(ev)
                th = threading.Thread(
                    target=self._capture_device_loop,
                    args=(dev_id, dev_name, score, ev, open_results),
                    daemon=True,
                )
                self._threads.append(th)
                th.start()

        deadline = time.time() + 2.0
        for ev in events:
            remaining = max(0.0, deadline - time.time())
            ev.wait(timeout=remaining)

        with self._chunks_lock:
            self._opened_devices.sort(key=lambda item: -item[1])
            opened_names = [name for name, _score in self._opened_devices]

        if not opened_names and not any(cid == 0xFFFFFFFF for cid, _, _ in candidates):
            ev_map = threading.Event()
            with self._lock:
                th_map = threading.Thread(
                    target=self._capture_device_loop,
                    args=(0xFFFFFFFF, "Windows Default Mapper", 0, ev_map, open_results),
                    daemon=True,
                )
                self._threads.append(th_map)
                th_map.start()
            ev_map.wait(timeout=1.5)
            with self._chunks_lock:
                opened_names = [name for name, _score in self._opened_devices]

        if not opened_names:
            self.stop_and_discard()
            errs = ", ".join(msg for ok, msg, _ in open_results if not ok) or "no capture devices"
            raise RuntimeError(f"No Windows microphone available ({errs}).")

        self._device_name = " + ".join(opened_names)
        with self._lock:
            self._transcriber_thread = threading.Thread(target=self._live_transcribe_loop, daemon=True)
            self._transcriber_thread.start()
        return self._device_name

    def get_live_status(self) -> dict[str, Any]:
        """Return snapshot of active host mics, live RMS level, partial STT transcript, and final payload."""
        with self._chunks_lock:
            devices = [name for name, _ in self._opened_devices]
        return {
            "ok": True,
            "recording": self._recording,
            "auto_stopped": self.auto_stopped,
            "speech_detected": self.speech_detected,
            "device": self._device_name,
            "devices": devices,
            "rms": round(self.current_rms, 1),
            "max_rms": round(self.max_rms, 1),
            "live_transcript": self.live_transcript,
            "final_payload": self.final_payload,
            "error": self._status_error,
        }

    def stop_and_discard(self) -> None:
        """Stop all active microphone capture threads cleanly without keeping audio bytes."""
        with self._lock:
            threads = list(self._threads)
            t_thread = self._transcriber_thread
            self.auto_stopped = False
            self._stop_event.set()
            self._recording = False
            self._threads = []
            self._transcriber_thread = None
        for th in threads:
            if th.is_alive():
                th.join(timeout=1.5)
        if t_thread and t_thread.is_alive():
            t_thread.join(timeout=1.5)

    def stop_to_wav_bytes(self) -> bytes:
        """Stop all microphone captures manually and encode strongest hardware channel to 16kHz WAV."""
        with self._lock:
            threads = list(self._threads)
            t_thread = self._transcriber_thread
            self.auto_stopped = False
            self._stop_event.set()
            self._recording = False
            self._threads = []
            self._transcriber_thread = None
        for th in threads:
            if th.is_alive():
                th.join(timeout=2.0)
        if t_thread and t_thread.is_alive():
            t_thread.join(timeout=2.0)

        pcm, _best_dev = self._get_best_pcm()
        with self._chunks_lock:
            self._device_chunks = {}
            self._device_peak_rms = {}
        if not pcm:
            return b""
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.sample_rate)
            wf.writeframes(pcm)
        return buf.getvalue()


_host_recorder = WindowsWaveInRecorder(sample_rate=16000)


def _normalize_audio_f32(audio_f32: np.ndarray) -> tuple[np.ndarray, float]:
    """Center DC offset, apply clean silence gate, and apply gentle bounded AGC (max 6x gain)."""
    if audio_f32.size == 0:
        return np.zeros(0, dtype=np.float32), 0.0
    centered = audio_f32 - float(np.mean(audio_f32))
    peak = float(np.max(np.abs(centered)))
    raw_rms = float(np.sqrt(np.mean(centered * centered)))
    if peak < 0.006 or raw_rms < 0.002:
        return np.zeros(0, dtype=np.float32), 0.0
    if peak < 0.65:
        gain = min(6.0, 0.70 / peak)
        centered = np.clip(centered * gain, -1.0, 1.0)
    return centered.astype(np.float32), raw_rms


_WHISPER_HALLUCINATIONS = {
    "",
    "thank you",
    "thanks for watching",
    "you",
    "bye",
    "subtitle",
    "subtitles",
    "...",
}


def _is_whisper_hallucination(text: str, raw_rms: float) -> bool:
    """Filter common Whisper artifacts emitted on very quiet ambient noise."""
    cleaned = text.strip()
    if not cleaned or not re.search(r"[a-zA-Z0-9]", cleaned):
        return True
    normalized = re.sub(r"[.!?,]+$", "", cleaned).strip().lower()
    if raw_rms < 0.005 and normalized in _WHISPER_HALLUCINATIONS:
        return True
    return False


_PROPER_NOUN_REPLACEMENTS: tuple[tuple[str, str], ...] = (
    (r"\bmalcolm\b", "Malcolm"),
    (r"\bmika\b", "Mika"),
    (r"\bhamalainen\b", "Hämäläinen"),
    (r"\baino\b", "Aino"),
    (r"\bvirtanen\b", "Virtanen"),
    (r"\bjukka\b", "Jukka"),
    (r"\bkorhonen\b", "Korhonen"),
    (r"\bsanna\b", "Sanna"),
    (r"\blaine\b", "Laine"),
    (r"\bbpm\b", "BPM"),
    (r"\bi\b", "I"),
    (r"\bi'm\b", "I'm"),
    (r"\bi'd\b", "I'd"),
    (r"\bi'll\b", "I'll"),
)


def _normalize_clinical_stt(text: str) -> str:
    """Format streaming transducer text, convert spoken clinical numbers, and fix medical ASR homophones."""
    if not text:
        return ""
    out = re.sub(r"\s+", " ", text.strip())
    if not out:
        return ""
    if out.isupper():
        out = out.lower()
        out = out[0].upper() + out[1:]
        for pat, repl in _PROPER_NOUN_REPLACEMENTS:
            out = re.sub(pat, repl, out, flags=re.IGNORECASE)

    # Spoken patient IDs & room numbers -> digits
    spoken_nums = (
        (r"\bfourteen\s+twenty[\s-]*nine\b", "1429"),
        (r"\bone\s+four\s+two\s+nine\b", "1429"),
        (r"\bfourteen\s+thirty[\s-]*one\b", "1431"),
        (r"\bone\s+four\s+three\s+one\b", "1431"),
        (r"\bfourteen\s+thirty[\s-]*two\b", "1432"),
        (r"\bone\s+four\s+three\s+two\b", "1432"),
        (r"\bfourteen\s+thirty\b", "1430"),
        (r"\bone\s+four\s+three\s+(zero|oh)\b", "1430"),
        (r"\bone\s+fifty[\s-]*three\b", "153"),
        (r"\btwo\s+(oh|zero)\s+four\b", "204"),
        (r"\bthree\s+twelve\b", "312"),
        (r"\bone\s+eighteen\b", "118"),
        (r"\b14\s+29\b", "1429"),
        (r"\b14\s+30\b", "1430"),
        (r"\b14\s+31\b", "1431"),
        (r"\b14\s+32\b", "1432"),
    )
    for pat, repl in spoken_nums:
        out = re.sub(pat, repl, out, flags=re.IGNORECASE)

    # Medical & hospital ward ASR homophone & phonetic entity repairs
    homophones = (
        (r"\b(all|our|list|every|four|the)\s+patience\b", r"\1 patients"),
        (r"\blist\s+our\s+patients\b", "list all patients"),
        (r"\bheart\s+weight\b", "heart rate"),
        (r"\bsurgo\s+(?:had|ahead)\b", "Sure, go ahead"),
        (r"\bsure\s+go\s+head\b", "Sure, go ahead"),
        (r"\bgained\s+the\s+people\b", "tell me more about the people"),
        (r"\b(?:into\s+his\s+aboard|in\s+his\s+aboard|into\s+the\s+board|in\s+the\s+board|on\s+the\s+board|in\s+award|on\s+award)\b", "in the ward"),
        (r"\b(people|patients|everyone)\s+(?:in|on)\s+the\s+(?:war|world|word|board)\b", r"\1 in the ward"),
        (r"\bi'll\s+dripping\s+a\s+coffee\s+was\b", "whilst I get my coffee ready"),
        # Finnish patient name phonetic biasing (Patient 1429-1432)
        (r"\b(?:aham\s+linen|a\s+ham\s+linen|hama\s+linen|ham\s+a\s+line\s+in|hamalainen|mika\s+hamalainen)\b", "Mika Hämäläinen"),
        (r"\b(?:yuka|you\s+ka)\s*(?:korhonen)?\b", "Jukka Korhonen"),
        (r"\b(?:i\s+know\s+virtanen|aino\s+virtanen)\b", "Aino Virtanen"),
        (r"\b(?:sanna\s+line|sana\s+lane|sana\s+laine)\b", "Sanna Laine"),
        # Clinical & hospital wing homophones
        (r"\bwing\s+(?:see|sea)\b", "Wing C"),
        (r"\bwing\s+(?:ay|eh)\b", "Wing A"),
        (r"\bwing\s+bee\b", "Wing B"),
        (r"^cardia\s+in\s+wing\b", "Tachycardia in Wing"),
        (r"\bon\s+particularly\s+interested\b", "I'm particularly interested"),
        (r"\band\s+a\s+after\s+you\s+told\b", "and after you've told"),
        (r"\b(?:said\s+you'll|shed\s+you'll|said\s+you\s+will)\s+in\b", "scheduled in"),
        (r"\b(?:dot\s+)?x\s*l\s*s\s*x\b", ".xlsx"),
        (r"\b(?:dot\s+)?p\s*n\s*g\b", ".png"),
        (r"\bp\s+and\s+g\s+file\b", ".png file"),
    )
    for pat, repl in homophones:
        out = re.sub(pat, repl, out, flags=re.IGNORECASE)

    # Strip spoken acoustic hesitation fillers ('er', 'uh', 'erm', 'um') and stutters
    out = re.sub(r"\b(?:er|uh|erm|um)\b\s*,?", "", out, flags=re.IGNORECASE)
    out = re.sub(r"\b(I\s+will)\s+\1\b", r"\1", out, flags=re.IGNORECASE)
    out = re.sub(r"\s+", " ", out).strip()
    if out:
        out = out[0].upper() + out[1:]
    return out


def _trim_speech_pcm(pcm_bytes: bytes, sample_rate: int = 16000) -> bytes:
    """
    Trim leading/trailing dead air around active speech and apply only gentle un-clipped
    linear scaling when quiet, preserving natural dynamics for Google Speech Recognition.
    """
    if not pcm_bytes:
        return b""
    samples = np.frombuffer(pcm_bytes, dtype=np.int16)
    if samples.size < int(sample_rate * 0.20):
        return b""
    centered = samples.astype(np.float32) - float(np.mean(samples))
    peak_raw = float(np.max(np.abs(centered)))
    if peak_raw < 25.0:
        return b""

    frame_len = max(1, int(sample_rate * 0.02))  # 20ms frames
    num_frames = centered.size // frame_len
    if num_frames >= 5:
        framed = centered[: num_frames * frame_len].reshape(num_frames, frame_len)
        frame_rms = np.sqrt(np.mean(framed * framed, axis=1))
        noise_floor = float(np.percentile(frame_rms, 15))
        speech_thresh = max(12.0, noise_floor * 1.65, peak_raw * 0.04)
        active_idx = np.flatnonzero(frame_rms > speech_thresh)
        if active_idx.size > 0:
            start_f = max(0, int(active_idx[0]) - 12)  # 240ms pre-speech pad
            end_f = min(num_frames, int(active_idx[-1]) + 15)  # 300ms post-speech pad
            centered = centered[start_f * frame_len : end_f * frame_len]

    peak_norm = float(np.max(np.abs(centered))) / 32768.0
    if peak_norm < 0.001:
        return b""
    if peak_norm < 0.45:
        gain = min(6.0, 0.70 / peak_norm)
        centered = centered * gain
    return np.clip(centered, -32767.0, 32767.0).astype(np.int16).tobytes()


def _transcribe_sherpa_f32(audio_f32: np.ndarray, sample_rate: int = 16000) -> str:
    """Decode float32 mono PCM using Sherpa-ONNX Streaming Zipformer Transducer."""
    sherpa_rec = _get_sherpa_recognizer()
    if sherpa_rec is None or audio_f32.size == 0:
        return ""
    centered = audio_f32 - float(np.mean(audio_f32))
    peak = float(np.max(np.abs(centered)))
    if peak < 0.005:
        return ""
    if peak < 0.25:
        centered = np.clip(centered * min(3.0, 0.60 / peak), -0.98, 0.98)
    stream = sherpa_rec.create_stream()
    chunk_len = max(1, int(sample_rate * 0.16))
    for idx in range(0, centered.size, chunk_len):
        stream.accept_waveform(sample_rate, centered[idx : idx + chunk_len])
        while sherpa_rec.is_ready(stream):
            sherpa_rec.decode_stream(stream)
    tail = np.zeros(int(sample_rate * 0.35), dtype=np.float32)
    stream.accept_waveform(sample_rate, tail)
    stream.input_finished()
    while sherpa_rec.is_ready(stream):
        sherpa_rec.decode_stream(stream)
    raw = sherpa_rec.get_result(stream).strip()
    return _normalize_clinical_stt(raw)


def _transcribe_google_pcm(pcm_bytes: bytes, sample_rate: int = 16000, timeout_sec: float = 1.35) -> str:
    """Fast bounded-timeout Google Speech Recognition verifier over trimmed, un-clipped speech PCM."""
    if not _SR_AVAILABLE or sr is None or not pcm_bytes:
        return ""
    pcm_clean = _trim_speech_pcm(pcm_bytes, sample_rate)
    if not pcm_clean:
        return ""
    result_box: list[str] = []

    def _worker() -> None:
        try:
            recognizer = sr.Recognizer()
            audio_data = sr.AudioData(pcm_clean, sample_rate, 2)
            text = recognizer.recognize_google(audio_data, language="en-US").strip()
            if text:
                result_box.append(_normalize_clinical_stt(text))
        except Exception:
            pass

    th = threading.Thread(target=_worker, daemon=True)
    th.start()
    th.join(timeout=timeout_sec)
    return result_box[0] if result_box else ""


def _select_best_final_transcript(
    google_text: str,
    sherpa_text: str,
    pcm_bytes: bytes,
    sample_rate: int = 16000,
) -> str:
    """Select the highest-accuracy transcript from Google Speech, Sherpa-ONNX, and Whisper fallback."""
    g_clean = google_text.strip()
    s_clean = sherpa_text.strip()
    if g_clean:
        # If Google returned only 1 word while Sherpa heard a full multi-word sentence, keep the fuller one
        g_words = len(g_clean.split())
        s_words = len(s_clean.split())
        if g_words == 1 and s_words >= 4:
            return s_clean
        return g_clean
    if s_clean:
        return s_clean
    if pcm_bytes:
        return _transcribe_pcm_bytes(pcm_bytes, sample_rate)
    return ""


def _transcribe_pcm_bytes(pcm_bytes: bytes, sample_rate: int = 16000) -> str:
    """Transcribe 16-bit mono PCM via Sherpa-ONNX Streaming Zipformer + Google STT + clean VAD Whisper fallback."""
    if not pcm_bytes:
        return ""
    audio_f32 = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    norm_audio, raw_rms = _normalize_audio_f32(audio_f32)
    if norm_audio.size == 0:
        return ""

    # 1. Instant local C++ Sherpa-ONNX Zipformer Transducer
    sherpa_res = _transcribe_sherpa_f32(audio_f32, sample_rate)
    if sherpa_res:
        return sherpa_res

    # 2. Fast Google Speech Recognition
    google_res = _transcribe_google_pcm(pcm_bytes, sample_rate, timeout_sec=1.0)
    if google_res:
        return google_res

    # 3. Clean faster-whisper fallback (NO initial_prompt so it never hallucinates prompt continuations)
    try:
        model = _get_whisper_model()
        segments, _info = model.transcribe(
            norm_audio,
            beam_size=1,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 250, "speech_pad_ms": 150},
            language="en",
            condition_on_previous_text=False,
            without_timestamps=True,
            temperature=0.0,
        )
        parts = [
            seg.text.strip()
            for seg in segments
            if seg.text.strip() and getattr(seg, "no_speech_prob", 0.0) <= 0.75
        ]
        result = _normalize_clinical_stt(" ".join(parts).strip())
        return "" if _is_whisper_hallucination(result, raw_rms) else result
    except Exception:
        return ""


def _load_audio_file_16k_float32(audio_path: str) -> np.ndarray:
    """Decode WAV or browser audio file into 16kHz mono float32 PCM without PyAV."""
    try:
        with wave.open(audio_path, "rb") as wf:
            nchannels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            framerate = wf.getframerate()
            frames = wf.readframes(wf.getnframes())
        if sampwidth == 2 and frames:
            samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
            if nchannels > 1:
                usable = (samples.size // nchannels) * nchannels
                samples = samples[:usable].reshape(-1, nchannels).mean(axis=1)
            if framerate != 16000 and samples.size > 0:
                target_len = max(1, int(round(samples.size * 16000.0 / float(framerate))))
                x_old = np.linspace(0.0, 1.0, samples.size, endpoint=False)
                x_new = np.linspace(0.0, 1.0, target_len, endpoint=False)
                samples = np.interp(x_new, x_old, samples).astype(np.float32)
            return samples
    except Exception:
        pass

    if _MINIAUDIO_AVAILABLE and miniaudio is not None:
        try:
            decoded = miniaudio.decode_file(
                audio_path,
                output_format=miniaudio.SampleFormat.SIGNED16,
                nchannels=1,
                sample_rate=16000,
            )
            samples = np.frombuffer(decoded.samples, dtype=np.int16).astype(np.float32) / 32768.0
            return samples
        except Exception:
            pass

    return np.zeros(0, dtype=np.float32)


def _transcribe_wav_file(wav_path: str) -> str:
    """Transcribe WAV/audio file via Google Speech + Sherpa-ONNX Zipformer + clean VAD Whisper fallback."""
    audio_f32 = _load_audio_file_16k_float32(wav_path)
    norm_audio, raw_rms = _normalize_audio_f32(audio_f32)
    if norm_audio.size == 0:
        return ""

    pcm16_bytes = (np.clip(audio_f32, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()
    dur_sec = audio_f32.size / 16000.0
    google_res = _transcribe_google_pcm(pcm16_bytes, 16000, timeout_sec=min(2.8, max(1.5, dur_sec * 0.16)))
    if google_res:
        return google_res

    sherpa_res = _transcribe_sherpa_f32(audio_f32, 16000)
    if sherpa_res:
        return sherpa_res

    try:
        model = _get_whisper_model()
        segments, _info = model.transcribe(
            norm_audio,
            beam_size=1,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 250, "speech_pad_ms": 150},
            language="en",
            condition_on_previous_text=False,
            without_timestamps=True,
            temperature=0.0,
        )
        parts = [
            seg.text.strip()
            for seg in segments
            if seg.text.strip() and getattr(seg, "no_speech_prob", 0.0) <= 0.75
        ]
        result = _normalize_clinical_stt(" ".join(parts).strip())
        return "" if _is_whisper_hallucination(result, raw_rms) else result
    except ImportError:
        pass

    if not _SR_AVAILABLE or sr is None:
        raise RuntimeError(
            "Neither faster-whisper nor SpeechRecognition is installed for STT."
        )

    recognizer = sr.Recognizer()
    with sr.AudioFile(wav_path) as source:
        audio_data = recognizer.record(source)
    try:
        return _normalize_clinical_stt(recognizer.recognize_google(audio_data, language="en-US").strip())
    except sr.UnknownValueError:
        return ""
    except sr.RequestError as exc:
        raise RuntimeError(f"Speech recognition service unavailable: {exc}") from exc


@app.after_request
def add_cors_headers(response: Any) -> Any:
    """Enable CORS so the React/Vite clinician UI can poll and post to port 5000."""
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response


@app.route("/")
def index() -> str:
    """Render the dark-mode mobile-responsive Patient Bedside Web UI."""
    return render_template("bedside.html")


@app.route("/api/state", methods=["GET", "OPTIONS"])
def api_state() -> Any:
    """Return deterministic locked patient state for both React UI bridge and Bedside Web UI."""
    if request.method == "OPTIONS":
        return ("", 204)
    state = state_manager.load_state()
    return jsonify(_build_unified_state_payload(state))


@app.route("/api/clinician", methods=["POST", "OPTIONS"])
def api_clinician() -> Any:
    """Accept a clinician voice/text command and advance the Malcolm RAG triage workflow."""
    if request.method == "OPTIONS":
        return ("", 204)

    payload = request.get_json(silent=True) or {}
    message = str(payload.get("message", "")).strip()
    if not message:
        return jsonify({"ok": False, "error": "Please provide a clinician message."}), 400

    return jsonify(_run_clinician_message(message))


@app.route("/api/tts", methods=["GET", "POST", "OPTIONS"])
def api_tts() -> Any:
    """Synthesize calm 1.0x speed British neural audio with dead-air pauses removed."""
    if request.method == "OPTIONS":
        return ("", 204)

    if request.method == "GET":
        text = str(request.args.get("text", "")).strip()
    else:
        payload = request.get_json(silent=True) or {}
        text = str(payload.get("text", "")).strip()

    if not text or not _clean_text_for_speech(text):
        return jsonify({"ok": False, "error": "Text is required for TTS."}), 400

    try:
        audio_bytes, mimetype = _synthesize_jarvis_audio(text, voice=TTS_VOICE)
        return Response(
            audio_bytes,
            mimetype=mimetype,
            headers={
                "Cache-Control": "no-store",
                "Accept-Ranges": "none",
                "Content-Length": str(len(audio_bytes)),
            },
        )
    except Exception as exc:
        return (
            jsonify(
                {
                    "ok": False,
                    "fallback": "browser_tts",
                    "error": str(exc),
                }
            ),
            503,
        )


_active_mic_target: str = "clinician"


def _synthesize_code_blue_alarm_wav(sample_rate: int = 24000) -> bytes:
    """Synthesize an urgent IEC-60601 High-Priority Code Blue medical alarm burst (C6-E6-G6-C6) as WAV bytes."""
    pulses = [
        (1046.50, 0.13, 0.05),
        (1318.51, 0.13, 0.05),
        (1567.98, 0.13, 0.12),
        (1046.50, 0.16, 0.05),
        (1567.98, 0.22, 0.14),
    ]
    segments: list[np.ndarray] = []
    for freq, on_dur, off_dur in pulses:
        t = np.linspace(0.0, on_dur, int(sample_rate * on_dur), endpoint=False)
        env = np.minimum(1.0, t / 0.008) * np.minimum(1.0, (on_dur - t) / 0.012)
        tone = (
            0.60 * np.sin(2.0 * np.pi * freq * t)
            + 0.28 * np.sin(2.0 * np.pi * (freq * 2.0) * t)
            + 0.12 * np.sin(2.0 * np.pi * (freq * 3.0) * t)
        ) * env
        segments.append(tone)
        if off_dur > 0:
            segments.append(np.zeros(int(sample_rate * off_dur)))
    full = np.concatenate(segments)
    pcm16 = np.clip(full * 25000.0, -32767, 32767).astype(np.int16)
    return _pcm16_to_wav_bytes(pcm16, sample_rate=sample_rate)


def _broadcast_bpm_emergency_page(
    state: dict[str, Any],
    bpm: int,
    level: str,
) -> dict[str, Any]:
    """
    Broadcast either a High-BPM Doctor Page or a Critical Code Blue Crash Cart Demand
    over the hospital loudspeaker & notify the doctor on the main AI screen.
    """
    pid = str(state.get("patient_id", "1429"))
    name = str(state.get("name", "Mika Hämäläinen"))
    wing = str(state.get("wing", "Wing A"))
    room = str(state.get("room", "Room 153"))

    if level == "critical":
        title = f"CODE BLUE — CRASH CART DEMANDED ({bpm} BPM)"
        announcement = (
            f"CODE BLUE EMERGENCY! Attention Doctor and Crash Cart Team: "
            f"Patient {pid}, {name} in {wing}, {room} is in critical tachycardia at {bpm} BPM! "
            f"Crash cart demanded to {wing}, {room} immediately!"
        )
    else:
        title = f"HIGH BPM ALERT — DOCTOR PAGED ({bpm} BPM)"
        announcement = (
            f"Hospital Paging System. Attention Doctor: Patient {pid}, {name} in {wing}, {room} "
            f"has an elevated heart rate of {bpm} BPM. Please review telemetry or check in immediately."
        )

    alert_obj = {
        "id": f"emerg_{level}_{int(time.time() * 1000)}",
        "eventId": f"bpm_{level}_{pid}",
        "severity": level,
        "crashCart": level == "critical",
        "patientId": pid,
        "patientName": name,
        "title": title,
        "location": f"{wing} • {room}",
        "displayTime": datetime.now().strftime("%I:%M %p").lstrip("0"),
        "reminderMinutesBefore": 0,
        "announcement": announcement,
        "timestamp": datetime.now().strftime("%I:%M:%S %p").lstrip("0"),
        "isTest": False,
    }
    state_manager.record_page_alert(alert_obj)
    _prewarm_reply_sentences_async(announcement)

    def _play_emergency_speaker() -> None:
        if not _WINSOUND_AVAILABLE or not _winsound_enabled or winsound is None:
            return
        with _speaker_page_lock:
            try:
                tone_wav = (
                    _synthesize_code_blue_alarm_wav(24000)
                    if level == "critical"
                    else _synthesize_hospital_pa_chime_wav(24000)
                )
                if _winsound_enabled:
                    winsound.PlaySound(tone_wav, winsound.SND_MEMORY)
            except Exception:
                pass
            if not _winsound_enabled:
                return
            try:
                speech_bytes, mime = _synthesize_jarvis_audio(announcement, voice=TTS_VOICE)
                if _winsound_enabled and mime == "audio/wav" and speech_bytes:
                    winsound.PlaySound(speech_bytes, winsound.SND_MEMORY)
            except Exception:
                pass

    threading.Thread(target=_play_emergency_speaker, daemon=True).start()
    return alert_obj


def _apply_bedside_bpm_and_alerts(
    patient_id: str,
    bpm: int,
    reason: str = "",
) -> tuple[dict[str, Any], str]:
    """
    Update a patient's live BPM, evaluate the 3-tier threshold:
      - Normal (< 115 BPM): Blue background, normal monitoring
      - High (115-144 BPM): Red background on bedside, main AI pages the Doctor
      - Critical (>= 145 BPM): Alarm goes off, demands Crash Cart, shows Crash Cart Disable button on bedside, notifies Doctor on main AI
    Returns (updated_state, bedside_spoken_reply).
    """
    bpm_clamped = max(35, min(220, int(bpm)))
    state_manager.set_active_patient(patient_id)
    state = state_manager.load_state(patient_id)
    name = str(state.get("name", "Mika Hämäläinen"))
    first_name = name.split()[0] if name else "Patient"
    wing = str(state.get("wing", "Wing A"))
    room = str(state.get("room", "Room 153"))

    if bpm_clamped >= 145:
        level = "critical"
        wave_state = f"CRITICAL VENTRICULAR TACHYCARDIA ({bpm_clamped} BPM) — CODE BLUE"
        rhythm_short = "Critical Tachycardia — CODE BLUE"
        emerg_msg = f"CODE BLUE ({bpm_clamped} BPM) — CRASH CART DEMANDED TO {wing.upper()}, {room.upper()}"
        bedside_reply = (
            f"CRITICAL EMERGENCY: Heart rate is {bpm_clamped} BPM. "
            f"Code Blue alarm activated. Demanding a Crash Cart and paging your attending doctor to {wing}, {room} immediately!"
        )
        alert_obj = _broadcast_bpm_emergency_page(state, bpm_clamped, level="critical")
        main_transcript = alert_obj["announcement"]
    elif bpm_clamped >= 115:
        level = "high"
        wave_state = f"Sustained Sinus Tachycardia ({bpm_clamped} BPM) — Doctor Paged"
        rhythm_short = "Sustained Sinus Tachycardia"
        emerg_msg = f"HIGH BPM ({bpm_clamped} BPM) — ATTENDING DOCTOR PAGED BY MALCOLM"
        bedside_reply = (
            f"{first_name}, I notice your heart rate is elevated at {bpm_clamped} BPM. "
            f"I have just paged your attending doctor to {wing}, {room}. Please rest quietly and take slow breaths."
        )
        alert_obj = _broadcast_bpm_emergency_page(state, bpm_clamped, level="high")
        main_transcript = alert_obj["announcement"]
    else:
        level = "normal"
        if bpm_clamped < 68:
            wave_state = "Sleeping / Slow Sinus Rhythm"
            rhythm_short = "Slow Sinus Rhythm"
        elif bpm_clamped >= 100:
            wave_state = "Awake / Mild Sinus Tachycardia"
            rhythm_short = "Mild Sinus Tachycardia"
        else:
            wave_state = "Awake / Normal Sinus Rhythm"
            rhythm_short = "Normal Sinus Rhythm"
        emerg_msg = ""
        bedside_reply = (
            f"Heart rate is stable at {bpm_clamped} BPM ({rhythm_short}). "
            "Malcolm is monitoring your comfort quietly."
        )
        main_transcript = state.get("last_transcript", "Listening for clinician request...")

    telemetry = {"bpm": bpm_clamped, "wave_state": wave_state}
    projected = dict(state)
    projected["current_bpm"] = bpm_clamped
    projected["wave_state"] = wave_state
    projected["alert_level"] = level
    projected["doctor_paged"] = level in {"high", "critical"}
    projected["crash_cart_active"] = level == "critical"
    projected["emergency_message"] = emerg_msg
    if level in {"high", "critical"}:
        sym_label = (
            f"CRITICAL CODE BLUE: {bpm_clamped} BPM — Crash Cart Demanded"
            if level == "critical"
            else f"Elevated Heart Rate ({bpm_clamped} BPM) — Doctor Paged"
        )
        syms = list(projected.get("patient_symptoms") or [])
        if sym_label not in syms:
            syms.append(sym_label)
        projected["patient_symptoms"] = syms
    panel = rag_agent.synthesize_patient_panel(projected, telemetry)
    panel["rhythm"] = rhythm_short
    panel["telemetryDetail"] = (
        f"CRITICAL {bpm_clamped} BPM • CRASH CART DEMANDED"
        if level == "critical"
        else (
            f"HIGH {bpm_clamped} BPM • DOCTOR PAGED"
            if level == "high"
            else f"Live {bpm_clamped} BPM • Baseline {state.get('baseline_bpm', 90)} BPM • Signal Locked"
        )
    )
    if level == "critical":
        panel["dispatchStatus"] = f"CODE BLUE — CRASH CART & DOCTOR DEMANDED TO {wing.upper()}, {room.upper()}"
    elif level == "high":
        panel["dispatchStatus"] = f"HIGH BPM ({bpm_clamped}) — ATTENDING DOCTOR PAGED TO {wing.upper()}, {room.upper()}"

    def _mutate(s: dict[str, Any]) -> None:
        s["current_bpm"] = bpm_clamped
        s["wave_state"] = wave_state
        s["alert_level"] = level
        s["doctor_paged"] = level in {"high", "critical"}
        s["crash_cart_active"] = level == "critical"
        s["emergency_message"] = emerg_msg
        s["patient_panel"] = panel
        if level in {"high", "critical"}:
            s["ui_mode"] = "patient_stats"
            s["custom_pane"] = {
                "kind": "patient",
                "title": "",
                "subtitle": "",
                "filterWing": "",
                "generatedHtml": "",
            }
            s["last_transcript"] = main_transcript
            s["patient_symptoms"] = projected.get("patient_symptoms", [])

    updated = state_manager.update_state(
        _mutate,
        patient_id=patient_id,
        announce=(level in {"high", "critical"}),
    )
    if level in {"high", "critical"} or reason:
        state_manager.add_bedside_message("malcolm", bedside_reply, patient_id=patient_id)
        updated = state_manager.load_state(patient_id)
    _prewarm_reply_sentences_async(bedside_reply)
    return updated, bedside_reply


def _run_bedside_message(message: str) -> dict[str, Any]:
    """Handle a patient voice or text utterance from the Bedside App and return a unified payload."""
    cleaned = message.strip()
    active_id = state_manager.get_active_patient_id()
    if not cleaned:
        state = state_manager.load_state(active_id)
        return {"ok": False, "error": "Please enter or speak a message.", "state": state}

    lowered = cleaned.lower()
    # Check if the patient's spoken message describes an acute cardiac/respiratory crisis
    is_critical_phrase = any(
        kw in lowered
        for kw in (
            "can't breathe",
            "cannot breathe",
            "can't catch my breath",
            "crushing chest",
            "severe chest pain",
            "code blue",
            "crash cart",
            "heart attack",
            "fainting",
            "passing out",
        )
    )
    is_high_hr_phrase = not is_critical_phrase and any(
        kw in lowered
        for kw in (
            "heart is racing",
            "heart is beating really fast",
            "heart pounding",
            "palpitations",
            "chest feels tight",
            "chest pain",
        )
    )

    result = rag_agent.process_bedside_turn(cleaned)
    updated = result.get("state") or state_manager.load_state(active_id)
    reply_text = str(result.get("reply") or "")

    if is_critical_phrase:
        updated, emerg_reply = _apply_bedside_bpm_and_alerts(active_id, 168, reason="patient_voice_critical")
        reply_text = emerg_reply
    elif is_high_hr_phrase:
        updated, high_reply = _apply_bedside_bpm_and_alerts(active_id, 128, reason="patient_voice_high")
        reply_text = high_reply
    else:
        _prewarm_reply_sentences_async(reply_text)

    return {
        "ok": bool(result.get("ok", True)),
        "transcript": cleaned,
        "reply": reply_text,
        "engine": result.get("engine", rag_agent.get_engine_label()),
        "stage": updated.get("conversation_stage", "idle"),
        "check_in_active": bool(updated.get("check_in_active")),
        "active_patient_id": updated.get("patient_id", active_id),
        "patients_list": state_manager.list_patients(),
        "state": updated,
        "ui_state": updated.get("ui_mode", "idle"),
        "data": _build_bridge_data(updated),
        "raw_state": updated,
    }


def _finalize_stt_response(wav_path: str, target: str = "clinician") -> Any:
    """Transcribe a temporary WAV file, advance Malcolm triage or bedside state, and return JSON."""
    try:
        transcript = _transcribe_wav_file(wav_path)
    except RuntimeError as exc:
        code = 503 if "installed" in str(exc).lower() or "unavailable" in str(exc).lower() else 500
        return jsonify({"ok": False, "error": str(exc)}), code
    except Exception as exc:
        return jsonify({"ok": False, "error": f"STT failed: {exc}"}), 500
    finally:
        if wav_path and os.path.exists(wav_path):
            try:
                os.remove(wav_path)
            except OSError:
                pass

    if not transcript:
        return jsonify({"ok": False, "error": "No speech detected in audio."}), 400

    runner = _run_bedside_message if target == "bedside" else _run_clinician_message
    response_payload = runner(transcript)
    response_payload["transcript"] = str(response_payload.get("transcript") or transcript)
    return jsonify(response_payload)


@app.route("/api/stt", methods=["POST", "OPTIONS"])
def api_stt() -> Any:
    """Transcribe uploaded WAV audio and advance clinician or bedside workflow."""
    if request.method == "OPTIONS":
        return ("", 204)

    target = str(request.args.get("target", "clinician")).strip().lower()
    audio_file = request.files.get("audio")
    if audio_file is None:
        return jsonify({"ok": False, "error": "Missing 'audio' file upload."}), 400

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
    audio_file.save(tmp_path)
    return _finalize_stt_response(tmp_path, target=target)


@app.route("/api/mic/start", methods=["POST", "OPTIONS"])
def api_mic_start() -> Any:
    """Start capturing speech from all active physical Windows microphones for clinician or bedside."""
    global _active_mic_target
    if request.method == "OPTIONS":
        return ("", 204)

    target = str(request.args.get("target", "")).strip().lower()
    if not target:
        payload = request.get_json(silent=True) or {}
        target = str(payload.get("target", "clinician")).strip().lower()
    _active_mic_target = "bedside" if target == "bedside" else "clinician"
    runner = _run_bedside_message if _active_mic_target == "bedside" else _run_clinician_message

    try:
        device_name = _host_recorder.start(on_final=runner)
        status = _host_recorder.get_live_status()
        return jsonify(
            {
                "ok": True,
                "target": _active_mic_target,
                "device": device_name,
                "devices": status.get("devices", []),
            }
        )
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 503


@app.route("/api/mic/live", methods=["GET", "OPTIONS"])
def api_mic_live() -> Any:
    """Return real-time partial transcription and VAD auto-finalization status."""
    if request.method == "OPTIONS":
        return ("", 204)
    return jsonify(_host_recorder.get_live_status())


@app.route("/api/mic/stream", methods=["GET", "OPTIONS"])
def api_mic_stream() -> Any:
    """Server-Sent Events stream pushing partial transcripts every ~80ms until completion."""
    if request.method == "OPTIONS":
        return ("", 204)

    def _sse_events() -> Any:
        last_sent = ""
        while True:
            status = _host_recorder.get_live_status()
            serialized = json.dumps(status)
            if serialized != last_sent:
                last_sent = serialized
                yield f"data: {serialized}\n\n"
            if not status.get("recording"):
                break
            time.sleep(0.05)

    return Response(
        _sse_events(),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
    )


@app.route("/api/mic/stop", methods=["POST", "OPTIONS"])
def api_mic_stop() -> Any:
    """Stop Windows host microphone capture, verify transcript via Google STT + Sherpa, and route to Malcolm."""
    if request.method == "OPTIONS":
        return ("", 204)

    if request.args.get("discard") == "1":
        _host_recorder.stop_and_discard()
        return jsonify({"ok": True, "discarded": True})

    target = str(request.args.get("target", "")).strip().lower() or _active_mic_target
    runner = _run_bedside_message if target == "bedside" else _run_clinician_message

    live_status = _host_recorder.get_live_status()
    if live_status.get("final_payload"):
        return jsonify(live_status["final_payload"])

    cached_partial = str(live_status.get("live_transcript") or "").strip()
    wav_bytes = _host_recorder.stop_to_wav_bytes()

    # Check again in case the background transcriber thread finished right as stop_to_wav_bytes joined it
    post_status = _host_recorder.get_live_status()
    if post_status.get("final_payload"):
        return jsonify(post_status["final_payload"])

    if not wav_bytes:
        if not cached_partial:
            return jsonify({"ok": False, "error": "No audio captured from host microphone."}), 400
        response_payload = runner(cached_partial)
        response_payload["transcript"] = str(response_payload.get("transcript") or cached_partial)
        return jsonify(response_payload)

    tmp_path = ""
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(wav_bytes)
            tmp_path = tmp.name
        transcript = _transcribe_wav_file(tmp_path)
    except RuntimeError as exc:
        code = 503 if "installed" in str(exc).lower() or "unavailable" in str(exc).lower() else 500
        return jsonify({"ok": False, "error": str(exc)}), code
    except Exception as exc:
        return jsonify({"ok": False, "error": f"STT failed: {exc}"}), 500
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    final_transcript = _select_best_final_transcript(transcript, cached_partial, b"")
    if not final_transcript:
        return jsonify({"ok": False, "error": "No speech detected in audio."}), 400

    response_payload = runner(final_transcript)
    response_payload["transcript"] = str(response_payload.get("transcript") or final_transcript)
    return jsonify(response_payload)


@app.route("/api/message", methods=["POST", "OPTIONS"])
@app.route("/api/chat", methods=["POST", "OPTIONS"])
def api_message() -> Any:
    """Accept a patient bedside message and route it through Malcolm's bedside state machine."""
    if request.method == "OPTIONS":
        return ("", 204)

    payload = request.get_json(silent=True) or {}
    message = str(payload.get("message", "")).strip()
    if not message:
        return jsonify({"ok": False, "error": "Please enter a message."}), 400

    response_payload = _run_bedside_message(message)
    status_code = 200 if response_payload.get("ok") else 409
    return jsonify(response_payload), status_code


@app.route("/api/bedside/telemetry", methods=["POST", "OPTIONS"])
def api_bedside_telemetry() -> Any:
    """
    Simulate or update bedside patient BPM and trigger the 3-tier response:
      - Normal (< 115 BPM): Blue background
      - High (115-144 BPM): Red background + main AI pages the Doctor
      - Critical (>= 145 BPM): Alarm + Crash Cart demand + Crash Cart disable button + Doctor notified
    """
    if request.method == "OPTIONS":
        return ("", 204)

    payload = request.get_json(silent=True) or {}
    patient_id = str(payload.get("patient_id") or state_manager.get_active_patient_id()).strip()
    scenario = str(payload.get("scenario") or "").strip().lower()
    raw_bpm = payload.get("bpm")

    state = state_manager.load_state(patient_id)
    if scenario == "normal" and raw_bpm is None:
        target_bpm = int(state.get("baseline_bpm", 86))
    elif scenario == "high" and raw_bpm is None:
        target_bpm = 128
    elif scenario == "critical" and raw_bpm is None:
        target_bpm = 168
    else:
        try:
            target_bpm = int(raw_bpm if raw_bpm is not None else state.get("current_bpm", 90))
        except (TypeError, ValueError):
            target_bpm = 90

    updated, bedside_reply = _apply_bedside_bpm_and_alerts(
        patient_id=patient_id,
        bpm=target_bpm,
        reason=scenario or "telemetry_update",
    )
    return jsonify(
        {
            "ok": True,
            "reply": bedside_reply,
            "active_patient_id": patient_id,
            "patients_list": state_manager.list_patients(),
            "state": updated,
            "ui_state": updated.get("ui_mode", "patient_stats"),
            "data": _build_bridge_data(updated),
            "raw_state": updated,
        }
    )


@app.route("/api/bedside/ack_crash_cart", methods=["POST", "OPTIONS"])
def api_bedside_ack_crash_cart() -> Any:
    """
    Triggered when the Crash Cart team pushes the 'CRASH CART ARRIVED — DISABLE ALARM' button
    on the Bedside App. Disables the alarm, stabilizes BPM, and notifies the Doctor on the main AI.
    """
    if request.method == "OPTIONS":
        return ("", 204)

    payload = request.get_json(silent=True) or {}
    patient_id = str(payload.get("patient_id") or state_manager.get_active_patient_id()).strip()
    state = state_manager.load_state(patient_id)
    baseline_bpm = min(92, int(state.get("baseline_bpm", 86)))
    name = str(state.get("name", "Mika Hämäläinen"))
    wing = str(state.get("wing", "Wing A"))
    room = str(state.get("room", "Room 153"))

    doctor_notification = (
        f"Update for Doctor: The Crash Cart team has arrived at {wing}, {room} "
        f"and disabled the bedside Code Blue alarm. Patient {patient_id}, {name} is stabilized at {baseline_bpm} BPM."
    )
    bedside_confirmation = (
        f"Crash Cart response acknowledged. Code Blue alarm disabled. "
        f"{name}'s rhythm is stabilized at {baseline_bpm} BPM."
    )
    now_hm = datetime.now().strftime("%H:%M")
    rn_note = f"{now_hm} Crash Cart Team: Code Blue alarm acknowledged at bedside; rhythm stabilized at {baseline_bpm} BPM."

    telemetry = {"bpm": baseline_bpm, "wave_state": "Awake / Normal Sinus Rhythm (Stabilized)"}
    projected = dict(state)
    projected["current_bpm"] = baseline_bpm
    projected["wave_state"] = telemetry["wave_state"]
    projected["alert_level"] = "normal"
    projected["doctor_paged"] = False
    projected["crash_cart_active"] = False
    projected["emergency_message"] = ""
    notes = list(projected.get("nurse_notes") or [])
    notes.append(rn_note)
    projected["nurse_notes"] = notes
    panel = rag_agent.synthesize_patient_panel(projected, telemetry)
    panel["dispatchStatus"] = f"CRASH CART ARRIVED AT {wing.upper()}, {room.upper()} — ALARM DISABLED & STABILIZED"

    def _ack_mutate(s: dict[str, Any]) -> None:
        s["current_bpm"] = baseline_bpm
        s["wave_state"] = telemetry["wave_state"]
        s["alert_level"] = "normal"
        s["doctor_paged"] = False
        s["crash_cart_active"] = False
        s["emergency_message"] = ""
        s["nurse_notes"] = notes
        s["patient_panel"] = panel
        s["ui_mode"] = "patient_stats"
        s["last_transcript"] = doctor_notification

    updated = state_manager.update_state(_ack_mutate, patient_id=patient_id, announce=True)
    state_manager.add_bedside_message("malcolm", bedside_confirmation, patient_id=patient_id)
    state_manager.record_page_alert(
        {
            "id": f"ack_crash_{int(time.time() * 1000)}",
            "eventId": f"ack_crash_{patient_id}",
            "severity": "resolved",
            "crashCart": False,
            "patientId": patient_id,
            "patientName": name,
            "title": "CRASH CART ARRIVED — ALARM DISABLED",
            "location": f"{wing} • {room}",
            "displayTime": datetime.now().strftime("%I:%M %p").lstrip("0"),
            "reminderMinutesBefore": 0,
            "announcement": doctor_notification,
            "timestamp": datetime.now().strftime("%I:%M:%S %p").lstrip("0"),
            "isTest": False,
        }
    )
    updated = state_manager.load_state(patient_id)
    _prewarm_reply_sentences_async(doctor_notification)
    _prewarm_reply_sentences_async(bedside_confirmation)
    return jsonify(
        {
            "ok": True,
            "reply": bedside_confirmation,
            "doctor_notification": doctor_notification,
            "active_patient_id": patient_id,
            "patients_list": state_manager.list_patients(),
            "state": updated,
            "ui_state": updated.get("ui_mode", "patient_stats"),
            "data": _build_bridge_data(updated),
            "raw_state": updated,
        }
    )


@app.route("/api/select_patient", methods=["POST", "OPTIONS"])
def api_select_patient() -> Any:
    """Switch the active patient in patients_db.json and synthesize their RAG split-screen panel."""
    if request.method == "OPTIONS":
        return ("", 204)

    payload = request.get_json(silent=True) or {}
    patient_id = str(payload.get("patient_id", "")).strip()
    if not patient_id:
        return jsonify({"ok": False, "error": "patient_id is required."}), 400

    try:
        state = state_manager.set_active_patient(patient_id)
    except KeyError:
        return jsonify({"ok": False, "error": f"Patient {patient_id} not found."}), 404

    telemetry = telemetry_provider.get_reading(
        baseline_bpm=int(state.get("baseline_bpm", 90)),
        wave_state=str(state.get("wave_state", "Awake / Normal Sinus Rhythm")),
    )
    panel = rag_agent.synthesize_patient_panel(state, telemetry)
    transcript = (
        f"Loaded chart for Patient {state['patient_id']}, {state['name']} "
        f"in {state['wing']}, {state['room']}."
    )
    updated = state_manager.update_state(
        {
            "ui_mode": "patient_stats",
            "custom_pane": {
                "kind": "patient",
                "title": "",
                "subtitle": "",
                "filterWing": "",
            },
            "current_bpm": telemetry["bpm"],
            "wave_state": telemetry["wave_state"],
            "last_clinician_input": f"Open chart for patient {patient_id}",
            "last_transcript": transcript,
            "patient_panel": panel,
        },
        patient_id=patient_id,
    )
    return jsonify(
        {
            "ok": True,
            "active_patient_id": patient_id,
            "patients_list": state_manager.list_patients(),
            "state": updated["ui_mode"],
            "data": _build_bridge_data(updated),
            "raw_state": updated,
        }
    )


@app.route("/api/patients/add", methods=["POST", "OPTIONS"])
def api_patients_add() -> Any:
    """Admit/add a new patient to patients_db.json and refresh the Ward Roster."""
    if request.method == "OPTIONS":
        return ("", 204)

    payload = request.get_json(silent=True) or {}
    name = str(payload.get("name", "Alex Rivera")).strip() or "Alex Rivera"
    wing = str(payload.get("wing", "Wing B")).strip() or "Wing B"
    room = str(payload.get("room", "Room 210")).strip() or "Room 210"
    condition = str(payload.get("condition", "General Clinical Observation")).strip() or "General Clinical Observation"
    department = str(payload.get("department", "Internal Medicine")).strip() or "Internal Medicine"
    protocol = str(payload.get("protocol", "Continuous Telemetry & Q4H Vitals")).strip() or "Continuous Telemetry & Q4H Vitals"
    try:
        bpm = int(payload.get("baseline_bpm", payload.get("bpm", 82)))
    except (TypeError, ValueError):
        bpm = 82

    created = state_manager.add_patient(
        name=name,
        wing=wing,
        room=room,
        condition=condition,
        department=department,
        protocol=protocol,
        baseline_bpm=bpm,
        symptom=str(payload.get("symptom", "")),
        nurse_note=str(payload.get("nurse_note", "")),
        patient_id=str(payload.get("patient_id", "")).strip() or None,
    )
    new_id = str(created.get("patient_id", "1433"))
    total_count = len(state_manager.load_db().get("patients", {}))
    reply = (
        f"Admitted Patient {new_id}, {created.get('name')} to {created.get('wing')}, "
        f"{created.get('room')} ({created.get('condition')}, {created.get('current_bpm')} BPM). "
        f"The ward roster now has {total_count} monitored patients."
    )
    updated = state_manager.update_state(
        {
            "ui_mode": "patient_stats",
            "custom_pane": {
                "kind": "ward_table",
                "title": "Active Ward Roster — All Monitored Patients",
                "subtitle": "Real-Time Multi-Patient EHR & Telemetry Matrix",
                "filterWing": "",
                "generatedHtml": "",
            },
            "last_transcript": reply,
        },
        patient_id=new_id,
    )
    _prewarm_reply_sentences_async(reply)
    return jsonify(
        {
            "ok": True,
            "reply": reply,
            "active_patient_id": new_id,
            "patients_list": state_manager.list_patients(),
            "state": updated["ui_mode"],
            "data": _build_bridge_data(updated),
            "raw_state": updated,
        }
    )


@app.route("/api/patients/delete", methods=["POST", "OPTIONS"])
def api_patients_delete() -> Any:
    """Discharge/remove a patient from patients_db.json and refresh the Ward Roster."""
    if request.method == "OPTIONS":
        return ("", 204)

    payload = request.get_json(silent=True) or {}
    patient_id = str(payload.get("patient_id", "")).strip()
    if not patient_id:
        return jsonify({"ok": False, "error": "patient_id is required."}), 400

    deleted_entry, next_active = state_manager.delete_patient(patient_id)
    if deleted_entry is None:
        return jsonify({"ok": False, "error": f"Patient {patient_id} not found."}), 404

    next_id = str(next_active.get("patient_id", state_manager.get_active_patient_id()))
    rem_count = len(state_manager.load_db().get("patients", {}))
    del_name = deleted_entry.get("name", f"Patient {patient_id}")
    reply = (
        f"Discharged and removed Patient {patient_id}, {del_name} from the active ward database. "
        f"We now have {rem_count} monitored patient{'s' if rem_count != 1 else ''} on the floor."
    )
    telemetry = telemetry_provider.get_reading(
        baseline_bpm=int(next_active.get("baseline_bpm", 90)),
        wave_state=str(next_active.get("wave_state", "Awake / Normal Sinus Rhythm")),
    )
    panel = rag_agent.synthesize_patient_panel(next_active, telemetry)
    updated = state_manager.update_state(
        {
            "ui_mode": "patient_stats",
            "custom_pane": {
                "kind": "ward_table",
                "title": "Active Ward Roster — All Monitored Patients",
                "subtitle": "Real-Time Multi-Patient EHR & Telemetry Matrix",
                "filterWing": "",
                "generatedHtml": "",
            },
            "last_transcript": reply,
            "patient_panel": panel,
        },
        patient_id=next_id,
    )
    _prewarm_reply_sentences_async(reply)
    return jsonify(
        {
            "ok": True,
            "reply": reply,
            "deleted_patient_id": patient_id,
            "active_patient_id": next_id,
            "patients_list": state_manager.list_patients(),
            "state": updated["ui_mode"],
            "data": _build_bridge_data(updated),
            "raw_state": updated,
        }
    )


@app.route("/api/pane", methods=["POST", "OPTIONS"])
def api_pane() -> Any:
    """Switch or close the split-screen HUD pane on demand (ward_table, ward_stats, patient, or close)."""
    if request.method == "OPTIONS":
        return ("", 204)

    payload = request.get_json(silent=True) or {}
    action = str(payload.get("action", "")).strip().lower()
    patient_id = str(payload.get("patient_id", "")).strip()

    if patient_id:
        try:
            state_manager.set_active_patient(patient_id)
        except KeyError:
            pass

    active_id = state_manager.get_active_patient_id()
    state = state_manager.load_state(active_id)
    telemetry = telemetry_provider.get_reading(
        baseline_bpm=int(state.get("baseline_bpm", 90)),
        wave_state=str(state.get("wave_state", "Awake / Normal Sinus Rhythm")),
    )
    panel = rag_agent.synthesize_patient_panel(state, telemetry)

    if action == "close":
        updates: dict[str, Any] = {
            "ui_mode": "idle",
            "custom_pane": {
                "kind": "patient",
                "title": "",
                "subtitle": "",
                "filterWing": "",
                "generatedHtml": "",
            },
            "patient_panel": panel,
        }
    elif action == "ward_table":
        wing_filter = str(payload.get("filterWing", "")).strip()
        updates = {
            "ui_mode": "patient_stats",
            "custom_pane": {
                "kind": "ward_table",
                "title": "Active Ward Roster — All Monitored Patients",
                "subtitle": "Real-Time Multi-Patient EHR & Telemetry Matrix",
                "filterWing": wing_filter,
                "generatedHtml": "",
            },
            "patient_panel": panel,
        }
    elif action == "ward_stats":
        updates = {
            "ui_mode": "patient_stats",
            "custom_pane": {
                "kind": "ward_stats",
                "title": "Ward Telemetry & Acuity Statistics",
                "subtitle": "Live Statistical Breakdown • Monitored Ward Beds",
                "filterWing": "",
                "generatedHtml": "",
            },
            "patient_panel": panel,
        }
    elif action == "timetable":
        evts = state_manager.get_calendar_events()
        if not evts:
            evts = build_timetable_events("create a time table with a 5pm surgery")
            state_manager.set_calendar_events(evts)
        updates = {
            "ui_mode": "patient_stats",
            "custom_pane": {
                "kind": "timetable",
                "title": "Clinician Shift Timetable & Reminder Schedule",
                "subtitle": "Synced to Real PC Clock • 5-Min Prior Loudspeaker Paging",
                "filterWing": "",
                "generatedHtml": render_timetable_html_css(evts),
            },
            "patient_panel": panel,
        }
    else:
        updates = {
            "ui_mode": "patient_stats",
            "custom_pane": {
                "kind": "patient",
                "title": "",
                "subtitle": "",
                "filterWing": "",
                "generatedHtml": "",
            },
            "patient_panel": panel,
        }

    updated = state_manager.update_state(updates, patient_id=active_id)
    return jsonify(
        {
            "ok": True,
            "active_patient_id": active_id,
            "patients_list": state_manager.list_patients(),
            "state": updated["ui_mode"],
            "data": _build_bridge_data(updated),
            "raw_state": updated,
        }
    )


@app.route("/api/reset", methods=["POST", "OPTIONS"])
def api_reset() -> Any:
    """Reset Patient 1429-1432 database back to idle for repeat expo/live demos."""
    if request.method == "OPTIONS":
        return ("", 204)

    fresh = state_manager.reset_state()
    return jsonify(
        {
            "ok": True,
            "active_patient_id": fresh["patient_id"],
            "patients_list": state_manager.list_patients(),
            "state": fresh["ui_mode"],
            "data": _build_bridge_data(fresh),
            "raw_state": fresh,
        }
    )


def _synthesize_hospital_pa_chime_wav(sample_rate: int = 24000) -> bytes:
    """Synthesize a warm 3-tone Hospital Loudspeaker PA Chime (G4 -> C5 -> E5) as 16-bit PCM WAV bytes."""
    notes = [
        (392.00, 0.32),  # G4
        (523.25, 0.32),  # C5
        (659.25, 0.58),  # E5
    ]
    segments: list[np.ndarray] = []
    for freq, dur in notes:
        t = np.linspace(0.0, dur, int(sample_rate * dur), endpoint=False)
        env = np.exp(-t * 4.5) * np.minimum(1.0, t / 0.012)
        wave_sig = (
            0.65 * np.sin(2.0 * np.pi * freq * t)
            + 0.25 * np.sin(2.0 * np.pi * (freq * 2.0) * t)
            + 0.10 * np.sin(2.0 * np.pi * (freq * 3.0) * t)
        ) * env
        segments.append(wave_sig)
    segments.append(np.zeros(int(sample_rate * 0.12)))
    full = np.concatenate(segments)
    pcm16 = np.clip(full * 22000.0, -32767, 32767).astype(np.int16)
    return _pcm16_to_wav_bytes(pcm16, sample_rate=sample_rate)


_speaker_page_lock = threading.Lock()
_pager_daemon_started = False


def _ensure_pager_daemon_running() -> None:
    """Start the background calendar/timer loudspeaker pager daemon if not already running."""
    global _pager_daemon_started
    if _pager_daemon_started:
        return
    _pager_daemon_started = True
    threading.Thread(target=_calendar_pager_daemon, daemon=True).start()


def _broadcast_hospital_loudspeaker_page(event: dict[str, Any], is_test: bool = False) -> dict[str, Any]:
    """
    Broadcast a Hospital Loudspeaker Page over the laptop speaker (3-tone PA chime + Jarvis voice announcement)
    and record the alert in state_manager for the clinician UI.
    """
    title = str(event.get("title", "Scheduled Clinical Task"))
    location = str(event.get("location", "Ward Floor"))
    disp_time = str(event.get("display_time", "")).split("–")[0].strip() or str(event.get("start_time", ""))
    rem_mins = int(event.get("reminder_minutes_before", 5))
    if rem_mins <= 0:
        announcement = (
            f"Hospital Paging System. Attention Doctor: this is your scheduled loudspeaker page "
            f"for {title} in {location} at {disp_time}."
        )
    else:
        announcement = (
            f"Hospital Paging System. Attention Doctor: reminder, your scheduled {title} "
            f"in {location} begins in {rem_mins} minutes at {disp_time}."
        )
    alert_obj = {
        "id": f"page_{int(time.time() * 1000)}",
        "eventId": str(event.get("id", "")),
        "title": title,
        "location": location,
        "displayTime": disp_time,
        "reminderMinutesBefore": rem_mins,
        "announcement": announcement,
        "timestamp": datetime.now().strftime("%I:%M:%S %p").lstrip("0"),
        "isTest": is_test,
    }
    state_manager.record_page_alert(alert_obj)
    active_id = state_manager.get_active_patient_id()
    state_manager.update_state({"last_transcript": announcement}, patient_id=active_id, announce=True)

    def _play_on_laptop_speaker() -> None:
        if not _WINSOUND_AVAILABLE or not _winsound_enabled or winsound is None:
            return
        with _speaker_page_lock:
            try:
                chime_wav = _synthesize_hospital_pa_chime_wav(24000)
                if _winsound_enabled:
                    winsound.PlaySound(chime_wav, winsound.SND_MEMORY)
            except Exception:
                pass
            if not _winsound_enabled:
                return
            try:
                speech_bytes, mime = _synthesize_jarvis_audio(announcement, voice=TTS_VOICE)
                if _winsound_enabled and mime == "audio/wav" and speech_bytes:
                    winsound.PlaySound(speech_bytes, winsound.SND_MEMORY)
            except Exception:
                pass

    threading.Thread(target=_play_on_laptop_speaker, daemon=True).start()
    return alert_obj


def _calendar_pager_daemon() -> None:
    """
    Background clock daemon synced to the PC's real local time.
    Fires a Hospital Loudspeaker Page on exact second-level countdown timers (`remind_at_epoch`, e.g. 'page me in 10 seconds')
    as well as 5 minutes before each scheduled timetable task (`reminder_time`).
    """
    while True:
        try:
            now = datetime.now()
            now_epoch = time.time()
            now_hm = now.strftime("%H:%M")
            events = state_manager.get_calendar_events()
            changed = False
            fired_events: list[dict[str, Any]] = []
            for ev in events:
                if not isinstance(ev, dict) or ev.get("paged_reminder"):
                    continue
                rem_epoch = ev.get("remind_at_epoch")
                if rem_epoch is not None:
                    try:
                        if now_epoch >= float(rem_epoch):
                            ev["paged_reminder"] = True
                            ev["status"] = "Paged (Loudspeaker)"
                            changed = True
                            fired_events.append(ev)
                            continue
                    except (TypeError, ValueError):
                        pass
                else:
                    rem_hm = str(ev.get("reminder_time", "")).strip()
                    if rem_hm and rem_hm == now_hm:
                        ev["paged_reminder"] = True
                        ev["status"] = "Paged (5m Warning)"
                        changed = True
                        fired_events.append(ev)
            if changed:
                state_manager.set_calendar_events(events)
                active_id = state_manager.get_active_patient_id()
                st = state_manager.load_state(active_id)
                cp = dict(st.get("custom_pane") or {})
                if cp.get("kind") == "timetable":
                    cp["generatedHtml"] = render_timetable_html_css(events)
                    state_manager.update_state({"custom_pane": cp}, patient_id=active_id, announce=False)
                for fev in fired_events:
                    _broadcast_hospital_loudspeaker_page(fev, is_test=False)
        except Exception:
            pass
        time.sleep(0.5)


@app.route("/api/calendar/test_page", methods=["POST", "OPTIONS"])
def api_calendar_test_page() -> Any:
    """Trigger an immediate Hospital Loudspeaker Page over the laptop speaker for a given event (or the next scheduled task)."""
    if request.method == "OPTIONS":
        return ("", 204)

    payload = request.get_json(silent=True) or {}
    event_id = str(payload.get("event_id", "")).strip()
    events = state_manager.get_calendar_events()
    if not events:
        events = build_timetable_events("create a time table with a 5pm surgery")
        state_manager.set_calendar_events(events)

    target_ev = None
    if event_id:
        for ev in events:
            if str(ev.get("id", "")) == event_id:
                target_ev = ev
                break
    if target_ev is None and events:
        target_ev = events[0]

    alert = _broadcast_hospital_loudspeaker_page(target_ev or {}, is_test=True)
    active_id = state_manager.get_active_patient_id()
    updated = state_manager.load_state(active_id)
    return jsonify(
        {
            "ok": True,
            "pageAlert": alert,
            "state": updated.get("ui_mode", "patient_stats"),
            "data": _build_bridge_data(updated),
        }
    )


def _extract_export_table_data(kind_override: str = "") -> tuple[str, str, list[str], list[list[str]]]:
    """Build structured (title, subtitle, headers, rows) for the active pane view so .xlsx and .png exports are clean."""
    active_id = state_manager.get_active_patient_id()
    state = state_manager.load_state(active_id)
    custom_pane = state.get("custom_pane") or {}
    kind = (kind_override or str(custom_pane.get("kind") or "patient")).strip().lower()
    now_str = datetime.now().strftime("%Y-%m-%d %I:%M:%S %p")

    if kind == "timetable":
        events = state_manager.get_calendar_events()
        if not events:
            events = build_timetable_events("create a time table with a 5pm surgery")
        title = "CLINICIAN SHIFT TIMETABLE & REMINDER SCHEDULE"
        subtitle = f"Synced to PC System Clock • Default Reminder: 5 Minutes Prior via Hospital Loudspeaker • {now_str}"
        headers = [
            "Time Slot",
            "Category",
            "Scheduled Task / Clinical Activity",
            "Location / Unit",
            "Loudspeaker Reminder (T-5 Min)",
            "Status",
        ]
        rows = [
            [
                str(ev.get("display_time", "")),
                str(ev.get("category", "Task")),
                str(ev.get("title", "")),
                str(ev.get("location", "")),
                str(ev.get("reminder_display", "5m prior")),
                str(ev.get("status", "Scheduled")),
            ]
            for ev in events
        ]
        return title, subtitle, headers, rows

    hud = build_ward_hud_data()
    patients = hud.get("patients") or []
    stats = hud.get("stats") or {}

    if kind == "ward_stats":
        title = "WARD TELEMETRY & ACUITY STATISTICS SUMMARY"
        subtitle = f"{stats.get('summaryLine', '')} • Generated {now_str}"
        headers = [
            "Patient ID",
            "Patient Name",
            "Location",
            "Department & Protocol",
            "Live Heart Rate",
            "Baseline HR",
            "Wakefulness & Rhythm",
            "Acuity Status",
        ]
        rows = [
            [
                f"#{p['patientId']}",
                str(p["name"]),
                f"{p['wing']}, Room {p['room']}",
                f"{p['department']} ({p['protocol']})",
                f"{p['bpm']} BPM",
                f"{p['baselineBpm']} BPM",
                f"{p['wakeState']} • {p['rhythm']}",
                str(p["acuity"]),
            ]
            for p in patients
        ]
        rows.append(
            [
                "SUMMARY",
                f"Total: {stats.get('totalPatients', 4)} Patients",
                "All Wings (A, B, C)",
                "Aggregate Ward Telemetry",
                f"Mean: {stats.get('meanBpm', 83.5)} BPM",
                f"Range: {stats.get('minBpm', 64)}–{stats.get('maxBpm', 104)} BPM",
                f"{stats.get('awakeCount', 3)} Awake / {stats.get('sleepingCount', 1)} Sleeping",
                f"{stats.get('tachycardiaCount', 1)} Elevated HR",
            ]
        )
        return title, subtitle, headers, rows

    if kind in {"ward_table", "custom_html"}:
        title = "ACTIVE WARD ROSTER — ALL MONITORED PATIENTS"
        subtitle = f"Real-Time Multi-Patient EHR & Telemetry Matrix • Generated {now_str}"
        headers = [
            "Patient ID",
            "Patient Name",
            "Wing & Room",
            "Department & Protocol",
            "Live Telemetry",
            "Reported Symptoms",
            "Nursing Round Notes",
        ]
        rows = [
            [
                f"#{p['patientId']}",
                str(p["name"]),
                f"{p['wing']}, Room {p['room']}",
                f"{p['department']} — {p['protocol']}",
                f"{p['bpm']} BPM ({p['wakeState']} • {p['rhythm']})",
                str(p["symptom"]),
                str(p["nurseNote"]),
            ]
            for p in patients
        ]
        return title, subtitle, headers, rows

    # Single Patient Clinical Summary
    panel = state.get("patient_panel") or rag_agent.synthesize_patient_panel(state)
    title = f"CLINICAL PATIENT SUMMARY — {str(panel.get('name', 'Mika Hämäläinen')).upper()} (#{panel.get('patientId', '1429')})"
    subtitle = f"{panel.get('wing', 'Wing A')} • Room {panel.get('room', '153')} • Generated {now_str}"
    headers = ["Clinical Field", "Recorded Value / Live Telemetry Status"]
    rows = [
        ["Patient Name & ID", f"{panel.get('name', '')} (Patient #{panel.get('patientId', '')})"],
        ["Location", f"{panel.get('wing', '')}, Room {panel.get('room', '')}"],
        ["Department", str(panel.get("department", ""))],
        ["Clinical Protocol", str(panel.get("protocol", ""))],
        ["Live Heart Rate & Rhythm", f"{state.get('current_bpm', 90)} BPM — {panel.get('rhythm', '')}"],
        ["Telemetry Signal Detail", str(panel.get("telemetryDetail", ""))],
        ["Previous Nursing Rounds", str(panel.get("nurseRounds", ""))],
        ["Patient-Reported Symptoms", str(panel.get("symptom", ""))],
    ]
    return title, subtitle, headers, rows


@app.route("/api/export/xlsx", methods=["GET"])
def api_export_xlsx() -> Any:
    """Export the active table/timetable/statistics/summary as a formatted .xlsx file on a plain white background."""
    if not _OPENPYXL_AVAILABLE or openpyxl is None:
        return jsonify({"ok": False, "error": "openpyxl is not installed."}), 500

    kind_arg = str(request.args.get("kind", "")).strip()
    title, subtitle, headers, rows = _extract_export_table_data(kind_arg)

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Malcolm Clinical Report"
    ws.views.sheetView[0].showGridLines = True

    white_fill = PatternFill("solid", fgColor="FFFFFF")
    header_fill = PatternFill("solid", fgColor="F1F5F9")
    thin_side = Side(border_style="thin", color="CBD5E1")
    header_bottom = Side(border_style="medium", color="64748B")
    cell_border = Border(left=thin_side, right=thin_side, top=thin_side, bottom=thin_side)
    header_border = Border(left=thin_side, right=thin_side, top=thin_side, bottom=header_bottom)

    # Paint a clean plain white background across the sheet canvas
    max_cols = max(len(headers), 6)
    max_rows = len(rows) + 10
    for r_idx in range(1, max_rows + 1):
        for c_idx in range(1, max_cols + 3):
            ws.cell(row=r_idx, column=c_idx).fill = white_fill

    # Title & Subtitle block
    ws.merge_cells(start_row=2, start_column=1, end_row=2, end_column=len(headers))
    title_cell = ws.cell(row=2, column=1, value=title)
    title_cell.font = Font(name="Calibri", size=15, bold=True, color="0F172A")
    title_cell.alignment = Alignment(vertical="center")
    ws.row_dimensions[2].height = 26

    ws.merge_cells(start_row=3, start_column=1, end_row=3, end_column=len(headers))
    sub_cell = ws.cell(row=3, column=1, value=subtitle)
    sub_cell.font = Font(name="Calibri", size=10, italic=True, color="475569")
    sub_cell.alignment = Alignment(vertical="center")
    ws.row_dimensions[3].height = 20

    # Header Row (Row 5)
    header_row_idx = 5
    ws.row_dimensions[header_row_idx].height = 28
    for c_idx, h_text in enumerate(headers, start=1):
        c = ws.cell(row=header_row_idx, column=c_idx, value=h_text)
        c.font = Font(name="Calibri", size=11, bold=True, color="0F172A")
        c.fill = header_fill
        c.border = header_border
        c.alignment = Alignment(horizontal="left", vertical="center", wrap_text=True)

    # Data Rows (Row 6+)
    for r_offset, row_vals in enumerate(rows):
        r_idx = header_row_idx + 1 + r_offset
        ws.row_dimensions[r_idx].height = 28
        for c_idx, val in enumerate(row_vals, start=1):
            c = ws.cell(row=r_idx, column=c_idx, value=val)
            c.font = Font(name="Calibri", size=10.5, bold=(c_idx == 1), color="1E293B")
            c.fill = white_fill
            c.border = cell_border
            c.alignment = Alignment(horizontal="left", vertical="center", wrap_text=True)

    # Auto-adjust column widths cleanly
    for c_idx in range(1, len(headers) + 1):
        col_letter = get_column_letter(c_idx)
        max_len = len(str(headers[c_idx - 1]))
        for row_vals in rows:
            if c_idx - 1 < len(row_vals):
                max_len = max(max_len, len(str(row_vals[c_idx - 1])))
        ws.column_dimensions[col_letter].width = max(16, min(44, max_len + 5))

    buf = io.BytesIO()
    wb.save(buf)
    buf.seek(0)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return send_file(
        buf,
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        as_attachment=True,
        download_name=f"malcolm_report_{stamp}.xlsx",
    )


@app.route("/api/export/png", methods=["GET"])
def api_export_png() -> Any:
    """Export the active table/timetable/statistics/summary as a high-res .png image on a plain white background."""
    if not _PIL_AVAILABLE or Image is None or ImageDraw is None or ImageFont is None:
        return jsonify({"ok": False, "error": "Pillow is not installed."}), 500

    kind_arg = str(request.args.get("kind", "")).strip()
    title, subtitle, headers, rows = _extract_export_table_data(kind_arg)

    def _load_font(names: tuple[str, ...], size: int) -> Any:
        for name in names:
            for candidate in (name, os.path.join("C:\\Windows\\Fonts", name)):
                try:
                    return ImageFont.truetype(candidate, size)
                except Exception:
                    continue
        return ImageFont.load_default()

    font_title = _load_font(("segoeuib.ttf", "arialbd.ttf"), 28)
    font_sub = _load_font(("segoeui.ttf", "arial.ttf"), 16)
    font_th = _load_font(("segoeuib.ttf", "arialbd.ttf"), 16)
    font_td = _load_font(("segoeui.ttf", "arial.ttf"), 15)
    font_td_bold = _load_font(("segoeuib.ttf", "arialbd.ttf"), 15)

    img_w = 1800
    pad_x = 48
    pad_y = 44
    usable_w = img_w - 2 * pad_x
    num_cols = max(1, len(headers))

    # Proportional column weights based on content length
    col_weights: list[float] = []
    for c_idx in range(num_cols):
        avg_len = max(
            len(str(headers[c_idx])),
            max((len(str(r[c_idx])) for r in rows if c_idx < len(r)), default=10),
        )
        col_weights.append(max(12.0, min(42.0, float(avg_len))))
    weight_sum = sum(col_weights) or 1.0
    col_widths = [int(usable_w * (w / weight_sum)) for w in col_weights]
    col_widths[-1] += usable_w - sum(col_widths)

    # Wrap cell text per column and calculate row heights
    wrapped_rows: list[list[list[str]]] = []
    row_heights: list[int] = []
    for r in rows:
        wrapped_cells: list[list[str]] = []
        max_lines = 1
        for c_idx in range(num_cols):
            val = str(r[c_idx]) if c_idx < len(r) else ""
            char_cap = max(14, int((col_widths[c_idx] - 24) / 8.2))
            lines = textwrap.wrap(val, width=char_cap) or [""]
            wrapped_cells.append(lines)
            max_lines = max(max_lines, len(lines))
        wrapped_rows.append(wrapped_cells)
        row_heights.append(max(48, max_lines * 22 + 24))

    header_h = 52
    top_block_h = 104
    footer_h = 50
    img_h = pad_y * 2 + top_block_h + header_h + sum(row_heights) + footer_h

    # Create pure Plain White (#FFFFFF) canvas
    img = Image.new("RGB", (img_w, img_h), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Title & Subtitle
    draw.text((pad_x, pad_y), title, fill=(15, 23, 42), font=font_title)
    draw.text((pad_x, pad_y + 42), subtitle, fill=(71, 85, 105), font=font_sub)

    # Table Header
    table_top = pad_y + top_block_h
    draw.rectangle(
        [pad_x, table_top, pad_x + usable_w, table_top + header_h],
        fill=(241, 245, 249),
        outline=(148, 163, 184),
        width=1,
    )
    curr_x = pad_x
    for c_idx, h_text in enumerate(headers):
        cw = col_widths[c_idx]
        draw.line([(curr_x, table_top), (curr_x, table_top + header_h)], fill=(203, 213, 225), width=1)
        draw.text((curr_x + 12, table_top + 15), h_text, fill=(15, 23, 42), font=font_th)
        curr_x += cw

    # Table Data Rows on Plain White Background
    curr_y = table_top + header_h
    for r_idx, wrapped_cells in enumerate(wrapped_rows):
        rh = row_heights[r_idx]
        draw.rectangle(
            [pad_x, curr_y, pad_x + usable_w, curr_y + rh],
            fill=(255, 255, 255),
            outline=(203, 213, 225),
            width=1,
        )
        curr_x = pad_x
        for c_idx, lines in enumerate(wrapped_cells):
            cw = col_widths[c_idx]
            if c_idx > 0:
                draw.line([(curr_x, curr_y), (curr_x, curr_y + rh)], fill=(203, 213, 225), width=1)
            f_use = font_td_bold if c_idx == 0 else font_td
            for l_idx, line_str in enumerate(lines):
                draw.text(
                    (curr_x + 12, curr_y + 13 + l_idx * 22),
                    line_str,
                    fill=(30, 41, 59),
                    font=f_use,
                )
            curr_x += cw
        curr_y += rh

    # Footer stamp
    draw.text(
        (pad_x, curr_y + 16),
        "Malcolm Clinical Triage Copilot • Formatted Plain-White Export",
        fill=(100, 116, 139),
        font=font_sub,
    )

    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    buf.seek(0)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return send_file(
        buf,
        mimetype="image/png",
        as_attachment=True,
        download_name=f"malcolm_report_{stamp}.png",
    )


def _handle_terminal_command(raw_cmd: str) -> str:
    """Process an interactive terminal command to enable, disable, or inspect winsound."""
    cmd = raw_cmd.strip().lower()
    if not cmd:
        return ""
    if cmd in {"winsound on", "sound on", "enable", "on", "unmute"}:
        set_winsound_enabled(True)
        return "[Terminal Audio] winsound ENABLED (terminal loudspeaker alerts active)"
    if cmd in {"winsound off", "sound off", "disable", "off", "mute"}:
        set_winsound_enabled(False)
        return "[Terminal Audio] winsound DISABLED (alerts play on web portals only)"
    if cmd in {"winsound", "w", "toggle", "sound"}:
        new_state = set_winsound_enabled(not _winsound_enabled)
        label = "ENABLED" if new_state else "DISABLED"
        return f"[Terminal Audio] winsound {label}"
    if cmd in {"status", "help", "?"}:
        label = "ENABLED" if _winsound_enabled else "DISABLED"
        return (
            f"[Terminal Audio] winsound is {label}. "
            "Commands: 'winsound on' | 'winsound off' | 'w' (toggle) | 'status'"
        )
    return f"[Terminal] Unknown command '{cmd}'. Type 'winsound on', 'winsound off', 'w', or 'status'."


def _terminal_command_loop() -> None:
    """Listen for interactive terminal commands without blocking Flask or exiting on EOF."""
    while True:
        try:
            line = sys.stdin.readline()
            if not line:
                time.sleep(0.5)
                continue
            msg = _handle_terminal_command(line)
            if msg:
                print(msg, flush=True)
        except Exception:
            time.sleep(0.5)


if __name__ == "__main__":
    threading.Thread(target=_prewarm_tts_cache, daemon=True).start()
    _ensure_pager_daemon_running()
    threading.Thread(target=_terminal_command_loop, daemon=True).start()
    lan_ip = get_lan_ip()
    ws_status = "ENABLED" if _winsound_enabled else "DISABLED"
    print("============================================================")
    print("  Malcolm Clinician API + Bedside Web UI")
    print(f"  Local API / Bedside:  http://127.0.0.1:5000")
    print(f"  Phone Bedside LAN:    http://{lan_ip}:5000")
    print(f"  Terminal winsound:    {ws_status}")
    print("  Type 'winsound off', 'winsound on', 'w' (toggle), or 'status'")
    print("============================================================")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
