"""Bedside Web UI and Clinician React UI API server for Malcolm."""

from __future__ import annotations

import asyncio
from collections import deque
import ctypes
import io
import json
import os
import re
import socket
import tempfile
import threading
import time
import wave
from typing import Any, Callable

import numpy as np
from dotenv import load_dotenv
from flask import Flask, Response, jsonify, render_template, request

from local_rag import build_ward_hud_data, rag_agent
from state_manager import state_manager
from telemetry_provider import get_telemetry_provider

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
    """Build the { bpm, transcript, clinicianTranscript, patient, customPane, wardHud } payload expected by the React UI."""
    panel = state.get("patient_panel")
    if not isinstance(panel, dict) or not panel:
        panel = rag_agent.synthesize_patient_panel(state)
    custom_pane = state.get("custom_pane")
    if not isinstance(custom_pane, dict) or not custom_pane:
        custom_pane = {"kind": "patient", "title": "", "subtitle": "", "filterWing": ""}
    return {
        "bpm": state.get("current_bpm", 90),
        "transcript": state.get("last_transcript", "Listening for clinician request..."),
        "clinicianTranscript": state.get("last_clinician_input", ""),
        "patient": panel,
        "customPane": custom_pane,
        "wardHud": build_ward_hud_data(),
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
                    and (now - self._record_start_time >= 0.75)
                    and (
                        (endpoint_hit and silence_elapsed > 0.65)
                        or silence_elapsed > 0.76
                        or (stable_ticks >= 9 and silence_elapsed > 0.60)
                    )
                ):
                    should_finalize = True
                    self._stop_event.set()
                    break
            else:
                if (
                    self.speech_detected
                    and (now - self._record_start_time >= 0.75)
                    and silence_elapsed > 0.72
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


def _finalize_stt_response(wav_path: str) -> Any:
    """Transcribe a temporary WAV file, advance Malcolm triage state, and return JSON."""
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

    response_payload = _run_clinician_message(transcript)
    response_payload["transcript"] = str(response_payload.get("transcript") or transcript)
    return jsonify(response_payload)


@app.route("/api/stt", methods=["POST", "OPTIONS"])
def api_stt() -> Any:
    """Transcribe uploaded WAV audio and advance clinician workflow."""
    if request.method == "OPTIONS":
        return ("", 204)

    audio_file = request.files.get("audio")
    if audio_file is None:
        return jsonify({"ok": False, "error": "Missing 'audio' file upload."}), 400

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
    audio_file.save(tmp_path)
    return _finalize_stt_response(tmp_path)


@app.route("/api/mic/start", methods=["POST", "OPTIONS"])
def api_mic_start() -> Any:
    """Start capturing clinician speech from all active physical Windows microphones."""
    if request.method == "OPTIONS":
        return ("", 204)

    try:
        device_name = _host_recorder.start(on_final=_run_clinician_message)
        status = _host_recorder.get_live_status()
        return jsonify({"ok": True, "device": device_name, "devices": status.get("devices", [])})
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
        response_payload = _run_clinician_message(cached_partial)
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

    response_payload = _run_clinician_message(final_transcript)
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

    result = rag_agent.process_bedside_turn(message)
    status_code = 200 if result.get("ok") else 409
    return jsonify(result), status_code


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
            "custom_pane": {"kind": "patient", "title": "", "subtitle": "", "filterWing": ""},
            "patient_panel": panel,
        }
    elif action == "ward_table":
        updates = {
            "ui_mode": "patient_stats",
            "custom_pane": {
                "kind": "ward_table",
                "title": "Active Ward Roster — All Monitored Patients",
                "subtitle": "Real-Time Multi-Patient EHR & Telemetry Matrix",
                "filterWing": str(payload.get("filterWing", "")),
            },
            "patient_panel": panel,
        }
    elif action == "ward_stats":
        updates = {
            "ui_mode": "patient_stats",
            "custom_pane": {
                "kind": "ward_stats",
                "title": "Ward Telemetry & Acuity Statistics",
                "subtitle": "Live Statistical Breakdown • 4 Monitored Patients",
                "filterWing": "",
            },
            "patient_panel": panel,
        }
    else:
        updates = {
            "ui_mode": "patient_stats",
            "custom_pane": {"kind": "patient", "title": "", "subtitle": "", "filterWing": ""},
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


if __name__ == "__main__":
    threading.Thread(target=_prewarm_tts_cache, daemon=True).start()
    lan_ip = get_lan_ip()
    print("============================================================")
    print("  Malcolm Clinician API + Bedside Web UI")
    print(f"  Local API / Bedside:  http://127.0.0.1:5000")
    print(f"  Phone Bedside LAN:    http://{lan_ip}:5000")
    print("============================================================")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
