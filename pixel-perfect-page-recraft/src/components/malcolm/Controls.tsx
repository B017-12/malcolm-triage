import { useCallback, useEffect, useRef, useState, type FormEvent } from "react";
import {
  ExternalLink,
  Mic,
  MicOff,
  RotateCcw,
  Send,
  SlidersHorizontal,
  Volume2,
  VolumeX,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { DEFAULT_CAPTIONS, malcolmStore, useMalcolm } from "@/lib/malcolm/store";
import type { MalcolmMode } from "@/lib/malcolm/signal";

interface SpeechRecognitionResultItem {
  transcript: string;
}

interface SpeechRecognitionResult {
  isFinal: boolean;
  0: SpeechRecognitionResultItem;
}

interface SpeechRecognitionEventLike {
  resultIndex: number;
  results: ArrayLike<SpeechRecognitionResult>;
}

interface SpeechRecognitionInstance {
  continuous: boolean;
  interimResults: boolean;
  lang: string;
  onresult: ((event: SpeechRecognitionEventLike) => void) | null;
  onerror: ((event: { error?: string }) => void) | null;
  onend: (() => void) | null;
  start: () => void;
  stop: () => void;
}

type SpeechRecognitionCtor = new () => SpeechRecognitionInstance;

interface WebAISession {
  prompt: (input: string) => Promise<string>;
  destroy?: () => void;
}

interface WebAILanguageModelAPI {
  availability?: () => Promise<string>;
  capabilities?: () => Promise<{ available?: string }>;
  create?: (opts?: { systemPrompt?: string }) => Promise<WebAISession>;
}

function getWebAILanguageModel(): WebAILanguageModelAPI | null {
  if (typeof window === "undefined") return null;
  const w = window as unknown as {
    LanguageModel?: WebAILanguageModelAPI;
    ai?: { languageModel?: WebAILanguageModelAPI };
  };
  return w.LanguageModel || w.ai?.languageModel || null;
}

/** Invoke Chrome Built-in WebAI (Gemini Nano) when available in the browser. */
export async function tryGoogleWebAI(prompt: string, systemPrompt?: string): Promise<string | null> {
  const api = getWebAILanguageModel();
  if (!api?.create) return null;
  try {
    const status = api.availability
      ? await api.availability()
      : (await api.capabilities?.())?.available;
    if (status === "unavailable" || status === "no") return null;
    const session = await api.create(systemPrompt ? { systemPrompt } : undefined);
    try {
      return (await session.prompt(prompt)).trim() || null;
    } finally {
      session.destroy?.();
    }
  } catch {
    return null;
  }
}

function getSpeechRecognitionCtor(): SpeechRecognitionCtor | null {
  if (typeof window === "undefined") return null;
  const w = window as unknown as {
    SpeechRecognition?: SpeechRecognitionCtor;
    webkitSpeechRecognition?: SpeechRecognitionCtor;
  };
  return w.SpeechRecognition || w.webkitSpeechRecognition || null;
}

function getApiBase(): string {
  if (typeof window === "undefined") return "http://localhost:5000";
  const pollParam = new URLSearchParams(window.location.search).get("poll");
  if (pollParam) {
    try {
      return new URL(pollParam).origin;
    } catch {
      // Fallback below
    }
  }
  const host = window.location.hostname || "localhost";
  return `http://${host}:5000`;
}

function encodeWavBlob(chunks: Float32Array[], sampleRate: number): Blob {
  let totalLength = 0;
  for (const chunk of chunks) totalLength += chunk.length;
  const buffer = new ArrayBuffer(44 + totalLength * 2);
  const view = new DataView(buffer);

  const writeStr = (offset: number, str: string) => {
    for (let i = 0; i < str.length; i++) {
      view.setUint8(offset + i, str.charCodeAt(i));
    }
  };

  writeStr(0, "RIFF");
  view.setUint32(4, 36 + totalLength * 2, true);
  writeStr(8, "WAVE");
  writeStr(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true); // PCM
  view.setUint16(22, 1, true); // mono
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeStr(36, "data");
  view.setUint32(40, totalLength * 2, true);

  let offset = 44;
  for (const chunk of chunks) {
    for (let i = 0; i < chunk.length; i++) {
      const clamped = Math.max(-1, Math.min(1, chunk[i] || 0));
      view.setInt16(offset, clamped < 0 ? clamped * 0x8000 : clamped * 0x7fff, true);
      offset += 2;
    }
  }

  return new Blob([buffer], { type: "audio/wav" });
}

function cleanTextForSpeech(text: string): string {
  return text
    .replace(/(\d+)\s*[–—-]\s*(\d+)/g, "$1 to $2")
    .replace(/\(([^)]+)\)/g, (_m, inner: string) => ` ${inner.replace(/,/g, " ")} `)
    .replace(/[—–]/g, " ")
    .replace(/;/g, ",")
    .replace(/:/g, " ")
    .replace(/\b(Wing\s+[A-Z0-9]+),\s*(Room\s+\d+)\b/gi, "$1 $2")
    .replace(/\b(Patient\s+\d+),\s*/gi, "$1 ")
    .replace(/\s+,/g, ",")
    .replace(/\s+/g, " ")
    .trim();
}

function pickJarvisBrowserVoice(): SpeechSynthesisVoice | null {
  if (typeof window === "undefined" || !("speechSynthesis" in window)) return null;
  const voices = window.speechSynthesis.getVoices();
  if (!voices.length) return null;
  return (
    voices.find((v) => /Ryan|Natural.*UK|Google UK English Male|Daniel|George/i.test(v.name)) ||
    voices.find((v) => /^en[-_]GB/i.test(v.lang)) ||
    voices.find((v) => /^en/i.test(v.lang)) ||
    null
  );
}

/** Zero-latency (0ms) optimistic split-screen pane controller before backend round-trip completes. */
function applyOptimisticPaneIntent(utterance: string) {
  const lowered = utterance.toLowerCase().trim();
  if (!lowered) return;

  // 1. Close / hide / dismiss pane
  if (
    /\b(close|hide|dismiss|remove|minimize|collapse|put away|shut)\b.*\b(pane|panel|table|chart|stats|statistics|display|screen|board|roster|view|it|that)\b/.test(
      lowered,
    ) ||
    /^(close|hide|dismiss)\s*(pane|table|stats|display|it|that)?[.!?]*$/.test(lowered)
  ) {
    malcolmStore.setMode("idle");
    malcolmStore.setCustomPane({ kind: "patient", title: "", subtitle: "", filterWing: "" });
    return;
  }

  // 2. Statistics / analytics / metrics pane
  if (
    /\b(statistics|statistic|ward stats|vitals stats|telemetry stats|analytics|metrics|average heart rate|mean heart rate|compare vitals|compare heart)\b/.test(
      lowered,
    )
  ) {
    malcolmStore.setCustomPane({
      kind: "ward_stats",
      title: "Ward Telemetry & Acuity Statistics",
      subtitle: "Live Statistical Breakdown • 4 Monitored Patients",
      filterWing: "",
    });
    malcolmStore.setMode("patient");
    return;
  }

  // 3. Ward Table / All-patients roster
  if (
    /\b(table|roster|spreadsheet|grid|board|matrix)\b/.test(lowered) ||
    (/\b(show|display|pull up|bring up|open|create|construct|build|make|give me)\b/.test(lowered) &&
      /\b(all patients|all patience|every patient|the ward|our patients|four patients|all four|patient list)\b/.test(
        lowered,
      ))
  ) {
    malcolmStore.setCustomPane({
      kind: "ward_table",
      title: "Active Ward Roster — All Monitored Patients",
      subtitle: "Real-Time Multi-Patient EHR & Telemetry Matrix",
      filterWing: "",
    });
    malcolmStore.setMode("patient");
  }
}

/** Clean dual-line live voice readout: Doctor STT transcript above Malcolm's spoken reply. */
export function Caption() {
  const { caption, clinicianCaption } = useMalcolm();
  const [text, setText] = useState(caption);
  useEffect(() => {
    if (caption) setText(caption);
  }, [caption]);

  return (
    <div className="pointer-events-none relative z-20 flex w-full max-w-2xl shrink-0 flex-col items-center gap-2">
      {clinicianCaption ? (
        <div
          aria-live="polite"
          className="ledger-glass flex max-w-full items-center gap-2 border border-frame px-3 py-1 shadow-lg"
        >
          <span className="shrink-0 font-mono text-[9px] uppercase tracking-[0.18em] text-primary">
            DOCTOR STT
          </span>
          <span className="text-balance font-sans text-xs text-text-bright">
            &ldquo;{clinicianCaption}&rdquo;
          </span>
        </div>
      ) : null}
      <p
        aria-live="polite"
        key={text}
        className={cn(
          "ledger-glass pointer-events-auto max-h-[18vh] w-full max-w-[min(92%,34rem)] overflow-y-auto border border-frame/50 px-3 py-1.5 text-center font-sans text-[clamp(0.85rem,1.2vw,1rem)] leading-snug tracking-tight text-foreground/95 shadow-xl transition-opacity duration-500",
          caption ? "animate-in fade-in slide-in-from-bottom-1 opacity-100" : "opacity-0",
        )}
      >
        {text}
      </p>
    </div>
  );
}

const OPTIONS: { id: MalcolmMode; label: string; key: string }[] = [
  { id: "idle", label: "Idle", key: "1" },
  { id: "heart", label: "Heart rate", key: "2" },
  { id: "patient", label: "Patient stats", key: "3" },
];

function selectMode(mode: MalcolmMode) {
  malcolmStore.setMode(mode, mode === "idle" ? undefined : 90);
  malcolmStore.setCaption(DEFAULT_CAPTIONS[mode]);
}

/** Keys 1 / 2 / 3 switch Idle / Heart rate / Patient stats. */
export function useModeHotkeys() {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.metaKey || e.ctrlKey || e.altKey || e.repeat) return;
      const el = e.target as HTMLElement | null;
      if (el && (el.isContentEditable || /^(INPUT|TEXTAREA|SELECT)$/.test(el.tagName))) return;
      const opt = OPTIONS.find((o) => o.key === e.key);
      if (opt) selectMode(opt.id);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);
}

/** Three tiny square test indicators, bottom-right. */
export function MicroDots() {
  const { mode } = useMalcolm();
  return (
    <div role="radiogroup" aria-label="Demo state" className="fixed bottom-3 right-3 z-30 flex gap-1.5">
      {OPTIONS.map((o) => {
        const on = mode === o.id;
        return (
          <button
            key={o.id}
            role="radio"
            aria-checked={on}
            aria-label={`${o.label} (key ${o.key})`}
            title={`${o.label} — press ${o.key}`}
            onClick={() => selectMode(o.id)}
            className={cn(
              "size-2 border border-foreground opacity-25 transition-opacity hover:opacity-100 focus-visible:opacity-100 focus-visible:outline-none",
              on && "bg-foreground",
            )}
          />
        );
      })}
    </div>
  );
}

/**
 * Hands-free Jarvis-style Neural TTS + hybrid Web Speech / Whisper STT dock
 * with a collapsible testing console.
 */
export function VoiceDock() {
  const { caption, patient, mode } = useMalcolm();
  const isSplit = mode === "patient";
  const [micActive, setMicActive] = useState(false);
  const [ttsEnabled, setTtsEnabled] = useState(true);
  const [testOpen, setTestOpen] = useState(false);
  const [busy, setBusy] = useState(false);
  const [manualInput, setManualInput] = useState("");
  const [speechSupported, setSpeechSupported] = useState(true);
  const [webAiReady, setWebAiReady] = useState(false);
  const [statusNote, setStatusNote] = useState("");

  const recognitionRef = useRef<SpeechRecognitionInstance | null>(null);
  const micActiveRef = useRef(false);
  const hostMicRecordingRef = useRef(false);
  const hostSpeechDetectedRef = useRef(false);
  const livePollRef = useRef<number | null>(null);
  const liveSseRef = useRef<EventSource | null>(null);
  const sessionTranscriptRef = useRef<string>("");
  const browserTranscriptRef = useRef<string>("");
  const isSpeakingRef = useRef(false);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const audioUrlRef = useRef<string | null>(null);
  const ttsAbortRef = useRef<AbortController | null>(null);
  const lastSpokenCaptionRef = useRef<string>(DEFAULT_CAPTIONS.idle);

  // Fallback WAV recorder refs when host + browser SpeechRecognition are unavailable
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const audioCtxRef = useRef<AudioContext | null>(null);
  const pcmChunksRef = useRef<Float32Array[]>([]);
  const recorderSampleRateRef = useRef<number>(16000);

  const apiBase = getApiBase();

  useEffect(() => {
    setSpeechSupported(getSpeechRecognitionCtor() !== null);
    const api = getWebAILanguageModel();
    if (!api) return;
    void (async () => {
      try {
        const status = api.availability
          ? await api.availability()
          : (await api.capabilities?.())?.available;
        setWebAiReady(Boolean(status && status !== "unavailable" && status !== "no"));
      } catch {
        setWebAiReady(false);
      }
    })();
  }, []);

  const stopAllSpeech = useCallback(() => {
    isSpeakingRef.current = false;
    ttsAbortRef.current?.abort();
    ttsAbortRef.current = null;

    if (audioRef.current) {
      audioRef.current.onended = null;
      audioRef.current.onerror = null;
      audioRef.current.pause();
      audioRef.current.src = "";
      audioRef.current = null;
    }
    if (audioUrlRef.current) {
      URL.revokeObjectURL(audioUrlRef.current);
      audioUrlRef.current = null;
    }
    if (typeof window !== "undefined" && "speechSynthesis" in window) {
      window.speechSynthesis.cancel();
    }
  }, []);

  const speakWithBrowserFallback = useCallback((text: string) => {
    if (typeof window === "undefined" || !("speechSynthesis" in window)) return;
    try {
      const cleaned = cleanTextForSpeech(text);
      if (!cleaned) return;
      window.speechSynthesis.cancel();
      const utterance = new SpeechSynthesisUtterance(cleaned);
      const voice = pickJarvisBrowserVoice();
      if (voice) utterance.voice = voice;
      utterance.rate = 1.08;
      utterance.pitch = 0.98;
      isSpeakingRef.current = true;
      utterance.onend = () => {
        isSpeakingRef.current = false;
      };
      utterance.onerror = () => {
        isSpeakingRef.current = false;
      };
      window.speechSynthesis.speak(utterance);
    } catch {
      isSpeakingRef.current = false;
    }
  }, []);

  // Pipelined sentence-by-sentence Jarvis Neural TTS (parallel fetch, zero-wait first clause playback)
  useEffect(() => {
    if (!ttsEnabled || typeof window === "undefined") return;
    const trimmed = caption.trim();
    if (!trimmed || trimmed === lastSpokenCaptionRef.current) return;
    lastSpokenCaptionRef.current = trimmed;

    stopAllSpeech();
    const controller = new AbortController();
    ttsAbortRef.current = controller;

    const rawSentences = trimmed
      .split(/(?<=[.!?])\s+/)
      .map((s) => s.trim())
      .filter(Boolean);
    const sentences = rawSentences.length > 0 ? rawSentences : [trimmed];

    void (async () => {
      try {
        const blobPromises = sentences.map((sentence) =>
          fetch(`${apiBase}/api/tts`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text: sentence }),
            signal: controller.signal,
          }).then(async (res) => {
            if (!res.ok) throw new Error("Neural TTS unavailable");
            return res.blob();
          }),
        );

        for (let i = 0; i < blobPromises.length; i++) {
          const promise = blobPromises[i];
          if (!promise || controller.signal.aborted) return;
          const blob = await promise;
          if (!blob || controller.signal.aborted || blob.size === 0) return;

          await new Promise<void>((resolve, reject) => {
            const url = URL.createObjectURL(blob);
            audioUrlRef.current = url;
            const audio = new Audio(url);
            audio.playbackRate = 1.0;
            audioRef.current = audio;
            isSpeakingRef.current = true;

            const cleanupUrl = () => {
              if (audioUrlRef.current === url) {
                URL.revokeObjectURL(url);
                audioUrlRef.current = null;
              }
            };

            audio.onended = () => {
              cleanupUrl();
              if (i === blobPromises.length - 1) {
                isSpeakingRef.current = false;
              }
              resolve();
            };
            audio.onerror = () => {
              cleanupUrl();
              isSpeakingRef.current = false;
              reject(new Error("Audio playback failed"));
            };
            audio.play().catch((playErr) => {
              cleanupUrl();
              isSpeakingRef.current = false;
              reject(playErr);
            });
          });
        }
      } catch (err) {
        if ((err as { name?: string })?.name === "AbortError" || controller.signal.aborted) return;
        speakWithBrowserFallback(trimmed);
      }
    })();
  }, [apiBase, caption, speakWithBrowserFallback, stopAllSpeech, ttsEnabled]);

  useEffect(() => stopAllSpeech, [stopAllSpeech]);

  const sendClinicianCommand = useCallback(
    async (message: string) => {
      const cleaned = message.trim();
      if (!cleaned) return;
      setBusy(true);
      setStatusNote("");
      malcolmStore.setClinicianCaption(cleaned);
      applyOptimisticPaneIntent(cleaned);
      try {
        stopAllSpeech();
        const res = await fetch(`${apiBase}/api/clinician`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ message: cleaned }),
        });
        const payload = await res.json();
        if (res.ok && window.malcolm) {
          window.malcolm.apply(payload);
        } else if (!res.ok) {
          setStatusNote(payload.error || "Command rejected");
        }
      } catch {
        setStatusNote("Backend offline on port 5000");
      } finally {
        setBusy(false);
      }
    },
    [apiBase, stopAllSpeech],
  );

  const selectPatient = useCallback(
    async (patientId: string) => {
      setBusy(true);
      setStatusNote("");
      try {
        stopAllSpeech();
        const res = await fetch(`${apiBase}/api/select_patient`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ patient_id: patientId }),
        });
        const payload = await res.json();
        if (res.ok && window.malcolm) {
          window.malcolm.apply(payload);
        } else if (!res.ok) {
          setStatusNote(payload.error || "Failed to select patient");
        }
      } catch {
        setStatusNote("Backend offline on port 5000");
      } finally {
        setBusy(false);
      }
    },
    [apiBase, stopAllSpeech],
  );

  const sendBedsideTestCommand = useCallback(
    async (message: string) => {
      const cleaned = message.trim();
      if (!cleaned) return;
      setBusy(true);
      setStatusNote("");
      try {
        const res = await fetch(`${apiBase}/api/message`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ message: cleaned }),
        });
        const payload = await res.json();
        if (!res.ok) {
          setStatusNote(payload.error || "Bedside check-in not active yet");
        } else {
          const stateRes = await fetch(`${apiBase}/api/state`, { cache: "no-store" });
          if (stateRes.ok && window.malcolm) {
            window.malcolm.apply(await stateRes.json());
          }
        }
      } catch {
        setStatusNote("Backend offline on port 5000");
      } finally {
        setBusy(false);
      }
    },
    [apiBase],
  );

  const resetDemo = useCallback(async () => {
    setBusy(true);
    setStatusNote("");
    sessionTranscriptRef.current = "";
    malcolmStore.setClinicianCaption("");
    try {
      stopAllSpeech();
      lastSpokenCaptionRef.current = DEFAULT_CAPTIONS.idle;
      const res = await fetch(`${apiBase}/api/reset`, { method: "POST" });
      if (res.ok) {
        const payload = await res.json();
        window.malcolm?.apply(payload);
      }
    } catch {
      setStatusNote("Backend offline on port 5000");
    } finally {
      setBusy(false);
    }
  }, [apiBase, stopAllSpeech]);

  const stopWavFallbackAndTranscribe = useCallback(async () => {
    mediaStreamRef.current?.getTracks().forEach((t) => t.stop());
    mediaStreamRef.current = null;
    if (audioCtxRef.current) {
      try {
        await audioCtxRef.current.close();
      } catch {
        // Ignore close errors
      }
      audioCtxRef.current = null;
    }

    const chunks = pcmChunksRef.current;
    pcmChunksRef.current = [];
    if (!chunks.length) return;

    const wavBlob = encodeWavBlob(chunks, recorderSampleRateRef.current);
    const form = new FormData();
    form.append("audio", wavBlob, "clinician.wav");
    setBusy(true);
    setStatusNote("Transcribing via Whisper...");
    try {
      const res = await fetch(`${apiBase}/api/stt`, { method: "POST", body: form });
      const payload = await res.json();
      if (res.ok) {
        if (payload.transcript) {
          const text = String(payload.transcript);
          sessionTranscriptRef.current = text;
          malcolmStore.setClinicianCaption(text);
        }
        setStatusNote("");
        window.malcolm?.apply(payload);
      } else {
        setStatusNote(payload.error || "STT failed");
      }
    } catch {
      setStatusNote("STT upload failed");
    } finally {
      setBusy(false);
    }
  }, [apiBase]);

  const stopLiveMicPoll = useCallback(() => {
    if (livePollRef.current !== null) {
      window.clearInterval(livePollRef.current);
      livePollRef.current = null;
    }
    if (liveSseRef.current !== null) {
      liveSseRef.current.close();
      liveSseRef.current = null;
    }
  }, []);

  const stopHostMicAndTranscribe = useCallback(async () => {
    stopLiveMicPoll();
    setBusy(true);
    setStatusNote("Finalizing speech...");
    try {
      const res = await fetch(`${apiBase}/api/mic/stop`, { method: "POST" });
      const payload = await res.json();
      if (res.ok && payload.transcript) {
        const text = String(payload.transcript);
        sessionTranscriptRef.current = "";
        browserTranscriptRef.current = "";
        malcolmStore.setClinicianCaption(text);
        setStatusNote("");
        window.malcolm?.apply(payload);
      } else if (res.ok && !(sessionTranscriptRef.current.trim() || browserTranscriptRef.current.trim())) {
        setStatusNote("");
        window.malcolm?.apply(payload);
      } else {
        const fallbackText = (sessionTranscriptRef.current || browserTranscriptRef.current).trim();
        if (fallbackText) {
          sessionTranscriptRef.current = "";
          browserTranscriptRef.current = "";
          await sendClinicianCommand(fallbackText);
        } else {
          setStatusNote(payload.error || "STT failed");
        }
      }
    } catch {
      const fallbackText = (sessionTranscriptRef.current || browserTranscriptRef.current).trim();
      if (fallbackText) {
        sessionTranscriptRef.current = "";
        browserTranscriptRef.current = "";
        await sendClinicianCommand(fallbackText);
      } else {
        setStatusNote("Host STT failed");
      }
    } finally {
      setBusy(false);
    }
  }, [apiBase, sendClinicianCommand, stopLiveMicPoll]);

  const stopListening = useCallback(() => {
    micActiveRef.current = false;
    setMicActive(false);
    stopLiveMicPoll();
    try {
      if (recognitionRef.current) {
        recognitionRef.current.onend = null;
        recognitionRef.current.stop();
      }
    } catch {
      // Ignore stop errors
    }
    recognitionRef.current = null;

    if (hostMicRecordingRef.current) {
      hostMicRecordingRef.current = false;
      void stopHostMicAndTranscribe();
      return;
    }

    if (mediaStreamRef.current || audioCtxRef.current) {
      void stopWavFallbackAndTranscribe();
      return;
    }

    const pendingBrowserText = (sessionTranscriptRef.current || browserTranscriptRef.current).trim();
    if (pendingBrowserText) {
      sessionTranscriptRef.current = "";
      browserTranscriptRef.current = "";
      void sendClinicianCommand(pendingBrowserText);
    }
  }, [sendClinicianCommand, stopHostMicAndTranscribe, stopLiveMicPoll, stopWavFallbackAndTranscribe]);

  const startWavFallbackRecording = useCallback(async () => {
    if (!navigator.mediaDevices?.getUserMedia) {
      setStatusNote("Microphone unavailable on host or browser — type command in Test drawer");
      return;
    }
    try {
      let audioConstraint: MediaTrackConstraints | boolean = true;
      if (navigator.mediaDevices.enumerateDevices) {
        const devs = await navigator.mediaDevices.enumerateDevices();
        const hwMic = devs.find(
          (d) =>
            d.kind === "audioinput" &&
            !/camo|virtual|vb-audio|obs|droidcam|iriun/i.test(d.label),
        );
        if (hwMic?.deviceId) {
          audioConstraint = { deviceId: { ideal: hwMic.deviceId } };
        }
      }
      const stream = await navigator.mediaDevices.getUserMedia({ audio: audioConstraint });
      const AudioCtx = window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
      const ctx = new AudioCtx();
      const source = ctx.createMediaStreamSource(stream);
      const processor = ctx.createScriptProcessor(4096, 1, 1);

      pcmChunksRef.current = [];
      recorderSampleRateRef.current = ctx.sampleRate;
      processor.onaudioprocess = (ev) => {
        if (!micActiveRef.current) return;
        const input = ev.inputBuffer.getChannelData(0);
        pcmChunksRef.current.push(new Float32Array(input));
      };

      source.connect(processor);
      processor.connect(ctx.destination);
      mediaStreamRef.current = stream;
      audioCtxRef.current = ctx;
      micActiveRef.current = true;
      setMicActive(true);
      setStatusNote("Recording WAV — press Space again to transcribe");
    } catch {
      setStatusNote("Microphone permission denied");
      micActiveRef.current = false;
      setMicActive(false);
    }
  }, []);

  const startLiveMicPoll = useCallback(() => {
    stopLiveMicPoll();

    interface LiveMicStatus {
      live_transcript?: string;
      speech_detected?: boolean;
      recording?: boolean;
      auto_stopped?: boolean;
      final_payload?: { transcript?: string; [key: string]: unknown };
      error?: string;
    }

    const handleLivePayload = async (data: LiveMicStatus) => {
      if (!hostMicRecordingRef.current) return;

      if (data.speech_detected) {
        hostSpeechDetectedRef.current = true;
      }
      if (data.live_transcript) {
        const live = String(data.live_transcript);
        sessionTranscriptRef.current = live;
        malcolmStore.setClinicianCaption(live);
        applyOptimisticPaneIntent(live);
      }
      if (data.speech_detected && data.live_transcript && !data.auto_stopped) {
        setStatusNote("Listening (auto-sends on pause)...");
      } else if (data.auto_stopped && !data.final_payload && data.recording) {
        setStatusNote("Refining transcript...");
      }

      if (data.final_payload || !data.recording) {
        stopLiveMicPoll();
        hostMicRecordingRef.current = false;
        micActiveRef.current = false;
        setMicActive(false);
        try {
          if (recognitionRef.current) {
            recognitionRef.current.onend = null;
            recognitionRef.current.stop();
          }
        } catch {
          // Ignore stop errors
        }
        recognitionRef.current = null;

        const finalPayload = data.final_payload;
        if (finalPayload) {
          if (finalPayload.transcript) {
            const finalText = String(finalPayload.transcript);
            sessionTranscriptRef.current = "";
            browserTranscriptRef.current = "";
            malcolmStore.setClinicianCaption(finalText);
          }
          setStatusNote("");
          window.malcolm?.apply(finalPayload);
        } else if (sessionTranscriptRef.current.trim() || browserTranscriptRef.current.trim()) {
          const rescueText = (sessionTranscriptRef.current || browserTranscriptRef.current).trim();
          sessionTranscriptRef.current = "";
          browserTranscriptRef.current = "";
          await sendClinicianCommand(rescueText);
        } else if (data.error) {
          setStatusNote(String(data.error));
        } else {
          setStatusNote("");
        }
      }
    };

    if (typeof EventSource !== "undefined") {
      try {
        const sse = new EventSource(`${apiBase}/api/mic/stream`);
        liveSseRef.current = sse;
        sse.onmessage = (evt) => {
          try {
            const parsed = JSON.parse(evt.data) as LiveMicStatus;
            void handleLivePayload(parsed);
          } catch {
            // Ignore malformed SSE frame
          }
        };
        sse.onerror = () => {
          sse.close();
          if (liveSseRef.current === sse) {
            liveSseRef.current = null;
          }
        };
      } catch {
        // Fallback to interval polling below
      }
    }

    livePollRef.current = window.setInterval(() => {
      if (!hostMicRecordingRef.current) {
        stopLiveMicPoll();
        return;
      }
      if (liveSseRef.current && liveSseRef.current.readyState === EventSource.OPEN) {
        return;
      }
      void (async () => {
        try {
          const res = await fetch(`${apiBase}/api/mic/live`, { cache: "no-store" });
          if (!res.ok) return;
          const data = (await res.json()) as LiveMicStatus;
          await handleLivePayload(data);
        } catch {
          // Ignore transient poll glitches while recording
        }
      })();
    }, 180);
  }, [apiBase, sendClinicianCommand, stopLiveMicPoll]);

  const startHostMicStreaming = useCallback(async (): Promise<boolean> => {
    try {
      const res = await fetch(`${apiBase}/api/mic/start`, { method: "POST" });
      const payload = await res.json();
      if (res.ok && payload.ok) {
        hostMicRecordingRef.current = true;
        micActiveRef.current = true;
        setMicActive(true);
        setStatusNote("Listening on Mic — speak naturally (auto-sends on pause)");
        startLiveMicPoll();
        return true;
      }
    } catch {
      // Backend host mic unavailable; caller falls back to browser Web Speech / WebAudio
    }
    return false;
  }, [apiBase, startLiveMicPoll]);

  const startAuxiliarySpeechPreview = useCallback((onlyPreviewWhenHostActive: boolean) => {
    const Ctor = getSpeechRecognitionCtor();
    if (!Ctor) return false;

    const recognition = new Ctor();
    recognition.continuous = true;
    recognition.interimResults = true;
    recognition.lang = "en-US";
    let silenceTimer: ReturnType<typeof setTimeout> | null = null;
    let priorRestartChunks = "";
    let latestSessionText = "";

    const clearSilenceTimer = () => {
      if (silenceTimer !== null) {
        clearTimeout(silenceTimer);
        silenceTimer = null;
      }
    };

    const finalizeTranscript = (finalText: string) => {
      clearSilenceTimer();
      sessionTranscriptRef.current = finalText;
      browserTranscriptRef.current = finalText;
      malcolmStore.setClinicianCaption(finalText);
      if (micActiveRef.current) {
        micActiveRef.current = false;
        setMicActive(false);
        stopLiveMicPoll();
        try {
          recognition.onresult = null;
          recognition.onend = null;
          recognition.stop();
        } catch {
          // Ignore stop errors
        }
        recognitionRef.current = null;
        if (hostMicRecordingRef.current) {
          hostMicRecordingRef.current = false;
          void fetch(`${apiBase}/api/mic/stop?discard=1`, { method: "POST" }).catch(() => {});
        }
        sessionTranscriptRef.current = "";
        browserTranscriptRef.current = "";
        void sendClinicianCommand(finalText);
        return true;
      }
      return false;
    };

    recognition.onresult = (event) => {
      if (
        isSpeakingRef.current ||
        (typeof window !== "undefined" && window.speechSynthesis?.speaking)
      ) {
        return;
      }
      clearSilenceTimer();
      const parts: string[] = [];
      let allFinal = true;
      for (let i = 0; i < event.results.length; i++) {
        const result = event.results[i];
        if (!result) continue;
        const transcript = result[0]?.transcript?.trim() || "";
        if (!transcript) continue;
        parts.push(transcript);
        if (!result.isFinal) {
          allFinal = false;
        }
      }
      latestSessionText = parts.join(" ").trim();
      const fullText = [priorRestartChunks, latestSessionText].filter(Boolean).join(" ").trim();
      if (!fullText) return;

      browserTranscriptRef.current = fullText;
      const hostOwnsSession = onlyPreviewWhenHostActive || hostMicRecordingRef.current;

      if (!hostSpeechDetectedRef.current || !sessionTranscriptRef.current) {
        sessionTranscriptRef.current = fullText;
        malcolmStore.setClinicianCaption(fullText);
      }

      if (!hostOwnsSession && allFinal) {
        if (finalizeTranscript(fullText)) {
          return;
        }
        sessionTranscriptRef.current = "";
        browserTranscriptRef.current = "";
        void sendClinicianCommand(fullText);
        return;
      }

      const wordCount = fullText.split(/\s+/).filter(Boolean).length;
      if (wordCount >= 2 && micActiveRef.current) {
        const capturedText = fullText;
        const waitMs = hostOwnsSession ? 1500 : 950;
        silenceTimer = setTimeout(() => {
          silenceTimer = null;
          if (
            !micActiveRef.current ||
            recognitionRef.current !== recognition ||
            isSpeakingRef.current ||
            (typeof window !== "undefined" && window.speechSynthesis?.speaking)
          ) {
            return;
          }
          // When host mic is active and detecting speech, let the host mic VAD/Pass-2 finalize;
          // only rescue via browser SpeechRecognition if the host mic failed to detect speech.
          if ((onlyPreviewWhenHostActive || hostMicRecordingRef.current) && hostSpeechDetectedRef.current) {
            return;
          }
          if (!finalizeTranscript(capturedText) && !onlyPreviewWhenHostActive && !hostMicRecordingRef.current) {
            sessionTranscriptRef.current = "";
            browserTranscriptRef.current = "";
            void sendClinicianCommand(capturedText);
          }
        }, waitMs);
      }
    };

    recognition.onerror = (event) => {
      clearSilenceTimer();
      if (onlyPreviewWhenHostActive || hostMicRecordingRef.current) {
        recognition.onend = null;
        recognitionRef.current = null;
        return;
      }
      if (
        event.error === "not-allowed" ||
        event.error === "service-not-allowed" ||
        event.error === "audio-capture" ||
        event.error === "network"
      ) {
        recognition.onend = null;
        try {
          recognition.stop();
        } catch {
          // Ignore stop error
        }
        recognitionRef.current = null;
        micActiveRef.current = false;
        setMicActive(false);
        void startWavFallbackRecording();
      }
    };

    recognition.onend = () => {
      clearSilenceTimer();
      if (micActiveRef.current) {
        if (latestSessionText) {
          priorRestartChunks = [priorRestartChunks, latestSessionText].filter(Boolean).join(" ").trim();
          latestSessionText = "";
        }
        try {
          recognition.start();
        } catch {
          if (!hostMicRecordingRef.current) {
            micActiveRef.current = false;
            setMicActive(false);
          }
        }
      }
    };

    recognitionRef.current = recognition;
    try {
      recognition.start();
      return true;
    } catch {
      recognition.onend = null;
      recognitionRef.current = null;
      return false;
    }
  }, [apiBase, sendClinicianCommand, startWavFallbackRecording, stopLiveMicPoll]);

  const startListening = useCallback(() => {
    stopAllSpeech();
    sessionTranscriptRef.current = "";
    browserTranscriptRef.current = "";
    hostSpeechDetectedRef.current = false;
    micActiveRef.current = true;
    setMicActive(true);
    setStatusNote("Listening on Mic — speak naturally (auto-sends on pause)");

    void (async () => {
      const hostStarted = await startHostMicStreaming();
      if (hostStarted) {
        if (micActiveRef.current) {
          startAuxiliarySpeechPreview(true);
        }
        return;
      }

      const fallbackBrowserStarted = startAuxiliarySpeechPreview(false);
      if (fallbackBrowserStarted) {
        setStatusNote("Listening — speak naturally");
        return;
      }
      setSpeechSupported(false);
      await startWavFallbackRecording();
    })();
  }, [
    startAuxiliarySpeechPreview,
    startHostMicStreaming,
    startWavFallbackRecording,
    stopAllSpeech,
  ]);

  const toggleMic = useCallback(() => {
    if (micActiveRef.current) {
      stopListening();
    } else {
      startListening();
    }
  }, [startListening, stopListening]);

  // Hands-free keyboard shortcuts: Spacebar toggles mic, R resets demo
  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.metaKey || e.ctrlKey || e.altKey || e.repeat) return;
      const el = e.target as HTMLElement | null;
      if (el && (el.isContentEditable || /^(INPUT|TEXTAREA|SELECT)$/.test(el.tagName))) return;

      if (e.code === "Space") {
        e.preventDefault();
        toggleMic();
      } else if (e.key === "r" || e.key === "R") {
        e.preventDefault();
        void resetDemo();
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => {
      window.removeEventListener("keydown", onKeyDown);
      stopListening();
    };
  }, [resetDemo, stopListening, toggleMic]);

  const handleManualSubmit = (e: FormEvent) => {
    e.preventDefault();
    if (!manualInput.trim()) return;
    void sendClinicianCommand(manualInput);
    setManualInput("");
  };

  return (
    <div className="fixed bottom-3 left-3 z-30 flex flex-col items-start gap-2">
      {testOpen ? (
        <div className="ledger-glass w-[min(92vw,430px)] border border-frame p-3 shadow-2xl">
          <div className="mb-2 flex items-center justify-between border-b border-hairline pb-2">
            <div className="flex flex-col gap-0.5">
              <span className="font-mono text-[10px] uppercase tracking-[0.18em] text-label">
                Testing Console (Voice &amp; Bedside Simulation)
              </span>
              <span className="font-mono text-[9px] uppercase tracking-[0.12em] text-primary">
                {webAiReady
                  ? "Gemini 3.8 Flash API + Google WebAI + Local RAG"
                  : "Gemini 3.8 Flash API + Local RAG (WebAI Standby)"}
              </span>
            </div>
            <a
              href={apiBase}
              target="_blank"
              rel="noreferrer"
              className="flex items-center gap-1 font-mono text-[10px] uppercase tracking-[0.14em] text-primary hover:underline"
            >
              Bedside UI
              <ExternalLink className="size-3" />
            </a>
          </div>

          <div className="mb-2 flex flex-wrap gap-1">
            {[
              { id: "1429", label: "1429 Mika" },
              { id: "1430", label: "1430 Aino" },
              { id: "1431", label: "1431 Jukka" },
              { id: "1432", label: "1432 Sanna" },
            ].map((p) => {
              const active = patient?.patientId === p.id;
              return (
                <button
                  key={p.id}
                  type="button"
                  disabled={busy}
                  onClick={() => void selectPatient(p.id)}
                  className={cn(
                    "border px-2 py-0.5 font-mono text-[10px] uppercase tracking-wider transition-colors disabled:opacity-50",
                    active
                      ? "border-primary bg-primary/20 text-primary"
                      : "border-frame bg-background/60 text-text-soft hover:border-primary hover:text-primary",
                  )}
                >
                  {p.label}
                </button>
              );
            })}
            <button
              type="button"
              disabled={busy}
              onClick={() => void sendClinicianCommand("List all patients in the database")}
              className="border border-frame bg-background/60 px-2 py-0.5 font-mono text-[10px] uppercase tracking-wider text-text-bright transition-colors hover:border-primary hover:text-primary disabled:opacity-50"
            >
              List All Patients
            </button>
          </div>

          <div className="mb-2 flex flex-wrap gap-1.5">
            <button
              type="button"
              disabled={busy}
              onClick={() => void sendClinicianCommand("Malcolm, can you please give me the status of patient 1429?")}
              className="border border-frame bg-background/60 px-2 py-1 font-mono text-[10px] uppercase tracking-wider text-text-bright transition-colors hover:border-primary hover:text-primary disabled:opacity-50"
            >
              1. Ask 1429 Status
            </button>
            <button
              type="button"
              disabled={busy}
              onClick={() => void sendClinicianCommand("Yes, him.")}
              className="border border-frame bg-background/60 px-2 py-1 font-mono text-[10px] uppercase tracking-wider text-text-bright transition-colors hover:border-primary hover:text-primary disabled:opacity-50"
            >
              2. Yes, him.
            </button>
            <button
              type="button"
              disabled={busy}
              onClick={() => void sendClinicianCommand("Yes, please.")}
              className="border border-frame bg-background/60 px-2 py-1 font-mono text-[10px] uppercase tracking-wider text-text-bright transition-colors hover:border-primary hover:text-primary disabled:opacity-50"
            >
              3. Yes, please.
            </button>
          </div>

          <div className="mb-2.5 flex flex-wrap gap-1.5">
            <button
              type="button"
              disabled={busy}
              onClick={() =>
                void sendBedsideTestCommand(
                  "Yes, I have been facing this itch in my right foot. It won't go away and it's becoming very uncomfortable.",
                )
              }
              className="border border-rose/40 bg-alert-tint/30 px-2 py-1 font-mono text-[10px] uppercase tracking-wider text-rose-soft transition-colors hover:border-rose disabled:opacity-50"
            >
              Patient: Right Foot Itch
            </button>
            <button
              type="button"
              disabled={busy}
              onClick={() => void sendBedsideTestCommand("Yes please.")}
              className="border border-rose/40 bg-alert-tint/30 px-2 py-1 font-mono text-[10px] uppercase tracking-wider text-rose-soft transition-colors hover:border-rose disabled:opacity-50"
            >
              Patient: Yes please.
            </button>
          </div>

          <form onSubmit={handleManualSubmit} className="flex gap-1.5">
            <input
              type="text"
              value={manualInput}
              onChange={(e) => setManualInput(e.target.value)}
              placeholder="Type clinician test command..."
              className="min-w-0 flex-1 border border-frame bg-background/70 px-2.5 py-1.5 font-sans text-xs text-foreground placeholder:text-label focus:border-primary focus:outline-none"
            />
            <button
              type="submit"
              disabled={busy || !manualInput.trim()}
              className="flex items-center gap-1 border border-primary/60 bg-primary/15 px-2.5 py-1.5 font-mono text-[10px] uppercase tracking-wider text-primary hover:bg-primary/25 disabled:opacity-40"
            >
              <Send className="size-3" />
              Send
            </button>
          </form>
        </div>
      ) : null}

      <div
        className={cn(
          "ledger-glass flex flex-wrap items-center gap-1 border border-frame p-1",
          isSplit ? "max-w-[calc(38vw-1.5rem)]" : "max-w-[calc(100vw-4rem)]",
        )}
      >
        <button
          type="button"
          onClick={toggleMic}
          title={
            speechSupported
              ? "Toggle hands-free voice [Space]"
              : "Record audio via host microphone STT [Space]"
          }
          className={cn(
            "flex items-center gap-1.5 px-2.5 py-1 font-mono text-[11px] uppercase tracking-[0.14em] transition-colors",
            micActive
              ? "bg-alert text-foreground"
              : "text-text-soft hover:bg-accent hover:text-foreground",
          )}
        >
          {micActive ? <Mic className="size-3.5 animate-pulse" /> : <MicOff className="size-3.5" />}
          <span>{micActive ? "Listening" : "Voice [Space]"}</span>
        </button>

        <button
          type="button"
          onClick={() => {
            const next = !ttsEnabled;
            setTtsEnabled(next);
            if (!next) stopAllSpeech();
          }}
          title={ttsEnabled ? "Mute Malcolm Jarvis voice" : "Unmute Malcolm Jarvis voice"}
          className="flex items-center gap-1 px-2 py-1 font-mono text-[11px] uppercase tracking-[0.14em] text-text-soft transition-colors hover:bg-accent hover:text-foreground"
        >
          {ttsEnabled ? <Volume2 className="size-3.5 text-primary" /> : <VolumeX className="size-3.5" />}
        </button>

        <button
          type="button"
          onClick={() => void resetDemo()}
          disabled={busy}
          title="Reset patient demo state [R]"
          className="flex items-center gap-1 px-2 py-1 font-mono text-[11px] uppercase tracking-[0.14em] text-text-soft transition-colors hover:bg-accent hover:text-foreground disabled:opacity-40"
        >
          <RotateCcw className="size-3.5" />
          <span>Reset</span>
        </button>

        <button
          type="button"
          onClick={() => setTestOpen((v) => !v)}
          title="Toggle testing console"
          className={cn(
            "flex items-center gap-1 px-2 py-1 font-mono text-[11px] uppercase tracking-[0.14em] transition-colors",
            testOpen ? "bg-primary/20 text-primary" : "text-text-soft hover:bg-accent hover:text-foreground",
          )}
        >
          <SlidersHorizontal className="size-3.5" />
          <span>Test</span>
        </button>

        {statusNote ? (
          <span
            className={cn(
              "max-w-[200px] truncate px-2 font-mono text-[10px]",
              micActive || /^(Transcribing|Recording|Listening|Opening|Finalizing)/.test(statusNote)
                ? "text-primary"
                : "text-rose-soft",
            )}
          >
            {statusNote}
          </span>
        ) : null}
      </div>
    </div>
  );
}
