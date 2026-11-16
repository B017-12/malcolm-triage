import { useCallback, useEffect, useRef, useState } from "react";
import {
  Camera,
  CameraOff,
  Hand,
  Mic,
  MicOff,
  RotateCcw,
  Volume2,
  VolumeX,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { DEFAULT_CAPTIONS, malcolmStore, useMalcolm } from "@/lib/malcolm/store";
import type { MalcolmMode } from "@/lib/malcolm/signal";
import { getLastOpenPaneKind, scrollRightPaneBy, syncPaneBackend } from "@/components/malcolm/PatientPane";

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
function applyOptimisticPaneIntent(utterance: string, _triggerDownload = false) {
  const lowered = utterance.toLowerCase().trim();
  if (!lowered) return;

  // 1. Close / hide / dismiss pane
  if (
    /\b(close|hide|dismiss|remove|minimize|collapse|put away|shut)\b.*\b(pane|panel|table|timetable|schedule|chart|stats|statistics|display|screen|board|roster|view|it|that)\b/.test(
      lowered,
    ) ||
    /^(close|hide|dismiss)\s*(pane|table|stats|display|it|that)?[.!?]*$/.test(lowered)
  ) {
    malcolmStore.setMode("idle");
    malcolmStore.setCustomPane({
      kind: "patient",
      title: "",
      subtitle: "",
      filterWing: "",
      generatedHtml: "",
    });
    return;
  }

  // 2. Timetable / Schedule / Calendar / Reminder pane (MUST run before generic 'table' match)
  if (
    /\b(time table|timetable|time-table|schedule|calendar|agenda|itinerary|timings|clock out|clock-out|remind me|reminder|loudspeaker|page me)\b/.test(
      lowered,
    ) ||
    (/\bsurgery\b/.test(lowered) && /\b(pm|am|at|before|after|then|check in|check-in|table)\b/.test(lowered))
  ) {
    malcolmStore.setCustomPane({
      kind: "timetable",
      title: "Clinician Shift Timetable & Reminder Schedule",
      subtitle: "Synced to Real PC Clock • 5-Min Prior Loudspeaker Paging",
      filterWing: "",
    });
    malcolmStore.setMode("patient");
    return;
  }

  // 3. Statistics / analytics / metrics pane
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

  // 4. Ward Table / All-patients roster
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
  const { caption, replySeq, mode } = useMalcolm();
  const isSplit = mode === "patient";
  const [micActive, setMicActive] = useState(false);
  const [ttsEnabled, setTtsEnabled] = useState(true);
  const [cameraActive, setCameraActive] = useState(false);
  const [gestureLabel, setGestureLabel] = useState("SHOW HAND FOR GESTURES");
  const [busy, setBusy] = useState(false);
  const [speechSupported, setSpeechSupported] = useState(true);
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
  const lastSpokenReplySeqRef = useRef<number>(0);
  const lastGestureLabelRef = useRef<string>("SHOW HAND FOR GESTURES");

  const updateGestureLabel = useCallback((nextLabel: string) => {
    if (lastGestureLabelRef.current !== nextLabel) {
      lastGestureLabelRef.current = nextLabel;
      setGestureLabel(nextLabel);
    }
  }, []);

  // Fallback WAV recorder refs when host + browser SpeechRecognition are unavailable
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const audioCtxRef = useRef<AudioContext | null>(null);
  const pcmChunksRef = useRef<Float32Array[]>([]);
  const recorderSampleRateRef = useRef<number>(16000);

  const apiBase = getApiBase();

  useEffect(() => {
    setSpeechSupported(getSpeechRecognitionCtor() !== null);
  }, []);

  const stopAllSpeech = useCallback(() => {
    isSpeakingRef.current = false;
    ttsAbortRef.current?.abort();
    ttsAbortRef.current = null;

    // Mark current store caption & replySeq as already handled so polling cannot re-trigger muted speech
    const currentStore = malcolmStore.get();
    if (currentStore.caption?.trim()) {
      lastSpokenCaptionRef.current = currentStore.caption.trim();
    }
    if (typeof currentStore.replySeq === "number" && currentStore.replySeq > 0) {
      lastSpokenReplySeqRef.current = currentStore.replySeq;
    }

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
    if (!trimmed) return;
    const hasNewReplySeq =
      typeof replySeq === "number" && replySeq > 0 && replySeq !== lastSpokenReplySeqRef.current;
    if (!hasNewReplySeq && trimmed === lastSpokenCaptionRef.current) return;

    lastSpokenCaptionRef.current = trimmed;
    if (typeof replySeq === "number" && replySeq > 0) {
      lastSpokenReplySeqRef.current = replySeq;
    }

    stopAllSpeech();
    const controller = new AbortController();
    ttsAbortRef.current = controller;
    // Set isSpeakingRef immediately so no gesture/mic timer mistakes TTS fetch latency for silence
    isSpeakingRef.current = true;

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
          })
            .then(async (res) => {
              if (!res.ok) throw new Error("Neural TTS unavailable");
              return res.blob();
            })
            .catch((err) => {
              if ((err as { name?: string })?.name === "AbortError" || controller.signal.aborted) {
                return null;
              }
              throw err;
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
      } finally {
        if (
          ttsAbortRef.current === controller &&
          !(typeof window !== "undefined" && window.speechSynthesis?.speaking)
        ) {
          isSpeakingRef.current = false;
        }
      }
    })();
  }, [apiBase, caption, replySeq, speakWithBrowserFallback, stopAllSpeech, ttsEnabled]);

  useEffect(() => stopAllSpeech, [stopAllSpeech]);

  const sendClinicianCommand = useCallback(
    async (message: string) => {
      const cleaned = message.trim();
      if (!cleaned) return;
      setBusy(true);
      setStatusNote("");
      malcolmStore.setClinicianCaption(cleaned);
      applyOptimisticPaneIntent(cleaned, true);
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

  const cancelListeningAndDiscard = useCallback(() => {
    micActiveRef.current = false;
    setMicActive(false);
    stopLiveMicPoll();
    sessionTranscriptRef.current = "";
    browserTranscriptRef.current = "";
    hostSpeechDetectedRef.current = false;
    try {
      if (recognitionRef.current) {
        recognitionRef.current.onresult = null;
        recognitionRef.current.onend = null;
        recognitionRef.current.stop();
      }
    } catch {
      // Ignore stop errors
    }
    recognitionRef.current = null;

    if (hostMicRecordingRef.current) {
      hostMicRecordingRef.current = false;
      void fetch(`${apiBase}/api/mic/stop?discard=1`, { method: "POST" }).catch(() => {});
    }

    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach((t) => t.stop());
      mediaStreamRef.current = null;
    }
    if (audioCtxRef.current) {
      void audioCtxRef.current.close().catch(() => {});
      audioCtxRef.current = null;
    }
    pcmChunksRef.current = [];
    setStatusNote("");
  }, [apiBase, stopLiveMicPoll]);

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

  // Camera Hand-Gesture Navigation System refs
  const camVideoRef = useRef<HTMLVideoElement | null>(null);
  const camCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const camStreamRef = useRef<MediaStream | null>(null);
  const camRafRef = useRef<number | null>(null);
  const handLandmarkerRef = useRef<unknown>(null);
  const cameraActiveRef = useRef(false);
  const gestureHistoryRef = useRef<{ t: number; x: number; y: number }[]>([]);
  const prevWristRef = useRef<{ x: number; y: number } | null>(null);
  const poseWindowRef = useRef<("OPEN" | "FIST" | "SCROLL" | "NEUTRAL")[]>([]);
  const wasOpenHandRef = useRef(false);
  const wasClosedHandRef = useRef(false);
  const lastOpenListenAtRef = useRef(0);
  const lastSwipeAtRef = useRef(0);
  const lastScrollAtRef = useRef(0);

  // Open Hand -> Silently enters Listening mode ("opening it should mean its just listening to me")
  const triggerOpenHandListen = useCallback(() => {
    stopAllSpeech();
    if (!micActiveRef.current) {
      startListening();
    } else {
      setStatusNote("Listening on Mic — speak naturally (auto-sends on pause)");
    }
  }, [startListening, stopAllSpeech]);

  const stopCameraGestures = useCallback(() => {
    cameraActiveRef.current = false;
    setCameraActive(false);
    if (camRafRef.current !== null) {
      cancelAnimationFrame(camRafRef.current);
      camRafRef.current = null;
    }
    if (camStreamRef.current) {
      camStreamRef.current.getTracks().forEach((t) => t.stop());
      camStreamRef.current = null;
    }
    if (camVideoRef.current) {
      camVideoRef.current.srcObject = null;
    }
    prevWristRef.current = null;
    gestureHistoryRef.current = [];
    poseWindowRef.current = [];
    wasOpenHandRef.current = false;
    wasClosedHandRef.current = false;
  }, []);

  const startCameraGestures = useCallback(async () => {
    if (cameraActiveRef.current) return;
    setStatusNote("Starting Camera Gesture System...");
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 320 }, height: { ideal: 240 }, facingMode: "user" },
        audio: false,
      });
      camStreamRef.current = stream;
      cameraActiveRef.current = true;
      setCameraActive(true);
      updateGestureLabel("LOADING HAND TRACKER...");

      if (!handLandmarkerRef.current) {
        const visionPkg = await import("@mediapipe/tasks-vision");
        const { FilesetResolver, HandLandmarker } = visionPkg;
        let vision: Awaited<ReturnType<typeof FilesetResolver.forVisionTasks>>;
        try {
          vision = await FilesetResolver.forVisionTasks("/mediapipe/wasm");
        } catch {
          vision = await FilesetResolver.forVisionTasks(
            "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm",
          );
        }
        try {
          handLandmarkerRef.current = await HandLandmarker.createFromOptions(vision, {
            baseOptions: {
              modelAssetPath: "/mediapipe/hand_landmarker.task",
              delegate: "GPU",
            },
            runningMode: "VIDEO",
            numHands: 1,
            minHandDetectionConfidence: 0.56,
            minHandPresenceConfidence: 0.56,
            minTrackingConfidence: 0.54,
          });
        } catch {
          handLandmarkerRef.current = await HandLandmarker.createFromOptions(vision, {
            baseOptions: {
              modelAssetPath: "/mediapipe/hand_landmarker.task",
              delegate: "CPU",
            },
            runningMode: "VIDEO",
            numHands: 1,
            minHandDetectionConfidence: 0.56,
            minHandPresenceConfidence: 0.56,
            minTrackingConfidence: 0.54,
          });
        }
      }

      const video = camVideoRef.current;
      if (!video) return;
      video.srcObject = stream;
      await video.play();
      setStatusNote("");
      updateGestureLabel("READY — SHOW HAND");

      const CONNECTIONS: [number, number][] = [
        [0, 1], [1, 2], [2, 3], [3, 4],
        [0, 5], [5, 6], [6, 7], [7, 8],
        [5, 9], [9, 10], [10, 11], [11, 12],
        [9, 13], [13, 14], [14, 15], [15, 16],
        [13, 17], [17, 18], [18, 19], [19, 20],
        [0, 17],
      ];

      let lastVideoTime = -1;
      const detectFrame = () => {
        if (!cameraActiveRef.current) return;
        const vid = camVideoRef.current;
        const canvas = camCanvasRef.current;
        const landmarker = handLandmarkerRef.current as {
          detectForVideo?: (v: HTMLVideoElement, t: number) => {
            landmarks?: { x: number; y: number; z: number }[][];
            worldLandmarks?: { x: number; y: number; z: number }[][];
          };
        } | null;

        if (vid && canvas && landmarker?.detectForVideo && vid.readyState >= 2) {
          const now = performance.now();
          if (vid.currentTime !== lastVideoTime) {
            lastVideoTime = vid.currentTime;
            const res = landmarker.detectForVideo(vid, now);
            const ctx = canvas.getContext("2d");
            if (ctx) {
              ctx.save();
              ctx.clearRect(0, 0, canvas.width, canvas.height);
              // Draw mirrored webcam video
              ctx.translate(canvas.width, 0);
              ctx.scale(-1, 1);
              ctx.drawImage(vid, 0, 0, canvas.width, canvas.height);
              ctx.restore();
            }

            const lm = res?.landmarks?.[0];
            if (lm && lm.length >= 21) {
              // Draw mirrored hand skeleton
              if (ctx) {
                ctx.strokeStyle = "rgba(56, 189, 248, 0.85)";
                ctx.lineWidth = 1.75;
                for (const [a, b] of CONNECTIONS) {
                  const pa = lm[a]!;
                  const pb = lm[b]!;
                  ctx.beginPath();
                  ctx.moveTo((1 - pa.x) * canvas.width, pa.y * canvas.height);
                  ctx.lineTo((1 - pb.x) * canvas.width, pb.y * canvas.height);
                  ctx.stroke();
                }
                ctx.fillStyle = "#38bdf8";
                for (const pt of lm) {
                  ctx.beginPath();
                  ctx.arc((1 - pt.x) * canvas.width, pt.y * canvas.height, 2.5, 0, Math.PI * 2);
                  ctx.fill();
                }
              }

              // Use true 3D worldLandmarks when available (immune to camera tilt/foreshortening),
              // falling back to 2D normalized landmarks
              const pts = res?.worldLandmarks?.[0]?.length === 21 ? res.worldLandmarks[0]! : lm;
              const dist3 = (i: number, j: number) =>
                Math.hypot(
                  pts[i]!.x - pts[j]!.x,
                  pts[i]!.y - pts[j]!.y,
                  (pts[i]!.z || 0) - (pts[j]!.z || 0),
                );
              const angleAt = (a: number, b: number, c: number) => {
                const v1x = pts[a]!.x - pts[b]!.x;
                const v1y = pts[a]!.y - pts[b]!.y;
                const v1z = (pts[a]!.z || 0) - (pts[b]!.z || 0);
                const v2x = pts[c]!.x - pts[b]!.x;
                const v2y = pts[c]!.y - pts[b]!.y;
                const v2z = (pts[c]!.z || 0) - (pts[b]!.z || 0);
                const mag = Math.hypot(v1x, v1y, v1z) * Math.hypot(v2x, v2y, v2z);
                if (!mag) return 0;
                const cos = Math.max(-1, Math.min(1, (v1x * v2x + v1y * v2y + v1z * v2z) / mag));
                return Math.acos(cos) * (180 / Math.PI);
              };

              const palmSize = Math.max(0.03, dist3(0, 9));

              // Mirrored X: 0 = user's left, 1 = user's right.
              // Track the palm + index centroid for horizontal swipe (moves 2.5x more than the wrist pivot!)
              const rawHandX = 1 - (lm[0]!.x * 0.2 + lm[9]!.x * 0.5 + lm[8]!.x * 0.3);
              const rawHandY = lm[0]!.y * 0.3 + lm[9]!.y * 0.4 + lm[8]!.y * 0.3;
              const mirroredWristX = 1 - lm[0]!.x;
              const rawWristY = lm[0]!.y;

              // Exponentially smooth wrist coordinates to eliminate webcam micro-jitter during vertical scroll
              const prevWrist = prevWristRef.current;
              const smoothX = prevWrist ? prevWrist.x * 0.45 + mirroredWristX * 0.55 : mirroredWristX;
              const smoothY = prevWrist ? prevWrist.y * 0.55 + rawWristY * 0.45 : rawWristY;
              const frameDx = prevWrist ? smoothX - prevWrist.x : 0;
              const frameDy = prevWrist ? smoothY - prevWrist.y : 0;
              const wristSpeed = Math.hypot(frameDx, frameDy);
              prevWristRef.current = { x: smoothX, y: smoothY };

              // Finger is extended when straight at PIP (> 132 deg) & tip farther than PIP from wrist
              const isFingerExtended = (tip: number, pip: number, mcp: number) =>
                dist3(tip, 0) > dist3(pip, 0) * 1.05 && angleAt(mcp, pip, tip) > 132;

              // Strict Closed-Fist curl check so a relaxed resting hand NEVER triggers Closed Fist:
              // A finger is curled ONLY when its tip is folded closer to the wrist than its PIP joint,
              // bent below 98 deg at the PIP joint, and tucked close to its MCP knuckle & palm center.
              const isFingerCurled = (tip: number, pip: number, mcp: number) =>
                dist3(tip, 0) < dist3(pip, 0) * 0.98 &&
                dist3(tip, 0) < dist3(mcp, 0) * 1.24 &&
                angleAt(mcp, pip, tip) < 98 &&
                dist3(tip, 9) < palmSize * 0.84;

              const indexExt = isFingerExtended(8, 6, 5);
              const middleExt = isFingerExtended(12, 10, 9);
              const ringExt = isFingerExtended(16, 14, 13);
              const pinkyExt = isFingerExtended(20, 18, 17);

              const curledCount = [
                isFingerCurled(8, 6, 5),
                isFingerCurled(12, 10, 9),
                isFingerCurled(16, 14, 13),
                isFingerCurled(20, 18, 17),
              ].filter(Boolean).length;

              const fourFingersOut = indexExt && middleExt && ringExt && pinkyExt;

              // Thumb spread out from Index MCP (5) & Middle MCP (9) and straight at IP joint
              const thumbExt =
                dist3(4, 5) > palmSize * 0.46 &&
                dist3(4, 9) > palmSize * 0.60 &&
                dist3(4, 17) > dist3(3, 17) &&
                angleAt(2, 3, 4) > 138;

              const isVerticalScrollMotion =
                Math.abs(frameDy) > 0.008 && Math.abs(frameDy) > Math.abs(frameDx) * 1.2;

              // Strictly separated poses with velocity gating and a wide NEUTRAL zone for relaxed hands:
              // - Closed Fist: ALL 4 fingers tightly curled into the palm & wrist held relatively steady
              // - Open Hand: 5 fingers spread & held steady (not actively scrolling vertically)
              // - 4-Finger Scroll: 4 fingers extended with either thumb tucked OR active vertical wrist movement
              const isClosedHand =
                curledCount === 4 && !indexExt && !middleExt && wristSpeed < 0.022;
              const isFiveFingerOpen =
                fourFingersOut && thumbExt && wristSpeed < 0.016 && !isVerticalScrollMotion;
              const isFourFingerScroll =
                fourFingersOut && (!thumbExt || isVerticalScrollMotion);

              // Update un-damped 380ms sliding window for responsive horizontal hand swipe detection
              const hist = gestureHistoryRef.current;
              hist.push({ t: now, x: rawHandX, y: rawHandY });
              while (hist.length > 0 && now - hist[0]!.t > 380) {
                hist.shift();
              }

              // 1. HORIZONTAL SWIPE -> Move/swipe right closes pane, move/swipe left opens pane.
              //    Find peak horizontal displacement in the 380ms window & use state-aware cooldown
              //    so a natural wind-up in the already-active direction NEVER blocks the intended swipe!
              let swiped = false;
              if (hist.length >= 2) {
                let bestDx = 0;
                let bestDy = 0;
                let bestDt = 0;
                for (let i = 0; i < hist.length - 1; i++) {
                  const pt = hist[i]!;
                  const dt = now - pt.t;
                  if (dt < 30) continue;
                  const dx = rawHandX - pt.x;
                  const dy = rawHandY - pt.y;
                  if (Math.abs(dx) > Math.abs(bestDx)) {
                    bestDx = dx;
                    bestDy = dy;
                    bestDt = dt;
                  }
                }
                const absDx = Math.abs(bestDx);
                const absDy = Math.abs(bestDy);
                const vx = bestDt > 0 ? absDx / (bestDt / 1000) : 0;

                if (absDx > 0.105 && absDx > absDy * 1.35 && vx > 0.34) {
                  const currentMode = malcolmStore.get().mode;
                  const isRightSwipe = bestDx > 0; // Right -> Close pane (only when pane is open!)
                  const isLeftSwipe = bestDx < 0;  // Left  -> Open pane  (only when pane is closed!)
                  const timeSinceLastSwipe = now - lastSwipeAtRef.current;

                  // State-gated directional swipe:
                  // - When pane is OPEN (mode === "patient"), only Right Swipe triggers Close (leftward wind-up is ignored!).
                  // - When pane is CLOSED (mode !== "patient"), only Left Swipe triggers Open (rightward wind-up is ignored!).
                  if (isRightSwipe && currentMode === "patient" && timeSinceLastSwipe > 650) {
                    lastSwipeAtRef.current = now;
                    poseWindowRef.current = [];
                    gestureHistoryRef.current = [];
                    swiped = true;
                    void syncPaneBackend("close");
                    updateGestureLabel("👉 SWIPE RIGHT • PANE CLOSED");
                  } else if (isLeftSwipe && currentMode !== "patient" && timeSinceLastSwipe > 650) {
                    lastSwipeAtRef.current = now;
                    poseWindowRef.current = [];
                    gestureHistoryRef.current = [];
                    swiped = true;
                    const kind = getLastOpenPaneKind() || malcolmStore.get().customPane?.kind || "patient";
                    void syncPaneBackend(kind);
                    updateGestureLabel("👈 MOVE LEFT • PANE OPENED");
                  }
                }
              }

              if (!swiped) {
                // Push current frame classification into 7-frame sliding window for temporal majority voting
                const framePose: "OPEN" | "FIST" | "SCROLL" | "NEUTRAL" = isClosedHand
                  ? "FIST"
                  : isFiveFingerOpen
                    ? "OPEN"
                    : isFourFingerScroll
                      ? "SCROLL"
                      : "NEUTRAL";

                const win = poseWindowRef.current;
                win.push(framePose);
                if (win.length > 7) {
                  win.shift();
                }

                const fistVotes = win.filter((p) => p === "FIST").length;
                const openVotes = win.filter((p) => p === "OPEN").length;

                // Hold the swipe confirmation label on screen for 850ms so the user clearly sees it
                const inSwipeBannerHold = now - lastSwipeAtRef.current < 850;

                // 2. CLOSED HAND (FIST) -> Majority vote (>= 5 of last 7 frames): immediate stop speech + discard mic
                if (fistVotes >= 5) {
                  wasOpenHandRef.current = false;
                  if (!wasClosedHandRef.current) {
                    wasClosedHandRef.current = true;
                    stopAllSpeech();
                    if (micActiveRef.current || hostMicRecordingRef.current) {
                      cancelListeningAndDiscard();
                    }
                  }
                  updateGestureLabel("✊ CLOSED HAND • STOPPED TALKING");
                }
                // 3. 5-FINGER OPEN HAND -> Majority vote (>= 5 of last 7 frames): enter silent Listening mode
                else if (openVotes >= 5 && now - lastSwipeAtRef.current > 650) {
                  wasClosedHandRef.current = false;
                  if (!wasOpenHandRef.current && now - lastOpenListenAtRef.current > 1800) {
                    wasOpenHandRef.current = true;
                    lastOpenListenAtRef.current = now;
                    triggerOpenHandListen();
                  }
                  updateGestureLabel(
                    micActiveRef.current
                      ? "🖐️ OPEN HAND • LISTENING"
                      : "🖐️ OPEN HAND DETECTED",
                  );
                }
                // 4. 4 FINGERS OUT + WRIST UP/DOWN -> Proportional Scroll on Right Pane
                else if (isFourFingerScroll && !inSwipeBannerHold) {
                  wasClosedHandRef.current = false;
                  if (isVerticalScrollMotion) {
                    wasOpenHandRef.current = false;
                    const scrollPx = frameDy * window.innerHeight * 2.8;
                    scrollRightPaneBy(scrollPx);
                    lastScrollAtRef.current = now;
                    updateGestureLabel(
                      frameDy > 0
                        ? "🖖 4 FINGERS ↓ SCROLLING DOWN"
                        : "🖖 4 FINGERS ↑ SCROLLING UP",
                    );
                  } else {
                    updateGestureLabel("🖖 4 FINGERS • MOVE WRIST ↕ TO SCROLL");
                  }
                } else if (!inSwipeBannerHold) {
                  // Neutral / relaxed hand deadzone: reset edge latches once vote drops below 2
                  if (fistVotes <= 1) wasClosedHandRef.current = false;
                  if (openVotes <= 1) wasOpenHandRef.current = false;
                  updateGestureLabel(
                    micActiveRef.current ? "🖐️ LISTENING ON MIC" : "NEUTRAL HAND • READY",
                  );
                }
              }
            } else {
              // Expire stale swipe history (>380ms) but do NOT wipe recent points so 1-frame motion blur doesn't break a swipe
              const hist = gestureHistoryRef.current;
              while (hist.length > 0 && now - hist[0]!.t > 380) {
                hist.shift();
              }
              prevWristRef.current = null;
              poseWindowRef.current = [];
              wasOpenHandRef.current = false;
              wasClosedHandRef.current = false;
              if (now - lastSwipeAtRef.current >= 850) {
                updateGestureLabel(
                  micActiveRef.current ? "🖐️ LISTENING ON MIC" : "READY — SHOW HAND",
                );
              }
            }
          }
        }
        camRafRef.current = requestAnimationFrame(detectFrame);
      };

      camRafRef.current = requestAnimationFrame(detectFrame);
    } catch {
      cameraActiveRef.current = false;
      setCameraActive(false);
      setStatusNote("Camera permission denied or unavailable");
    }
  }, [cancelListeningAndDiscard, stopAllSpeech, triggerOpenHandListen, updateGestureLabel]);

  const toggleCameraGestures = useCallback(() => {
    if (cameraActiveRef.current) {
      stopCameraGestures();
    } else {
      void startCameraGestures();
    }
  }, [startCameraGestures, stopCameraGestures]);

  // Hands-free keyboard shortcuts: Spacebar toggles mic, G toggles gesture cam, R resets demo
  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.metaKey || e.ctrlKey || e.altKey || e.repeat) return;
      const el = e.target as HTMLElement | null;
      if (el && (el.isContentEditable || /^(INPUT|TEXTAREA|SELECT)$/.test(el.tagName))) return;

      if (e.code === "Space") {
        e.preventDefault();
        toggleMic();
      } else if (e.key === "g" || e.key === "G") {
        e.preventDefault();
        toggleCameraGestures();
      } else if (e.key === "r" || e.key === "R") {
        e.preventDefault();
        void resetDemo();
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => {
      window.removeEventListener("keydown", onKeyDown);
      stopListening();
      stopCameraGestures();
    };
  }, [resetDemo, stopCameraGestures, stopListening, toggleCameraGestures, toggleMic]);

  return (
    <div className="fixed bottom-3 left-3 z-30 flex flex-col items-start gap-2">
      <video ref={camVideoRef} playsInline muted className="hidden" />
      {cameraActive ? (
        <div className="ledger-glass flex flex-col gap-1.5 border border-frame p-2 shadow-2xl">
          <div className="flex items-center justify-between gap-3">
            <span className="flex items-center gap-1.5 font-mono text-[10px] font-semibold uppercase tracking-[0.16em] text-telemetry">
              <Hand className="size-3" />
              {gestureLabel}
            </span>
          </div>
          <div className="relative h-[126px] w-[180px] overflow-hidden border border-hairline bg-background/80">
            <canvas
              ref={camCanvasRef}
              width={180}
              height={126}
              className="h-full w-full object-cover"
            />
          </div>
          <div className="grid grid-cols-2 gap-x-3 gap-y-0.5 font-mono text-[9px] uppercase tracking-[0.12em] text-pane-meta">
            <span>🖐️ Open: Listen</span>
            <span>✊ Close: Stop Talk</span>
            <span>↕ 4-Finger: Scroll</span>
            <span>👈 Left: Open / 👉 Right: Close</span>
          </div>
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
          onClick={toggleCameraGestures}
          title="Toggle Camera Hand-Gesture Navigation [G]"
          className={cn(
            "flex items-center gap-1.5 px-2.5 py-1 font-mono text-[11px] uppercase tracking-[0.14em] transition-colors",
            cameraActive
              ? "bg-telemetry/20 text-telemetry"
              : "text-text-soft hover:bg-accent hover:text-foreground",
          )}
        >
          {cameraActive ? <Camera className="size-3.5 animate-pulse" /> : <CameraOff className="size-3.5" />}
          <span>{cameraActive ? "Gestures On" : "Gestures [G]"}</span>
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

        {statusNote ? (
          <span
            className={cn(
              "max-w-[200px] truncate px-2 font-mono text-[10px]",
              micActive || /^(Transcribing|Recording|Listening|Opening|Finalizing|Starting|Open Hand)/.test(statusNote)
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
