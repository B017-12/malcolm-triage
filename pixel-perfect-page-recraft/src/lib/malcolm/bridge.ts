/**
 * Global hooks so the Python backend (or devtools) can drive Malcolm without touching React:
 *
 *   window.setMalcolmState("idle" | "heart_rate" | "patient_stats", data?)
 *     data: { bpm?, transcript?, patient?: { name, patientId, location, department,
 *             vitals, nurseRounds, patientReport, reportStatus } }
 *
 * It also polls GET http://localhost:5000/api/state every second and applies the response
 * whenever it changes. Expected JSON: { "state": "heart_rate", "data": { "bpm": 90 } }.
 * Offline backends are ignored silently. Override the URL with ?poll=<url>&interval=<ms>.
 */
import { DEFAULT_CAPTIONS, malcolmStore, type MalcolmState } from "./store";
import { parseRemoteState, type RemoteState } from "./signal";

export const DEFAULT_POLL_URL = "http://localhost:5000/api/state";
export const DEFAULT_POLL_INTERVAL = 1000;

export interface MalcolmBridge {
  setState: (stateName: string, data?: unknown) => void;
  apply: (payload: unknown) => void;
  startPolling: (url?: string, intervalMs?: number) => void;
  stopPolling: () => void;
  getState: () => MalcolmState;
}

declare global {
  interface Window {
    malcolm?: MalcolmBridge;
    setMalcolmState?: MalcolmBridge["setState"];
  }
}

let suppressPollModeUntil = 0;
let lastHandledExportId = "";

export function markLocalPaneAction(lockMs = 1800) {
  suppressPollModeUntil = Date.now() + lockMs;
}

function triggerBrowserExportDownload(format: "xlsx" | "png", kind: string) {
  if (typeof window === "undefined" || typeof document === "undefined") return;
  const host = window.location.hostname || "localhost";
  const url = `http://${host}:5000/api/export/${format}?kind=${encodeURIComponent(kind || "patient")}`;
  const a = document.createElement("a");
  a.href = url;
  a.download = `malcolm_${kind || "report"}.${format}`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
}

function applyState(s: RemoteState, fromPoll = false) {
  const pollModeLocked = fromPoll && Date.now() < suppressPollModeUntil;
  if (s.patient) malcolmStore.setPatient(s.patient);
  if (s.customPane && !pollModeLocked) malcolmStore.setCustomPane(s.customPane);
  if (s.wardHud) malcolmStore.setWardHud(s.wardHud);
  const current = malcolmStore.get();
  const prevMode = current.mode;
  const prevSeq = current.replySeq;
  if (s.mode && !pollModeLocked) malcolmStore.setMode(s.mode, s.bpm);
  else if (s.bpm !== undefined) malcolmStore.setBpm(s.bpm);
  if (s.clinicianCaption !== undefined) malcolmStore.setClinicianCaption(s.clinicianCaption);

  if (s.exportRequest && s.exportRequest.id !== lastHandledExportId) {
    lastHandledExportId = s.exportRequest.id;
    const tsMatch = /^exp_(\d+)$/.exec(s.exportRequest.id);
    const isFresh = !tsMatch || Date.now() - Number(tsMatch[1]) < 15000;
    if (isFresh) {
      const k = s.exportRequest.kind || s.customPane?.kind || current.customPane.kind || "patient";
      if (s.exportRequest.format === "xlsx" || s.exportRequest.format === "both") {
        triggerBrowserExportDownload("xlsx", k);
      }
      if (s.exportRequest.format === "png" || s.exportRequest.format === "both") {
        setTimeout(() => triggerBrowserExportDownload("png", k), s.exportRequest.format === "both" ? 350 : 0);
      }
    }
  }

  const hasNewReplySeq = s.replySeq !== undefined && s.replySeq !== prevSeq;
  if (s.replySeq !== undefined) {
    malcolmStore.setReplySeq(s.replySeq);
  }
  if (s.caption !== undefined) {
    // During background polling, only update caption when replySeq actually changed
    // (or on initial load when current.caption is still the default)
    if (!fromPoll || hasNewReplySeq || s.replySeq === undefined || current.replySeq === 0) {
      malcolmStore.setCaption(s.caption);
    }
  } else if (s.mode && s.mode !== prevMode) {
    malcolmStore.setCaption(DEFAULT_CAPTIONS[s.mode]);
  }
}

export function installMalcolmBridge(): () => void {
  let pollTimer: ReturnType<typeof setInterval> | undefined;
  let inFlight = false;
  let lastBody = "";

  const stopPolling = () => {
    clearInterval(pollTimer);
    pollTimer = undefined;
  };

  const startPolling = (url = DEFAULT_POLL_URL, intervalMs = DEFAULT_POLL_INTERVAL) => {
    stopPolling();
    const tick = async () => {
      if (inFlight) return;
      inFlight = true;
      try {
        const res = await fetch(url, { cache: "no-store" });
        if (!res.ok) return;
        const body = await res.text();
        // Only apply changes, so the demo controller isn't overridden every second.
        if (body === lastBody) return;
        lastBody = body;
        applyState(parseRemoteState(JSON.parse(body)), true);
      } catch {
        // Backend offline or bad JSON: stay quiet and keep the current state.
      } finally {
        inFlight = false;
      }
    };
    void tick();
    pollTimer = setInterval(tick, Math.max(250, intervalMs));
  };

  const bridge: MalcolmBridge = {
    setState: (stateName, data) => applyState(parseRemoteState({ state: stateName }, data)),
    apply: (payload) => applyState(parseRemoteState(payload)),
    startPolling,
    stopPolling,
    getState: () => malcolmStore.get(),
  };

  window.malcolm = bridge;
  window.setMalcolmState = bridge.setState;

  const onMessage = (e: MessageEvent) => {
    const d = e.data as { type?: unknown } | null;
    if (d && typeof d === "object" && d.type === "malcolm") bridge.apply(d);
  };
  window.addEventListener("message", onMessage);

  const params = new URLSearchParams(window.location.search);
  startPolling(
    params.get("poll") || DEFAULT_POLL_URL,
    Number(params.get("interval")) || DEFAULT_POLL_INTERVAL,
  );

  return () => {
    stopPolling();
    window.removeEventListener("message", onMessage);
    delete window.malcolm;
    delete window.setMalcolmState;
  };
}
