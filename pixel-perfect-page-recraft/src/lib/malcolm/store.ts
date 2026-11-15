import { useSyncExternalStore } from "react";
import {
  DEFAULT_BPM,
  DEFAULT_CUSTOM_PANE,
  DEFAULT_PATIENT,
  DEFAULT_WARD_HUD,
  clampBpm,
  normalizeMode,
  type CustomPaneConfig,
  type MalcolmMode,
  type PatientSummary,
  type WardHudData,
} from "./signal";

export interface MalcolmState {
  mode: MalcolmMode;
  bpm: number;
  caption: string;
  clinicianCaption: string;
  patient: PatientSummary;
  customPane: CustomPaneConfig;
  wardHud: WardHudData;
}

export const DEFAULT_CAPTIONS: Record<MalcolmMode, string> = {
  idle: "Listening for clinician request...",
  heart: "Heart rate is 90 bpm, steady and within normal range.",
  patient: "Here is the current summary for Mika Hämäläinen.",
};

let state: MalcolmState = {
  mode: "idle",
  bpm: DEFAULT_BPM,
  caption: DEFAULT_CAPTIONS.idle,
  clinicianCaption: "",
  patient: DEFAULT_PATIENT,
  customPane: DEFAULT_CUSTOM_PANE,
  wardHud: DEFAULT_WARD_HUD,
};
const listeners = new Set<() => void>();
let captionTimer: ReturnType<typeof setTimeout> | undefined;

function emit(next: Partial<MalcolmState>) {
  state = { ...state, ...next };
  listeners.forEach((l) => l());
}

export const malcolmStore = {
  get: () => state,
  subscribe(listener: () => void) {
    listeners.add(listener);
    return () => listeners.delete(listener);
  },
  setMode(mode: unknown, bpm?: unknown) {
    const b = clampBpm(bpm);
    emit({ mode: normalizeMode(mode), ...(b !== null ? { bpm: b } : {}) });
  },
  setBpm(bpm: unknown) {
    const b = clampBpm(bpm);
    if (b !== null) emit({ bpm: b });
  },
  setPatient(patient: Partial<PatientSummary>) {
    emit({ patient: { ...state.patient, ...patient } });
  },
  setCustomPane(customPane: Partial<CustomPaneConfig>) {
    emit({ customPane: { ...state.customPane, ...customPane } });
  },
  setWardHud(wardHud: WardHudData) {
    emit({ wardHud });
  },
  setClinicianCaption(text: unknown) {
    const clinicianCaption = typeof text === "string" ? text.trim() : "";
    emit({ clinicianCaption });
  },
  /** Shows a caption line. It stays until replaced, or fades after durationMs when given. */
  setCaption(text: unknown, durationMs?: number) {
    clearTimeout(captionTimer);
    const caption = typeof text === "string" ? text.trim() : "";
    emit({ caption });
    if (caption && durationMs && durationMs > 0) {
      captionTimer = setTimeout(() => emit({ caption: "" }), durationMs);
    }
  },
};

export function useMalcolm(): MalcolmState {
  return useSyncExternalStore(malcolmStore.subscribe, malcolmStore.get, malcolmStore.get);
}
