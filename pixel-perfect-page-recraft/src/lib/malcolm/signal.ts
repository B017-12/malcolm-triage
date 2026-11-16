// Pure, framework-free math for Malcolm's visuals and remote-state parsing.

export type MalcolmMode = "idle" | "heart" | "patient";

export const BPM_MIN = 30;
export const BPM_MAX = 220;
export const DEFAULT_BPM = 90;

const HEART_ALIASES = new Set([
  "heart",
  "heart-rate",
  "heart_rate",
  "heartrate",
  "hr",
  "ecg",
  "vitals",
]);

const PATIENT_ALIASES = new Set([
  "patient",
  "patient_stats",
  "patient-stats",
  "patientstats",
  "stats",
  "patient_status",
  "status",
  "summary",
]);

/** Accepts loose strings from Python / API payloads and maps them to a mode. */
export function normalizeMode(input: unknown): MalcolmMode {
  if (typeof input !== "string") return "idle";
  const key = input.trim().toLowerCase();
  if (HEART_ALIASES.has(key)) return "heart";
  if (PATIENT_ALIASES.has(key)) return "patient";
  return "idle";
}

/** Rounds and clamps a BPM value; returns null for anything non-numeric. */
export function clampBpm(input: unknown): number | null {
  const n = typeof input === "string" ? Number(input) : input;
  if (typeof n !== "number" || !Number.isFinite(n)) return null;
  return Math.round(Math.min(BPM_MAX, Math.max(BPM_MIN, n)));
}

/** Seconds per beat for a given BPM (90 BPM -> 0.667s, i.e. 1.5 beats/s). */
export function beatPeriod(bpm: number): number {
  return 60 / bpm;
}

/**
 * Angular PQRST keypoints over one beat (phase in [0, 1]). The R peak sits at phase 0/1
 * so it lines up with the ring's outward pulse. Drawn as straight segments for a crisp trace.
 */
export const ECG_KEYPOINTS: readonly (readonly [number, number])[] = [
  [0, 1], // R
  [0.02, -0.3], // S
  [0.045, 0],
  [0.17, 0],
  [0.25, 0.3], // T
  [0.33, 0],
  [0.8, 0],
  [0.845, 0.14], // P
  [0.89, 0],
  [0.955, 0],
  [0.972, -0.16], // Q
  [1, 1], // R (next beat)
];

/** ECG value at beat phase p (any real number), linearly interpolated between keypoints. */
export function ecgSample(p: number): number {
  const ph = ((p % 1) + 1) % 1;
  for (let i = 1; i < ECG_KEYPOINTS.length; i++) {
    const [x1, y1] = ECG_KEYPOINTS[i]!;
    if (ph <= x1) {
      const [x0, y0] = ECG_KEYPOINTS[i - 1]!;
      return y0 + ((y1 - y0) * (ph - x0)) / (x1 - x0 || 1);
    }
  }
  return 1;
}

/** Ring pulse envelope (lub-dub) at beat phase p in [0, 1), peak 1 at p = 0. */
export function beatEnvelope(p: number): number {
  const ph = ((p % 1) + 1) % 1;
  const lub = Math.exp(-ph * 9);
  const dub = 0.35 * Math.exp(-((ph - 0.2) ** 2) / 0.002);
  return Math.min(1, lub + dub);
}

export interface PatientSummary {
  name: string;
  patientId: string;
  wing: string;
  room: string;
  department: string;
  protocol: string;
  rhythm: string;
  telemetryDetail: string;
  nurseRounds: string;
  symptom: string;
  dispatchStatus: string;
}

export type CustomPaneKind =
  | "patient"
  | "ward_table"
  | "ward_stats"
  | "timetable"
  | "custom_html";

export interface CustomPaneConfig {
  kind: CustomPaneKind;
  title?: string;
  subtitle?: string;
  filterWing?: string;
  generatedHtml?: string;
}

export interface WardPatientRow {
  patientId: string;
  name: string;
  wing: string;
  room: string;
  condition: string;
  department: string;
  protocol: string;
  bpm: number;
  baselineBpm: number;
  wakeState: string;
  rhythm: string;
  waveState: string;
  acuity: "Normal" | "Elevated" | "Attention";
  symptom: string;
  nurseNote: string;
  nurseDispatched: boolean;
  checkInActive: boolean;
}

export interface WardWingBreakdown {
  wing: string;
  count: number;
  meanBpm: number;
  patients: string[];
}

export interface WardStats {
  totalPatients: number;
  meanBpm: number;
  minBpm: number;
  minBpmPatient: string;
  maxBpm: number;
  maxBpmPatient: string;
  bpmSpread: number;
  awakeCount: number;
  sleepingCount: number;
  symptomaticCount: number;
  tachycardiaCount: number;
  nurseDispatchedCount: number;
  wingBreakdown: WardWingBreakdown[];
  summaryLine: string;
}

export interface WardHudData {
  patients: WardPatientRow[];
  stats: WardStats;
}

export const DEFAULT_CUSTOM_PANE: CustomPaneConfig = {
  kind: "patient",
  title: "",
  subtitle: "",
  filterWing: "",
  generatedHtml: "",
};

export const DEFAULT_WARD_HUD: WardHudData = {
  patients: [
    {
      patientId: "1429",
      name: "Mika Hämäläinen",
      wing: "Wing A",
      room: "153",
      condition: "Dialysis",
      department: "Nephrology — Renal Care",
      protocol: "Routine Dialysis",
      bpm: 90,
      baselineBpm: 90,
      wakeState: "Awake",
      rhythm: "Normal Sinus Rhythm",
      waveState: "Awake / Normal Sinus Rhythm",
      acuity: "Normal",
      symptom: "None reported",
      nurseNote: "No comments from nurse rounds.",
      nurseDispatched: false,
      checkInActive: false,
    },
    {
      patientId: "1430",
      name: "Aino Virtanen",
      wing: "Wing B",
      room: "204",
      condition: "Post-Op Cardiac",
      department: "Cardiothoracic — Post-Op",
      protocol: "Mitral Valve Recovery",
      bpm: 76,
      baselineBpm: 76,
      wakeState: "Awake",
      rhythm: "Regular Brady-Normal",
      waveState: "Awake / Regular Brady-Normal",
      acuity: "Normal",
      symptom: "Mild incision site soreness",
      nurseNote: "06:30 round: Chest tube output minimal; pain controlled on oral regimen.",
      nurseDispatched: false,
      checkInActive: false,
    },
    {
      patientId: "1431",
      name: "Jukka Korhonen",
      wing: "Wing C",
      room: "312",
      condition: "Respiratory Observation",
      department: "Pulmonology — Acute Care",
      protocol: "Pneumonia O2 Titration",
      bpm: 104,
      baselineBpm: 104,
      wakeState: "Awake",
      rhythm: "Mild Sinus Tachycardia",
      waveState: "Awake / Mild Sinus Tachycardia",
      acuity: "Elevated",
      symptom: "Shortness of breath when lying flat, dry cough",
      nurseNote: "07:00 round: SpO2 94% on 2L nasal cannula; IV ceftriaxone administered.",
      nurseDispatched: false,
      checkInActive: false,
    },
    {
      patientId: "1432",
      name: "Sanna Laine",
      wing: "Wing A",
      room: "118",
      condition: "Neurological Observation",
      department: "Neurology — Inpatient",
      protocol: "Post-Concussion Monitoring",
      bpm: 64,
      baselineBpm: 64,
      wakeState: "Sleeping",
      rhythm: "Deep Slow-Wave Pattern",
      waveState: "Sleeping / Deep Slow-Wave Pattern",
      acuity: "Attention",
      symptom: "Intermittent photophobia earlier this evening",
      nurseNote: "07:15 round: Pupils equal and reactive; resting comfortably in darkened room.",
      nurseDispatched: false,
      checkInActive: false,
    },
  ],
  stats: {
    totalPatients: 4,
    meanBpm: 83.5,
    minBpm: 64,
    minBpmPatient: "Sanna Laine (1432)",
    maxBpm: 104,
    maxBpmPatient: "Jukka Korhonen (1431)",
    bpmSpread: 40,
    awakeCount: 3,
    sleepingCount: 1,
    symptomaticCount: 3,
    tachycardiaCount: 1,
    nurseDispatchedCount: 0,
    wingBreakdown: [
      { wing: "Wing A", count: 2, meanBpm: 77.0, patients: ["Mika Hämäläinen", "Sanna Laine"] },
      { wing: "Wing B", count: 1, meanBpm: 76.0, patients: ["Aino Virtanen"] },
      { wing: "Wing C", count: 1, meanBpm: 104.0, patients: ["Jukka Korhonen"] },
    ],
    summaryLine:
      "4 Monitored Patients • Mean HR 83.5 BPM (Range 64–104 BPM) • 3 Awake / 1 Sleeping • 1 Elevated HR Alert",
  },
};

export const DEFAULT_PATIENT: PatientSummary = {
  name: "Mika Hämäläinen",
  patientId: "1429",
  wing: "Wing A",
  room: "153",
  department: "Nephrology — Renal Care",
  protocol: "Routine Dialysis",
  rhythm: "Normal Sinus Rhythm",
  telemetryDetail: "Awake • Baseline 90 BPM • Signal Locked",
  nurseRounds: "No comments from nurse rounds.",
  symptom: "Patient has not reported any discomfort or odd symptoms thus far.",
  dispatchStatus: "",
};

export interface ExportRequestPayload {
  id: string;
  format: "xlsx" | "png" | "both";
  kind: string;
}

export interface RemoteState {
  mode?: MalcolmMode;
  bpm?: number;
  caption?: string;
  clinicianCaption?: string;
  replySeq?: number;
  patient?: Partial<PatientSummary>;
  customPane?: CustomPaneConfig;
  wardHud?: WardHudData;
  exportRequest?: ExportRequestPayload;
}

const PATIENT_KEYS: Record<keyof PatientSummary, string[]> = {
  name: ["name", "patient_name", "patientName"],
  patientId: ["patientId", "patient_id", "id"],
  wing: ["wing"],
  room: ["room", "room_number", "roomNumber"],
  department: ["department"],
  protocol: ["protocol"],
  rhythm: ["rhythm"],
  telemetryDetail: ["telemetryDetail", "telemetry_detail"],
  nurseRounds: ["nurseRounds", "nurse_rounds"],
  symptom: ["symptom", "patientReport", "patient_report", "report"],
  dispatchStatus: ["dispatchStatus", "dispatch_status", "reportStatus", "report_status"],
};

function pickString(src: Record<string, unknown>, keys: string[]): string | undefined {
  for (const k of keys) {
    const v = src[k];
    if (typeof v === "string") return v;
    if (typeof v === "number" && Number.isFinite(v)) return String(v);
  }
  return undefined;
}

function asRecord(v: unknown): Record<string, unknown> | undefined {
  return v && typeof v === "object" && !Array.isArray(v) ? (v as Record<string, unknown>) : undefined;
}

/**
 * Parses an untrusted payload. Accepts { state, data } (preferred), or flat keys:
 * { state: "patient_stats", data: { bpm, transcript, patient: { name, ... } } }
 */
export function parseRemoteState(raw: unknown, data?: unknown): RemoteState {
  const top = asRecord(raw);
  if (!top) return {};
  const merged: Record<string, unknown> = { ...top, ...asRecord(top["data"]), ...asRecord(data) };
  const out: RemoteState = {};

  const modeRaw = top["state"] ?? top["mode"];
  if (modeRaw !== undefined) out.mode = normalizeMode(modeRaw);

  const bpm = clampBpm(merged["bpm"] ?? merged["heart_rate"] ?? merged["heartRate"]);
  if (bpm !== null) out.bpm = bpm;

  const caption = merged["transcript"] ?? merged["caption"] ?? merged["text"];
  if (typeof caption === "string") out.caption = caption;

  const clinicianCaption = merged["clinicianTranscript"] ?? merged["clinician_transcript"];
  if (typeof clinicianCaption === "string") out.clinicianCaption = clinicianCaption;

  const rawReplySeq = merged["replySeq"] ?? merged["reply_seq"];
  if (typeof rawReplySeq === "number" && Number.isFinite(rawReplySeq)) {
    out.replySeq = rawReplySeq;
  }

  const p = { ...merged, ...asRecord(merged["patient"]) };
  const patient: Partial<PatientSummary> = {};
  (Object.keys(PATIENT_KEYS) as (keyof PatientSummary)[]).forEach((key) => {
    const v = pickString(p, PATIENT_KEYS[key]);
    if (v !== undefined) patient[key] = v;
  });
  // Top-level "id"/"status" mean something else when no patient object was sent.
  if (!asRecord(merged["patient"])) {
    if (pickString(merged, ["patientId", "patient_id"]) === undefined) delete patient.patientId;
  }
  if (Object.keys(patient).length) out.patient = patient;

  const rawPane = asRecord(merged["customPane"] ?? merged["custom_pane"]);
  if (rawPane && typeof rawPane["kind"] === "string") {
    const k = rawPane["kind"].trim().toLowerCase();
    const kind: CustomPaneKind =
      k === "ward_table" ||
      k === "ward_stats" ||
      k === "timetable" ||
      k === "custom_html"
        ? k
        : "patient";
    out.customPane = {
      kind,
      title: typeof rawPane["title"] === "string" ? rawPane["title"] : "",
      subtitle: typeof rawPane["subtitle"] === "string" ? rawPane["subtitle"] : "",
      filterWing: typeof rawPane["filterWing"] === "string" ? rawPane["filterWing"] : "",
      generatedHtml:
        typeof rawPane["generatedHtml"] === "string" ? rawPane["generatedHtml"] : "",
    };
  }

  const rawHud = asRecord(merged["wardHud"] ?? merged["ward_hud"]);
  if (rawHud && Array.isArray(rawHud["patients"]) && asRecord(rawHud["stats"])) {
    out.wardHud = rawHud as unknown as WardHudData;
  }

  const rawExp = asRecord(merged["exportRequest"] ?? merged["export_request"]);
  if (rawExp && typeof rawExp["id"] === "string" && typeof rawExp["format"] === "string") {
    const f = rawExp["format"].toLowerCase();
    if (f === "xlsx" || f === "png" || f === "both") {
      out.exportRequest = {
        id: rawExp["id"],
        format: f,
        kind: typeof rawExp["kind"] === "string" ? rawExp["kind"] : "patient",
      };
    }
  }

  return out;
}
