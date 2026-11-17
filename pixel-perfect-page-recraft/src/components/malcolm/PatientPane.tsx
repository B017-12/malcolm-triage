import { useEffect, useState, type ReactNode } from "react";
import { markLocalPaneAction } from "@/lib/malcolm/bridge";
import {
  DEFAULT_WARD_HUD,
  parseRemoteState,
  type CustomPaneKind,
  type PatientSummary,
  type WardPatientRow,
} from "@/lib/malcolm/signal";
import { malcolmStore, useMalcolm } from "@/lib/malcolm/store";

// Sizes scale with viewport height and width so all zones always fit without scrolling.
const LABEL = "font-mono text-xs uppercase tracking-[0.18em] text-label";
const VALUE_XL = "text-[clamp(1.25rem,min(4.4vh,2.6vw),2.25rem)] leading-tight";
const SUB = "text-[clamp(0.95rem,min(3vh,1.8vw),1.5rem)] font-light leading-snug text-text-soft";
const PAD_X = "px-[clamp(1.5rem,3.2vw,3rem)]";

let lastOpenPaneKind: CustomPaneKind = "patient";

export function getLastOpenPaneKind(): CustomPaneKind {
  return lastOpenPaneKind;
}

function Zone({ className = "", children }: { className?: string; children: ReactNode }) {
  return (
    <div className={`flex min-h-0 flex-col justify-center border-b border-hairline ${className}`}>
      {children}
    </div>
  );
}

export async function syncPaneBackend(action: CustomPaneKind | "close", extra?: Record<string, string>) {
  markLocalPaneAction(1800);
  if (action === "close") {
    const currentKind = malcolmStore.get().customPane?.kind;
    if (currentKind) {
      lastOpenPaneKind = currentKind;
    }
    malcolmStore.setMode("idle");
  } else {
    lastOpenPaneKind = action;
    malcolmStore.setMode("patient");
    malcolmStore.setCustomPane({
      kind: action,
      ...(action === "patient" ? { generatedHtml: "" } : {}),
      ...(extra?.filterWing !== undefined ? { filterWing: extra.filterWing } : {}),
    });
  }
  try {
    const host = typeof window !== "undefined" && window.location.hostname ? window.location.hostname : "localhost";
    const res = await fetch(`http://${host}:5000/api/pane`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ action, ...extra }),
    });
    if (res.ok) {
      const payload = await res.json();
      const parsed = parseRemoteState(payload);
      if (parsed.patient) malcolmStore.setPatient(parsed.patient);
      if (action !== "close" && parsed.customPane) malcolmStore.setCustomPane(parsed.customPane);
      if (parsed.wardHud) malcolmStore.setWardHud(parsed.wardHud);
    }
  } catch {
    // Offline or standalone UI mode: local optimistic store update already applied.
  }
}

async function selectPatientFromRow(row: WardPatientRow) {
  malcolmStore.setPatient({
    patientId: row.patientId,
    name: row.name,
    wing: row.wing,
    room: row.room,
    department: row.department,
    protocol: row.protocol,
    rhythm: row.rhythm,
    telemetryDetail: `${row.wakeState} • Baseline ${row.baselineBpm} BPM • Signal Locked`,
    nurseRounds: row.nurseNote,
    symptom: row.symptom,
    dispatchStatus: row.nurseDispatched ? `Nurse dispatched to Room ${row.room}` : "",
  });
  malcolmStore.setBpm(row.bpm);
  malcolmStore.setCustomPane({ kind: "patient", title: "", subtitle: "", filterWing: "" });
  malcolmStore.setMode("patient");
  try {
    const res = await fetch("http://localhost:5000/api/select_patient", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ patient_id: row.patientId }),
    });
    if (res.ok) {
      const payload = await res.json();
      const parsed = parseRemoteState(payload);
      if (parsed.patient) malcolmStore.setPatient(parsed.patient);
      if (parsed.customPane) malcolmStore.setCustomPane(parsed.customPane);
      if (parsed.wardHud) malcolmStore.setWardHud(parsed.wardHud);
      if (parsed.bpm) malcolmStore.setBpm(parsed.bpm);
    }
  } catch {
    // Keep optimistic patient view.
  }
}

function playBrowserPaChime() {
  if (typeof window === "undefined") return;
  try {
    const AudioCtx = window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
    const ctx = new AudioCtx();
    const notes = [
      { freq: 392.0, start: 0.0, dur: 0.34 },
      { freq: 523.25, start: 0.34, dur: 0.34 },
      { freq: 659.25, start: 0.68, dur: 0.65 },
    ];
    for (const n of notes) {
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      osc.type = "sine";
      osc.frequency.setValueAtTime(n.freq, ctx.currentTime + n.start);
      gain.gain.setValueAtTime(0.001, ctx.currentTime + n.start);
      gain.gain.exponentialRampToValueAtTime(0.22, ctx.currentTime + n.start + 0.02);
      gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + n.start + n.dur);
      osc.connect(gain);
      gain.connect(ctx.destination);
      osc.start(ctx.currentTime + n.start);
      osc.stop(ctx.currentTime + n.start + n.dur + 0.02);
    }
  } catch {
    // Ignore audio context errors
  }
}

async function triggerLoudspeakerPage(eventId?: string) {
  playBrowserPaChime();
  try {
    const res = await fetch("http://localhost:5000/api/calendar/test_page", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ event_id: eventId || "" }),
    });
    if (res.ok) {
      const payload = await res.json();
      const parsed = parseRemoteState(payload);
      if (parsed.caption) malcolmStore.setCaption(parsed.caption);
    }
  } catch {
    // Backend offline
  }
}

function triggerExportDownload(format: "xlsx" | "png", activeKind: CustomPaneKind) {
  const url = `http://localhost:5000/api/export/${format}?kind=${encodeURIComponent(activeKind)}`;
  const a = document.createElement("a");
  a.href = url;
  a.download = `malcolm_${activeKind}.${format}`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
}

function PaneTopNav({ activeKind }: { activeKind: CustomPaneKind }) {
  const tabs: { kind: CustomPaneKind; label: string }[] = [
    { kind: "patient", label: "Patient Chart" },
    { kind: "ward_table", label: "Ward Table" },
    { kind: "ward_stats", label: "Statistics" },
    { kind: "timetable", label: "Timetable" },
  ];

  return (
    <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
      <div className="flex flex-wrap items-center gap-1.5">
        {tabs.map((t) => {
          const active = activeKind === t.kind;
          return (
            <button
              key={t.kind}
              type="button"
              onClick={() => void syncPaneBackend(t.kind)}
              className={`border px-2.5 py-1 font-mono text-[11px] uppercase tracking-[0.18em] transition-colors ${
                active
                  ? "border-telemetry/80 bg-telemetry/15 text-telemetry"
                  : "border-hairline bg-background/40 text-pane-meta hover:border-foreground/40 hover:text-foreground"
              }`}
            >
              {t.label}
            </button>
          );
        })}
      </div>
      <div className="flex flex-wrap items-center gap-1.5">
        <button
          type="button"
          onClick={() => triggerExportDownload("xlsx", activeKind)}
          title="Export formatted Excel (.xlsx) on a plain white background"
          className="border border-hairline bg-background/50 px-2.5 py-1 font-mono text-[11px] uppercase tracking-[0.16em] text-pane-meta transition-colors hover:border-telemetry hover:bg-telemetry/15 hover:text-telemetry"
        >
          ⬇ .XLSX
        </button>
        <button
          type="button"
          onClick={() => triggerExportDownload("png", activeKind)}
          title="Export formatted PNG image on a plain white background"
          className="border border-hairline bg-background/50 px-2.5 py-1 font-mono text-[11px] uppercase tracking-[0.16em] text-pane-meta transition-colors hover:border-telemetry hover:bg-telemetry/15 hover:text-telemetry"
        >
          ⬇ .PNG
        </button>
        <button
          type="button"
          onClick={() => void syncPaneBackend("close")}
          title="Close split-screen pane"
          className="border border-hairline bg-background/50 px-2.5 py-1 font-mono text-[11px] uppercase tracking-[0.18em] text-pane-meta transition-colors hover:border-rose hover:bg-rose/15 hover:text-rose-soft"
        >
          × Close Pane
        </button>
      </div>
    </div>
  );
}

async function deletePatientFromWard(patientId: string, e: React.MouseEvent) {
  e.stopPropagation();
  try {
    const host = typeof window !== "undefined" && window.location.hostname ? window.location.hostname : "localhost";
    const res = await fetch(`http://${host}:5000/api/patients/delete`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ patient_id: patientId }),
    });
    if (res.ok) {
      const payload = await res.json();
      window.malcolm?.apply(payload);
    }
  } catch {
    // Ignore offline error
  }
}

function WardTableView() {
  const { customPane, wardHud, patient } = useMalcolm();
  const hud = wardHud?.patients?.length ? wardHud : DEFAULT_WARD_HUD;
  const filterWing = (customPane.filterWing || "").trim();
  const rows = filterWing
    ? hud.patients.filter((p) => p.wing.toLowerCase() === filterWing.toLowerCase())
    : hud.patients;

  const [showAddForm, setShowAddForm] = useState(false);
  const [newName, setNewName] = useState("");
  const [newWing, setNewWing] = useState("Wing B");
  const [newRoom, setNewRoom] = useState("210");
  const [newCondition, setNewCondition] = useState("Cardiac Observation");
  const [newBpm, setNewBpm] = useState("84");

  const handleAddPatient = async (e: React.FormEvent) => {
    e.preventDefault();
    try {
      const host = typeof window !== "undefined" && window.location.hostname ? window.location.hostname : "localhost";
      const res = await fetch(`http://${host}:5000/api/patients/add`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          name: newName.trim() || "Alex Rivera",
          wing: newWing,
          room: `Room ${newRoom.replace(/^room\s*/i, "").trim() || "210"}`,
          condition: newCondition.trim() || "General Clinical Observation",
          baseline_bpm: Number(newBpm) || 84,
        }),
      });
      if (res.ok) {
        const payload = await res.json();
        window.malcolm?.apply(payload);
        setNewName("");
        setShowAddForm(false);
      }
    } catch {
      // Ignore offline error
    }
  };

  const wings = ["", "Wing A", "Wing B", "Wing C"];

  return (
    <div className="grid h-full grid-rows-[minmax(min-content,20fr)_minmax(min-content,64fr)_minmax(min-content,16fr)] text-foreground">
      {/* Zone 1 — Table Header & Wing Filter */}
      <Zone className={`${PAD_X} pt-[clamp(0.75rem,2.5vh,2rem)]`}>
        <PaneTopNav activeKind="ward_table" />
        <div className="flex flex-wrap items-end justify-between gap-4">
          <div>
            <p className="font-mono text-xs uppercase tracking-[0.2em] text-pane-meta">
              {customPane.subtitle || "Real-Time Multi-Patient EHR & Telemetry Matrix"}
            </p>
            <h2 className="mt-1 font-sans text-[clamp(1.6rem,min(5vh,3.2vw),2.75rem)] font-medium leading-none tracking-tight">
              {customPane.title || "Active Ward Roster — All Patients"}
            </h2>
          </div>
          <div className="flex flex-wrap items-center gap-1.5">
            {wings.map((w) => {
              const isSelected = filterWing.toLowerCase() === w.toLowerCase();
              return (
                <button
                  key={w || "all"}
                  type="button"
                  onClick={() => void syncPaneBackend("ward_table", { filterWing: w })}
                  className={`border px-2.5 py-1 font-mono text-[11px] uppercase tracking-[0.16em] transition-colors ${
                    isSelected
                      ? "border-foreground/70 bg-foreground/15 text-foreground"
                      : "border-hairline text-pane-meta hover:border-foreground/40 hover:text-foreground"
                  }`}
                >
                  {w || `All (${hud.patients.length})`}
                </button>
              );
            })}
            <button
              type="button"
              onClick={() => setShowAddForm((v) => !v)}
              className="border border-telemetry/70 bg-telemetry/15 px-2.5 py-1 font-mono text-[11px] uppercase tracking-[0.16em] text-telemetry transition-colors hover:bg-telemetry/25"
            >
              {showAddForm ? "× Cancel" : "+ Admit Patient"}
            </button>
          </div>
        </div>
        {showAddForm ? (
          <form onSubmit={(e) => void handleAddPatient(e)} className="mt-3 flex flex-wrap items-center gap-2 border border-hairline bg-background/60 p-2.5">
            <input
              type="text"
              value={newName}
              onChange={(e) => setNewName(e.target.value)}
              placeholder="Patient Name (e.g. Alex Rivera)"
              className="border border-hairline bg-background px-2.5 py-1 text-xs text-foreground outline-none focus:border-telemetry"
            />
            <select
              value={newWing}
              onChange={(e) => setNewWing(e.target.value)}
              className="border border-hairline bg-background px-2 py-1 font-mono text-xs text-foreground outline-none"
            >
              <option value="Wing A">Wing A</option>
              <option value="Wing B">Wing B</option>
              <option value="Wing C">Wing C</option>
            </select>
            <input
              type="text"
              value={newRoom}
              onChange={(e) => setNewRoom(e.target.value)}
              placeholder="Room #"
              className="w-20 border border-hairline bg-background px-2 py-1 font-mono text-xs text-foreground outline-none"
            />
            <input
              type="text"
              value={newCondition}
              onChange={(e) => setNewCondition(e.target.value)}
              placeholder="Condition"
              className="border border-hairline bg-background px-2.5 py-1 text-xs text-foreground outline-none"
            />
            <input
              type="number"
              value={newBpm}
              onChange={(e) => setNewBpm(e.target.value)}
              min={45}
              max={180}
              className="w-16 border border-hairline bg-background px-2 py-1 font-mono text-xs text-foreground outline-none"
            />
            <button
              type="submit"
              className="border border-telemetry bg-telemetry/20 px-3 py-1 font-mono text-[11px] font-semibold uppercase tracking-[0.14em] text-telemetry hover:bg-telemetry/30"
            >
              Save to Ward DB
            </button>
          </form>
        ) : null}
      </Zone>

      {/* Zone 2 — All-Patients Clinical Roster Table */}
      <div className={`flex min-h-0 flex-col justify-center border-b border-hairline ${PAD_X} py-2`}>
        <div className="w-full overflow-hidden border border-hairline">
          <table className="w-full border-collapse text-left">
            <thead>
              <tr className="border-b border-hairline bg-foreground/[0.03] font-mono text-[11px] uppercase tracking-[0.18em] text-label">
                <th className="py-2.5 pl-4 pr-3">Patient</th>
                <th className="px-3 py-2.5">Location</th>
                <th className="px-3 py-2.5">Department &amp; Protocol</th>
                <th className="px-3 py-2.5">Live Telemetry</th>
                <th className="py-2.5 pl-3 pr-2">Symptoms &amp; Round Status</th>
                <th className="py-2.5 pl-2 pr-4 text-right">Action</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-hairline">
              {rows.map((row) => {
                const isSelected = row.patientId === patient.patientId;
                const isElevated = row.bpm >= 100 || row.acuity === "Elevated";
                const isSleeping = row.wakeState.toLowerCase().includes("sleep");
                return (
                  <tr
                    key={row.patientId}
                    onClick={() => void selectPatientFromRow(row)}
                    className={`cursor-pointer transition-colors hover:bg-foreground/[0.06] ${
                      isSelected ? "bg-telemetry/[0.07]" : ""
                    }`}
                  >
                    <td className="py-3 pl-4 pr-3 align-top">
                      <div className="flex items-center gap-2">
                        <span className="font-mono text-xs font-semibold uppercase tracking-[0.14em] text-pane-meta">
                          #{row.patientId}
                        </span>
                        {isSelected ? (
                          <span className="border border-telemetry/60 bg-telemetry/15 px-1.5 py-0.5 font-mono text-[10px] uppercase tracking-[0.14em] text-telemetry">
                            Active
                          </span>
                        ) : null}
                      </div>
                      <p className="mt-0.5 font-sans text-[clamp(1rem,1.45vw,1.35rem)] font-medium leading-tight text-foreground">
                        {row.name}
                      </p>
                    </td>
                    <td className="px-3 py-3 align-top font-mono text-xs uppercase tracking-[0.14em] text-text-soft">
                      <div>{row.wing}</div>
                      <div className="mt-0.5 text-foreground">Room {row.room}</div>
                    </td>
                    <td className="px-3 py-3 align-top">
                      <p className="font-sans text-sm font-medium leading-snug text-foreground">
                        {row.department}
                      </p>
                      <p className="mt-0.5 text-xs text-text-soft">{row.protocol}</p>
                    </td>
                    <td className="px-3 py-3 align-top">
                      <div
                        className={`font-mono text-sm font-semibold tracking-tight ${
                          isElevated ? "text-rose-soft" : "text-telemetry"
                        }`}
                      >
                        {row.bpm} BPM
                      </div>
                      <div className="mt-0.5 font-mono text-[11px] uppercase tracking-[0.12em] text-pane-meta">
                        {isSleeping ? "Sleeping" : "Awake"} • {row.rhythm}
                      </div>
                    </td>
                    <td className="py-3 pl-3 pr-2 align-top">
                      <p className="line-clamp-2 text-xs leading-snug text-text-bright">
                        {row.symptom}
                      </p>
                      <p className="mt-1 line-clamp-1 font-mono text-[11px] text-pane-meta">
                        {row.nurseNote}
                      </p>
                    </td>
                    <td className="py-3 pl-2 pr-4 text-right align-top">
                      <button
                        type="button"
                        onClick={(e) => void deletePatientFromWard(row.patientId, e)}
                        title={`Discharge / remove Patient #${row.patientId}`}
                        className="border border-hairline bg-background/50 px-2 py-1 font-mono text-[10px] uppercase tracking-[0.12em] text-pane-meta transition-colors hover:border-rose hover:bg-rose/20 hover:text-rose-soft"
                      >
                        Discharge ✕
                      </button>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Zone 3 — Footer summary & quick action hint */}
      <div
        className={`flex min-h-0 flex-wrap items-center justify-between gap-4 border-t-2 border-rose bg-alert-tint/25 ${PAD_X} py-3`}
      >
        <div>
          <p className="font-mono text-xs uppercase tracking-[0.18em] text-rose-soft">
            Live Ward Summary
          </p>
          <p className="mt-0.5 text-sm font-normal text-foreground">
            {hud.stats.summaryLine}
          </p>
        </div>
        <span className="font-mono text-xs uppercase tracking-[0.16em] text-pane-meta">
          Click any patient row to open full chart
        </span>
      </div>
    </div>
  );
}

function WardStatsView() {
  const { customPane, wardHud } = useMalcolm();
  const hud = wardHud?.patients?.length ? wardHud : DEFAULT_WARD_HUD;
  const stats = hud.stats;

  return (
    <div className="grid h-full grid-rows-[minmax(min-content,22fr)_minmax(min-content,36fr)_minmax(min-content,24fr)_minmax(min-content,18fr)] text-foreground">
      {/* Zone 1 — Header & Aggregate KPI Strip */}
      <Zone className={`${PAD_X} pt-[clamp(0.75rem,2.5vh,2rem)]`}>
        <PaneTopNav activeKind="ward_stats" />
        <p className="font-mono text-xs uppercase tracking-[0.2em] text-pane-meta">
          {customPane.subtitle || "Live Statistical Breakdown • 4 Monitored Patients"}
        </p>
        <h2 className="mt-1 font-sans text-[clamp(1.6rem,min(4.8vh,3.1vw),2.6rem)] font-medium leading-none tracking-tight">
          {customPane.title || "Ward Telemetry & Acuity Statistics"}
        </h2>
        <div className="mt-3 grid grid-cols-4 gap-3">
          <div className="border border-hairline bg-foreground/[0.02] px-3.5 py-2">
            <p className="font-mono text-[10px] uppercase tracking-[0.18em] text-label">
              Mean Heart Rate
            </p>
            <p className="mt-1 font-mono text-xl font-semibold text-telemetry">
              {stats.meanBpm} BPM
            </p>
            <p className="text-[11px] text-text-soft">Across {stats.totalPatients} active beds</p>
          </div>
          <div className="border border-hairline bg-foreground/[0.02] px-3.5 py-2">
            <p className="font-mono text-[10px] uppercase tracking-[0.18em] text-label">
              BPM Range &amp; Spread
            </p>
            <p className="mt-1 font-mono text-xl font-semibold text-foreground">
              {stats.minBpm}–{stats.maxBpm} BPM
            </p>
            <p className="text-[11px] text-text-soft">Spread Δ {stats.bpmSpread} BPM</p>
          </div>
          <div className="border border-hairline bg-foreground/[0.02] px-3.5 py-2">
            <p className="font-mono text-[10px] uppercase tracking-[0.18em] text-label">
              Wakefulness State
            </p>
            <p className="mt-1 font-mono text-xl font-semibold text-foreground">
              {stats.awakeCount} Awake / {stats.sleepingCount} Sleep
            </p>
            <p className="text-[11px] text-text-soft">Live EEG/ECG wave lock</p>
          </div>
          <div className="border border-hairline bg-foreground/[0.02] px-3.5 py-2">
            <p className="font-mono text-[10px] uppercase tracking-[0.18em] text-label">
              Acuity &amp; Triage
            </p>
            <p className="mt-1 font-mono text-xl font-semibold text-rose-soft">
              {stats.tachycardiaCount} Elevated HR
            </p>
            <p className="text-[11px] text-text-soft">
              {stats.symptomaticCount} patients w/ logged symptoms
            </p>
          </div>
        </div>
      </Zone>

      {/* Zone 2 — Per-Patient BPM Distribution Bars */}
      <Zone className={PAD_X}>
        <div className="flex items-center justify-between">
          <p className={LABEL}>Live Heart Rate Distribution by Patient (Click to Inspect)</p>
          <span className="font-mono text-[11px] uppercase tracking-[0.16em] text-pane-meta">
            Scale: 40 — 140 BPM
          </span>
        </div>
        <div className="mt-2.5 grid grid-cols-1 gap-2">
          {hud.patients.map((p) => {
            const pct = Math.max(12, Math.min(100, Math.round(((p.bpm - 40) / 100) * 100)));
            const isHigh = p.bpm >= 100;
            return (
              <button
                key={p.patientId}
                type="button"
                onClick={() => void selectPatientFromRow(p)}
                className="group flex items-center gap-4 border border-hairline bg-foreground/[0.02] px-3.5 py-2 text-left transition-colors hover:border-foreground/40 hover:bg-foreground/[0.05]"
              >
                <div className="w-48 shrink-0">
                  <div className="flex items-center gap-2">
                    <span className="font-mono text-xs text-pane-meta">#{p.patientId}</span>
                    <span className="truncate font-sans text-sm font-medium text-foreground">
                      {p.name}
                    </span>
                  </div>
                  <p className="font-mono text-[10px] uppercase tracking-[0.14em] text-text-soft">
                    {p.wing} • Rm {p.room} • {p.condition}
                  </p>
                </div>
                <div className="flex-1">
                  <div className="h-2.5 w-full overflow-hidden bg-foreground/10">
                    <div
                      className={`h-full transition-all duration-500 ${
                        isHigh ? "bg-rose" : "bg-telemetry"
                      }`}
                      style={{ width: `${pct}%` }}
                    />
                  </div>
                  <div className="mt-1 flex justify-between font-mono text-[10px] uppercase tracking-[0.14em] text-pane-meta">
                    <span>{p.wakeState} • {p.rhythm}</span>
                    <span>Baseline {p.baselineBpm} BPM</span>
                  </div>
                </div>
                <div
                  className={`w-24 shrink-0 text-right font-mono text-base font-semibold ${
                    isHigh ? "text-rose-soft" : "text-telemetry"
                  }`}
                >
                  {p.bpm} BPM
                </div>
              </button>
            );
          })}
        </div>
      </Zone>

      {/* Zone 3 — Wing Breakdown */}
      <Zone className={PAD_X}>
        <p className={LABEL}>Hospital Wing Census &amp; Mean Telemetry</p>
        <div className="mt-2.5 grid grid-cols-3 gap-4">
          {stats.wingBreakdown.map((w) => (
            <button
              key={w.wing}
              type="button"
              onClick={() => void syncPaneBackend("ward_table", { filterWing: w.wing })}
              className="border border-hairline bg-foreground/[0.02] p-3 text-left transition-colors hover:border-foreground/40 hover:bg-foreground/[0.05]"
            >
              <div className="flex items-baseline justify-between">
                <span className="font-mono text-xs font-semibold uppercase tracking-[0.18em] text-foreground">
                  {w.wing}
                </span>
                <span className="font-mono text-xs text-pane-meta">
                  {w.count} {w.count === 1 ? "Patient" : "Patients"}
                </span>
              </div>
              <p className="mt-1.5 font-mono text-lg font-medium text-telemetry">
                Mean {w.meanBpm} BPM
              </p>
              <p className="mt-0.5 truncate text-xs text-text-soft">
                {w.patients.join(", ")}
              </p>
            </button>
          ))}
        </div>
      </Zone>

      {/* Zone 4 — Outliers & Priority Clinical Focus */}
      <div
        className={`flex min-h-0 flex-col justify-center gap-1.5 border-t-2 border-rose bg-alert-tint/30 ${PAD_X} py-3`}
      >
        <p className="font-mono text-xs uppercase tracking-[0.2em] text-rose-soft">
          Statistical Outliers &amp; Priority Triage Focus
        </p>
        <p className="text-[clamp(1rem,min(3.2vh,1.9vw),1.45rem)] font-normal leading-snug text-foreground">
          Peak HR: <strong>{stats.maxBpmPatient}</strong> at <strong>{stats.maxBpm} BPM</strong> (Mild Sinus Tachycardia) • Lowest HR: <strong>{stats.minBpmPatient}</strong> at <strong>{stats.minBpm} BPM</strong> (Sleeping / Post-Concussion).
        </p>
      </div>
    </div>
  );
}

function buildGeneratedIframeDoc(rawHtmlAndCss: string): string {
  return `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<style>
  *, *::before, *::after { box-sizing: border-box; }
  html, body {
    margin: 0;
    padding: 0;
    width: 100%;
    min-height: 100%;
    background: transparent;
    color: #f3f5f8;
    font-family: 'Inter', system-ui, -apple-system, sans-serif;
    overflow-x: hidden;
    overflow-y: auto;
    -webkit-font-smoothing: antialiased;
  }
  .hud-root {
    min-height: 135vh;
    width: 100%;
    box-sizing: border-box;
    padding: 18px 28px 64px 28px;
    display: flex;
    flex-direction: column;
    justify-content: flex-start;
    gap: 18px;
    color: #f3f5f8;
    font-family: 'Inter', system-ui, sans-serif;
  }
  .hud-kicker {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.2em;
    color: #8b95a7;
    margin: 0;
  }
  .hud-title {
    font-size: clamp(1.5rem, 2.6vw, 2.35rem);
    font-weight: 500;
    letter-spacing: -0.02em;
    margin: 4px 0 0 0;
    color: #ffffff;
  }
  .kpi-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
  }
  .kpi-card {
    border: 1px solid rgba(255, 255, 255, 0.12);
    background: rgba(255, 255, 255, 0.025);
    padding: 12px 14px;
  }
  .kpi-val {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.35rem;
    font-weight: 600;
    color: #38bdf8;
    margin: 6px 0 2px 0;
  }
  .kpi-val.alert { color: #fb7185; }
  .kpi-sub { font-size: 11px; color: #8b95a7; margin: 0; }
  .hud-table {
    width: 100%;
    border-collapse: collapse;
    border: 1px solid rgba(255, 255, 255, 0.12);
    font-size: 13px;
  }
  .hud-table th {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 0.16em;
    color: #8b95a7;
    text-align: left;
    padding: 10px 12px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.12);
    background: rgba(255, 255, 255, 0.03);
  }
  .hud-table td {
    padding: 11px 12px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.08);
    vertical-align: top;
  }
  .hud-table tr:hover {
    background: rgba(56, 189, 248, 0.07);
    cursor: pointer;
  }
  .bpm-pill {
    font-family: 'JetBrains Mono', monospace;
    font-weight: 600;
    color: #38bdf8;
  }
  .bpm-pill.high { color: #fb7185; }
  .bar-row {
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 9px 12px;
    border: 1px solid rgba(255, 255, 255, 0.1);
    background: rgba(255, 255, 255, 0.02);
    cursor: pointer;
  }
  .bar-track {
    flex: 1;
    height: 10px;
    background: rgba(255, 255, 255, 0.08);
    overflow: hidden;
  }
  .bar-fill { height: 100%; background: #38bdf8; }
  .bar-fill.high { background: #f43f5e; }
  .hud-footer {
    border-top: 2px solid #f43f5e;
    background: rgba(244, 63, 94, 0.14);
    padding: 12px 16px;
    display: flex;
    justify-content: space-between;
    align-items: center;
  }
</style>
<script>
  window.selectPatient = function(patientId) {
    window.parent.postMessage({ type: "malcolm_iframe", action: "select_patient", patientId: String(patientId) }, "*");
  };
  window.closePane = function() {
    window.parent.postMessage({ type: "malcolm_iframe", action: "close" }, "*");
  };
  window.triggerPage = function(eventId) {
    window.parent.postMessage({ type: "malcolm_iframe", action: "trigger_page", eventId: String(eventId || "") }, "*");
  };
  window.addEventListener("message", function(e) {
    if (e.data && e.data.type === "malcolm_scroll") {
      var dy = Number(e.data.deltaY) || 0;
      window.scrollBy(0, dy);
      if (document.documentElement) document.documentElement.scrollTop += dy;
      if (document.body) document.body.scrollTop += dy;
      var root = document.querySelector(".hud-root");
      if (root && root.scrollHeight > root.clientHeight) root.scrollTop += dy;
    }
  });
</script>
</head>
<body>
${rawHtmlAndCss}
</body>
</html>`;
}

function GeneratedHtmlView({ htmlCode, activeKind }: { htmlCode: string; activeKind: CustomPaneKind }) {
  const [showSource, setShowSource] = useState(false);

  return (
    <div className="flex h-full w-full flex-col text-foreground">
      <div className={`border-b border-hairline ${PAD_X} pt-[clamp(0.75rem,2vh,1.5rem)] pb-2`}>
        <div className="flex flex-wrap items-center justify-between gap-2">
          <div className="flex flex-wrap items-center gap-1.5">
            <span className="flex items-center gap-1.5 border border-telemetry/70 bg-telemetry/15 px-2.5 py-1 font-mono text-[10px] font-semibold uppercase tracking-[0.18em] text-telemetry">
              <span className="size-1.5 animate-pulse bg-telemetry" />
              AI Generated HTML + CSS
            </span>
            <button
              type="button"
              onClick={() => void syncPaneBackend("patient")}
              className="border border-hairline bg-background/40 px-2.5 py-1 font-mono text-[11px] uppercase tracking-[0.18em] text-pane-meta transition-colors hover:border-foreground/40 hover:text-foreground"
            >
              Patient Chart
            </button>
            <button
              type="button"
              onClick={() => void syncPaneBackend("ward_table")}
              className={`border px-2.5 py-1 font-mono text-[11px] uppercase tracking-[0.18em] transition-colors ${
                activeKind === "ward_table"
                  ? "border-telemetry/80 bg-telemetry/15 text-telemetry"
                  : "border-hairline bg-background/40 text-pane-meta hover:border-foreground/40 hover:text-foreground"
              }`}
            >
              Ward Table
            </button>
            <button
              type="button"
              onClick={() => void syncPaneBackend("ward_stats")}
              className={`border px-2.5 py-1 font-mono text-[11px] uppercase tracking-[0.18em] transition-colors ${
                activeKind === "ward_stats"
                  ? "border-telemetry/80 bg-telemetry/15 text-telemetry"
                  : "border-hairline bg-background/40 text-pane-meta hover:border-foreground/40 hover:text-foreground"
              }`}
            >
              Statistics
            </button>
            <button
              type="button"
              onClick={() => void syncPaneBackend("timetable")}
              className={`border px-2.5 py-1 font-mono text-[11px] uppercase tracking-[0.18em] transition-colors ${
                activeKind === "timetable"
                  ? "border-telemetry/80 bg-telemetry/15 text-telemetry"
                  : "border-hairline bg-background/40 text-pane-meta hover:border-foreground/40 hover:text-foreground"
              }`}
            >
              Timetable
            </button>
            <button
              type="button"
              onClick={() => setShowSource((v) => !v)}
              className={`border px-2.5 py-1 font-mono text-[11px] uppercase tracking-[0.18em] transition-colors ${
                showSource
                  ? "border-foreground/70 bg-foreground/15 text-foreground"
                  : "border-hairline bg-background/40 text-pane-meta hover:border-foreground/40 hover:text-foreground"
              }`}
            >
              {showSource ? "View Rendered UI" : "Inspect HTML/CSS"}
            </button>
          </div>
          <div className="flex flex-wrap items-center gap-1.5">
            <button
              type="button"
              onClick={() => triggerExportDownload("xlsx", activeKind)}
              title="Export formatted Excel (.xlsx) on a plain white background"
              className="border border-hairline bg-background/50 px-2.5 py-1 font-mono text-[11px] uppercase tracking-[0.16em] text-pane-meta transition-colors hover:border-telemetry hover:bg-telemetry/15 hover:text-telemetry"
            >
              ⬇ .XLSX
            </button>
            <button
              type="button"
              onClick={() => triggerExportDownload("png", activeKind)}
              title="Export formatted PNG image on a plain white background"
              className="border border-hairline bg-background/50 px-2.5 py-1 font-mono text-[11px] uppercase tracking-[0.16em] text-pane-meta transition-colors hover:border-telemetry hover:bg-telemetry/15 hover:text-telemetry"
            >
              ⬇ .PNG
            </button>
            <button
              type="button"
              onClick={() => void syncPaneBackend("close")}
              title="Close split-screen pane"
              className="border border-hairline bg-background/50 px-2.5 py-1 font-mono text-[11px] uppercase tracking-[0.18em] text-pane-meta transition-colors hover:border-rose hover:bg-rose/15 hover:text-rose-soft"
            >
              × Close Pane
            </button>
          </div>
        </div>
      </div>

      <div className="relative min-h-0 flex-1 w-full overflow-hidden">
        {showSource ? (
          <pre
            id="malcolm-source-pre"
            className="h-full w-full overflow-auto bg-background/90 p-6 font-mono text-xs leading-relaxed text-telemetry"
          >
            {htmlCode}
          </pre>
        ) : (
          <iframe
            id="malcolm-generated-iframe"
            key={activeKind + "-" + htmlCode.length}
            title="Malcolm AI Generated Clinical HUD"
            srcDoc={buildGeneratedIframeDoc(htmlCode)}
            sandbox="allow-scripts"
            className="h-full w-full border-0 bg-transparent"
          />
        )}
      </div>
    </div>
  );
}

export function scrollRightPaneBy(deltaY: number) {
  if (typeof window === "undefined" || !Number.isFinite(deltaY) || deltaY === 0) return;
  const iframe = document.getElementById("malcolm-generated-iframe") as HTMLIFrameElement | null;
  if (iframe?.contentWindow) {
    iframe.contentWindow.postMessage({ type: "malcolm_scroll", deltaY }, "*");
  }
  const sourcePre = document.getElementById("malcolm-source-pre");
  if (sourcePre) {
    sourcePre.scrollTop += deltaY;
  }
  const hostScroll = document.getElementById("malcolm-right-pane-scroll");
  if (hostScroll) {
    hostScroll.scrollTop += deltaY;
  }
}

/** Edge-to-edge clinical display supporting AI-Generated HTML+CSS, Single Patient Chart, Ward Table, and Ward Statistics. */
export function PatientPane({ patient, bpm }: { patient: PatientSummary; bpm: number }) {
  const { customPane, wardHud } = useMalcolm();
  const paneKind: CustomPaneKind = customPane?.kind || "patient";
  const generatedHtml = (customPane?.generatedHtml || "").trim();

  useEffect(() => {
    const onMessage = (e: MessageEvent) => {
      const d = e.data as {
        type?: string;
        action?: string;
        patientId?: string;
        eventId?: string;
      } | null;
      if (!d || d.type !== "malcolm_iframe") return;
      if (d.action === "close") {
        void syncPaneBackend("close");
      } else if (d.action === "trigger_page") {
        void triggerLoudspeakerPage(d.eventId);
      } else if (d.action === "select_patient" && d.patientId) {
        const hud = wardHud?.patients?.length ? wardHud : DEFAULT_WARD_HUD;
        const found = hud.patients.find((p) => p.patientId === d.patientId);
        if (found) {
          void selectPatientFromRow(found);
        }
      }
    };
    window.addEventListener("message", onMessage);
    return () => window.removeEventListener("message", onMessage);
  }, [wardHud]);

  if (generatedHtml && (paneKind === "timetable" || paneKind === "custom_html")) {
    return <GeneratedHtmlView htmlCode={generatedHtml} activeKind={paneKind} />;
  }

  if (paneKind === "ward_table") {
    return (
      <div id="malcolm-right-pane-scroll" className="h-full w-full overflow-y-auto">
        <div className="min-h-[130vh]">
          <WardTableView />
        </div>
      </div>
    );
  }
  if (paneKind === "ward_stats") {
    return (
      <div id="malcolm-right-pane-scroll" className="h-full w-full overflow-y-auto">
        <div className="min-h-[130vh]">
          <WardStatsView />
        </div>
      </div>
    );
  }

  const roomDisplay = /^room\b/i.test(patient.room) ? patient.room : `Room ${patient.room}`;
  const meta = [`Patient ${patient.patientId}`, patient.wing, roomDisplay].filter(
    (s) => s.trim(),
  );

  return (
    <div id="malcolm-right-pane-scroll" className="h-full w-full overflow-y-auto">
      <div className="grid min-h-[130vh] grid-rows-[minmax(min-content,24fr)_minmax(min-content,25fr)_minmax(min-content,21fr)_minmax(min-content,30fr)] text-foreground">
      {/* Zone 1 — patient header */}
      <Zone className={`${PAD_X} pt-[clamp(0.75rem,2.5vh,2rem)]`}>
        <PaneTopNav activeKind="patient" />
        <p className="flex flex-wrap gap-x-4 font-mono text-sm uppercase tracking-[0.2em] text-pane-meta">
          {meta.map((m, i) => (
            <span key={m} className="flex gap-4">
              {i > 0 ? <span aria-hidden>/</span> : null}
              {m}
            </span>
          ))}
        </p>
        <h2 className="mt-1.5 truncate font-sans text-[clamp(2rem,min(6.5vh,4.2vw),3.5rem)] font-medium leading-none tracking-tight">
          {patient.name}
        </h2>
      </Zone>

      {/* Zone 2 — department + live telemetry */}
      <Zone className={PAD_X}>
        <div className="grid grid-cols-2">
          <div className="border-r border-hairline pr-8">
            <p className={LABEL}>Department &amp; protocol</p>
            <p className={`mt-2 font-sans font-medium tracking-tight ${VALUE_XL}`}>
              {patient.department}
            </p>
            <p className={`mt-1 ${SUB}`}>{patient.protocol}</p>
          </div>
          <div className="min-w-0 pl-8">
            <p className={LABEL}>Live telemetry state</p>
            <p className="mt-2 font-mono text-[clamp(1.1rem,min(4vh,1.95vw),2.25rem)] font-medium leading-tight tracking-tight text-telemetry">
              {bpm} BPM — {patient.rhythm}
            </p>
            <p className={`mt-1 ${SUB}`}>{patient.telemetryDetail}</p>
          </div>
        </div>
      </Zone>

      {/* Zone 3 — nurse rounds */}
      <Zone className={PAD_X}>
        <p className={LABEL}>Previous nurse rounds</p>
        <p className="mt-2 text-[clamp(1.05rem,min(3.6vh,2.2vw),1.875rem)] font-light leading-snug text-text-bright">
          {patient.nurseRounds}
        </p>
      </Zone>

      {/* Zone 4 — active triage alert, full bleed */}
      <div
        className={`flex min-h-0 flex-col justify-center gap-3 border-t-2 border-rose bg-alert-tint/30 ${PAD_X} py-[clamp(0.75rem,3vh,2.25rem)]`}
      >
        <div className="flex flex-wrap items-center justify-between gap-x-6 gap-y-2">
          <p className="flex flex-wrap gap-x-[0.6em] font-mono text-sm uppercase tracking-[0.2em] text-rose-soft">
            <span className="whitespace-nowrap">Patient-reported symptom</span>
            <span className="whitespace-nowrap">(live check-in)</span>
          </p>
          {patient.dispatchStatus ? (
            <p className="flex shrink-0 items-center gap-2.5 whitespace-nowrap bg-alert px-4 py-1.5 font-mono text-sm font-semibold uppercase tracking-[0.15em] text-foreground">
              <span className="animate-beat-dot size-2 bg-foreground" style={{ animationDuration: "1.2s" }} />
              {patient.dispatchStatus}
            </p>
          ) : null}
        </div>
        <p className="text-[clamp(1.3rem,min(4.6vh,2.6vw),2.25rem)] font-normal leading-tight tracking-tight">
          &ldquo;{patient.symptom}&rdquo;
        </p>
      </div>
      </div>
    </div>
  );
}
