import type { ReactNode } from "react";
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

function Zone({ className = "", children }: { className?: string; children: ReactNode }) {
  return (
    <div className={`flex min-h-0 flex-col justify-center border-b border-hairline ${className}`}>
      {children}
    </div>
  );
}

async function syncPaneBackend(action: CustomPaneKind | "close", extra?: Record<string, string>) {
  if (action === "close") {
    malcolmStore.setMode("idle");
    malcolmStore.setCustomPane({ kind: "patient", title: "", subtitle: "", filterWing: "" });
  } else {
    malcolmStore.setMode("patient");
    malcolmStore.setCustomPane({
      kind: action,
      ...(extra?.filterWing !== undefined ? { filterWing: extra.filterWing } : {}),
    });
  }
  try {
    const res = await fetch("http://localhost:5000/api/pane", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ action, ...extra }),
    });
    if (res.ok) {
      const payload = await res.json();
      const parsed = parseRemoteState(payload);
      if (parsed.patient) malcolmStore.setPatient(parsed.patient);
      if (parsed.customPane) malcolmStore.setCustomPane(parsed.customPane);
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

function PaneTopNav({ activeKind }: { activeKind: CustomPaneKind }) {
  const tabs: { kind: CustomPaneKind; label: string }[] = [
    { kind: "patient", label: "Patient Chart" },
    { kind: "ward_table", label: "Ward Table" },
    { kind: "ward_stats", label: "Statistics" },
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
      <button
        type="button"
        onClick={() => void syncPaneBackend("close")}
        title="Close split-screen pane"
        className="border border-hairline bg-background/50 px-2.5 py-1 font-mono text-[11px] uppercase tracking-[0.18em] text-pane-meta transition-colors hover:border-rose hover:bg-rose/15 hover:text-rose-soft"
      >
        × Close Pane
      </button>
    </div>
  );
}

function WardTableView() {
  const { customPane, wardHud, patient } = useMalcolm();
  const hud = wardHud?.patients?.length ? wardHud : DEFAULT_WARD_HUD;
  const filterWing = (customPane.filterWing || "").trim();
  const rows = filterWing
    ? hud.patients.filter((p) => p.wing.toLowerCase() === filterWing.toLowerCase())
    : hud.patients;

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
          <div className="flex items-center gap-1.5">
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
          </div>
        </div>
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
                <th className="py-2.5 pl-3 pr-4">Symptoms &amp; Round Status</th>
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
                    <td className="py-3 pl-3 pr-4 align-top">
                      <p className="line-clamp-2 text-xs leading-snug text-text-bright">
                        {row.symptom}
                      </p>
                      <p className="mt-1 line-clamp-1 font-mono text-[11px] text-pane-meta">
                        {row.nurseNote}
                      </p>
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

/** Edge-to-edge clinical display supporting Single Patient Chart, Ward Table, and Ward Statistics. */
export function PatientPane({ patient, bpm }: { patient: PatientSummary; bpm: number }) {
  const { customPane } = useMalcolm();
  const paneKind: CustomPaneKind = customPane?.kind || "patient";

  if (paneKind === "ward_table") {
    return <WardTableView />;
  }
  if (paneKind === "ward_stats") {
    return <WardStatsView />;
  }

  const roomDisplay = /^room\b/i.test(patient.room) ? patient.room : `Room ${patient.room}`;
  const meta = [`Patient ${patient.patientId}`, patient.wing, roomDisplay].filter(
    (s) => s.trim(),
  );

  return (
    <div className="grid h-full grid-rows-[minmax(min-content,24fr)_minmax(min-content,25fr)_minmax(min-content,21fr)_minmax(min-content,30fr)] text-foreground">
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
  );
}
