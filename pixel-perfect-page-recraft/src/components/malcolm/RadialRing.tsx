import { useEffect, useRef, useState } from "react";
import { cn } from "@/lib/utils";
import { malcolmStore, useMalcolm } from "@/lib/malcolm/store";
import { beatEnvelope, beatPeriod } from "@/lib/malcolm/signal";

const SIZE = 400;
const C = SIZE / 2;
const BARS = 40;
const R = 132; // inner radius where blades start

// Fixed PQRST trace drawn inside the white disc (viewBox 0 0 240 70).
const ECG_D =
  "M 0 35 L 55 35 L 68 22 L 78 45 L 90 6 L 102 62 L 114 18 L 126 35 L 150 26 L 165 35 L 240 35";

/** 40-blade radial sunburst; calm harmonic wave when idle/non-patient pane, BPM pulse + ECG only for specific patient vitals. */
export function RadialRing() {
  const { mode, bpm, customPane } = useMalcolm();
  const paneKind = customPane?.kind || "patient";
  const vitals = mode === "heart" || (mode === "patient" && paneKind === "patient");
  const showIdleLabel = mode === "idle";
  const bars = useRef<(SVGLineElement | null)[]>([]);

  useEffect(() => {
    const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    let raf = 0;
    let t = 0;
    let phase = 0;
    let last = performance.now();
    const initState = malcolmStore.get();
    const initVitals =
      initState.mode === "heart" ||
      (initState.mode === "patient" && (initState.customPane?.kind || "patient") === "patient");
    let m = initVitals ? 1 : 0;

    const frame = (now: number) => {
      const dt = Math.min(0.05, (now - last) / 1000);
      last = now;
      t += dt * (reduced ? 0.35 : 1);
      const s = malcolmStore.get();
      const sVitals =
        s.mode === "heart" ||
        (s.mode === "patient" && (s.customPane?.kind || "patient") === "patient");
      phase += dt * (s.bpm / 60);
      m += ((sVitals ? 1 : 0) - m) * (1 - Math.exp(-dt * 3));
      const env = beatEnvelope(phase);

      const breath = 1 + 0.015 * Math.sin(t * 0.9);
      const scale = breath * (1 - m) + (1 + 0.035 * env) * m;
      const r0 = R * scale;

      for (let i = 0; i < BARS; i++) {
        const el = bars.current[i];
        if (!el) continue;
        const th = (i / BARS) * Math.PI * 2 - Math.PI / 2;
        const wave = 0.5 + 0.32 * Math.sin(3 * th - t * 1.2) + 0.18 * Math.sin(5 * th + t * 0.8);
        const idleLen = 12 + 26 * wave;
        const jit = 0.75 + 0.25 * Math.sin(i * 2.3 + 1.1);
        const heartLen = 10 + 34 * env * jit;
        const len = idleLen * (1 - m) + heartLen * m;
        const c = Math.cos(th);
        const sn = Math.sin(th);
        el.setAttribute("x1", (C + c * r0).toFixed(2));
        el.setAttribute("y1", (C + sn * r0).toFixed(2));
        el.setAttribute("x2", (C + c * (r0 + len)).toFixed(2));
        el.setAttribute("y2", (C + sn * (r0 + len)).toFixed(2));
      }

      raf = requestAnimationFrame(frame);
    };
    raf = requestAnimationFrame(frame);
    return () => cancelAnimationFrame(raf);
  }, []);

  return (
    <div className="relative aspect-square w-full" data-vitals={vitals}>
      <div className="core-disc absolute left-1/2 top-1/2 size-[62%] -translate-x-1/2 -translate-y-1/2 rounded-full" />

      <svg viewBox={`0 0 ${SIZE} ${SIZE}`} className="absolute inset-0 h-full w-full" aria-hidden>
        <defs>
          <radialGradient id="blade-grad" gradientUnits="userSpaceOnUse" cx={C} cy={C} r={R + 56}>
            <stop offset={R / (R + 56)} className="blade-stop-inner" />
            <stop offset="1" className="blade-stop-outer" />
          </radialGradient>
        </defs>
        <g stroke="url(#blade-grad)" strokeWidth={4} strokeLinecap="butt" fill="none">
          {Array.from({ length: BARS }, (_, i) => (
            <line
              key={i}
              ref={(el) => {
                bars.current[i] = el;
              }}
            />
          ))}
        </g>
      </svg>

      {/* Idle label */}
      <p
        className={cn(
          "pointer-events-none absolute inset-x-0 top-1/2 z-20 -translate-y-1/2 text-center font-sans text-[clamp(1.1rem,3.8vmin,1.5rem)] font-medium tracking-tight text-core-ink transition-all duration-700 ease-out",
          showIdleLabel ? "scale-100 opacity-100" : "scale-95 opacity-0",
        )}
      >
        At your Service
      </p>

      {/* Vitals readout: BPM + ECG, stacked inside the white disc */}
      <div
        className={cn(
          "pointer-events-none absolute inset-0 z-20 flex flex-col items-center justify-center pt-[4%] transition-all duration-700 ease-out",
          vitals ? "translate-y-0 opacity-100" : "translate-y-2 opacity-0",
        )}
      >
        <BpmReadout visible={vitals} bpm={bpm} />
        <svg
          viewBox="0 0 240 70"
          className="mt-1.5 aspect-[240/70] w-[42%] overflow-visible"
          aria-hidden
        >
          <path d={ECG_D} className="ecg-base" strokeWidth={2.5} fill="none" strokeLinejoin="miter" />
          <path
            d={ECG_D}
            pathLength={280}
            className="ecg-sweep"
            strokeWidth={4}
            fill="none"
            strokeLinejoin="miter"
            strokeDasharray={280}
            strokeDashoffset={280}
            style={{ animationDuration: `${beatPeriod(bpm)}s` }}
          />
        </svg>
      </div>
    </div>
  );
}

function BpmReadout({ visible, bpm }: { visible: boolean; bpm: number }) {
  const [shown, setShown] = useState(bpm);
  useEffect(() => {
    setShown(bpm);
    if (!visible) return;
    const id = setInterval(() => setShown(bpm + Math.round(Math.random() * 2 - 1)), 1600);
    return () => clearInterval(id);
  }, [bpm, visible]);

  return (
    <div aria-live="polite" className="flex flex-col items-center gap-0.5 leading-none">
      <span className="font-mono text-[clamp(2.1rem,7.2vmin,3.5rem)] font-extrabold leading-none tracking-tighter tabular-nums text-core-ink-strong">
        {shown}
      </span>
      <span className="flex items-center gap-1.5">
        <span
          className="animate-beat-dot size-1.5 bg-vitals"
          style={{ animationDuration: `${beatPeriod(bpm)}s` }}
        />
        <span className="font-mono text-[10px] font-bold uppercase tracking-[0.18em] text-core-ink-strong">
          BPM
        </span>
      </span>
    </div>
  );
}
