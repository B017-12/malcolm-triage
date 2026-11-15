import { useEffect, useRef } from "react";
import { useMalcolm } from "@/lib/malcolm/store";
import { beatPeriod } from "@/lib/malcolm/signal";

const W = 1440;
const H = 900;
const POINTS = 56;

// Back to front, spread across the full viewport height (top 0% to bottom 100%).
const LAYERS = [
  { base: 0.06, a1: 42, a2: 16, k1: 1.0, k2: 2.6, s1: 0.14, s2: 0.1, fill: 0.55, ridge: 0.3 },
  { base: 0.18, a1: 48, a2: 18, k1: 0.85, k2: 2.2, s1: -0.12, s2: 0.16, fill: 0.16, ridge: 0.4 },
  { base: 0.3, a1: 38, a2: 20, k1: 1.3, k2: 3.0, s1: 0.18, s2: -0.11, fill: 0.18, ridge: 0.45 },
  { base: 0.42, a1: 44, a2: 14, k1: 0.95, k2: 2.4, s1: -0.15, s2: 0.13, fill: 0.5, ridge: 0.35 },
  { base: 0.54, a1: 36, a2: 16, k1: 1.2, k2: 2.8, s1: 0.11, s2: -0.17, fill: 0.16, ridge: 0.5 },
  { base: 0.66, a1: 40, a2: 14, k1: 0.8, k2: 2.1, s1: -0.1, s2: 0.12, fill: 0.2, ridge: 0.45 },
  { base: 0.78, a1: 32, a2: 12, k1: 1.1, k2: 2.5, s1: 0.13, s2: -0.14, fill: 0.18, ridge: 0.5 },
  { base: 0.9, a1: 26, a2: 10, k1: 0.9, k2: 2.3, s1: -0.12, s2: 0.1, fill: 0.22, ridge: 0.55 },
];

function ridgePath(i: number, t: number): string {
  const L = LAYERS[i]!;
  const baseY = H * L.base;
  let d = "";
  for (let n = 0; n <= POINTS; n++) {
    const x = (n / POINTS) * W;
    const u = (x / W) * Math.PI * 2;
    const y =
      baseY +
      L.a1 * Math.sin(u * L.k1 + t * L.s1 * 2.4 + i * 1.7) +
      L.a2 * Math.sin(u * L.k2 - t * L.s2 * 2.4 + i * 0.6);
    d += n === 0 ? `M0,${y.toFixed(1)}` : ` L${x.toFixed(1)},${y.toFixed(1)}`;
  }
  return d;
}

/** Eight translucent full-height wave ribbons with 1px ridge highlights. Colors from CSS tokens. */
export function WaveBackground() {
  const { mode, bpm } = useMalcolm();
  const fills = useRef<(SVGPathElement | null)[]>([]);
  const ridges = useRef<(SVGPathElement | null)[]>([]);

  useEffect(() => {
    const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    let raf = 0;
    let t = 0;
    let last = performance.now();
    const frame = (now: number) => {
      t += (Math.min(50, now - last) / 1000) * (reduced ? 0.25 : 1);
      last = now;
      for (let i = 0; i < LAYERS.length; i++) {
        const r = ridgePath(i, t);
        ridges.current[i]?.setAttribute("d", r);
        fills.current[i]?.setAttribute("d", `${r} L${W},${H} L0,${H} Z`);
      }
      raf = requestAnimationFrame(frame);
    };
    raf = requestAnimationFrame(frame);
    return () => cancelAnimationFrame(raf);
  }, []);

  return (
    <div
      aria-hidden
      data-mode={mode === "idle" ? "idle" : "vitals"}
      className="mesh-base pointer-events-none absolute inset-0"
      style={{ ["--beat" as string]: `${beatPeriod(bpm)}s` }}
    >
      <svg
        className="waves-pulse absolute inset-0 h-full w-full"
        viewBox={`0 0 ${W} ${H}`}
        preserveAspectRatio="none"
      >
        {LAYERS.map((l, i) => {
          const r = ridgePath(i, 0);
          return (
            <g key={i}>
              <path
                ref={(el) => {
                  fills.current[i] = el;
                }}
                d={`${r} L${W},${H} L0,${H} Z`}
                className={`wave-layer wave-${i + 1}`}
                opacity={l.fill}
              />
              <path
                ref={(el) => {
                  ridges.current[i] = el;
                }}
                d={r}
                className="wave-ridge"
                strokeWidth={1}
                vectorEffect="non-scaling-stroke"
                opacity={l.ridge}
              />
            </g>
          );
        })}
      </svg>
    </div>
  );
}
