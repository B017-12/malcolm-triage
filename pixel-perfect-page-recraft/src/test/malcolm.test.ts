import { describe, expect, it } from "vitest";
import {
  beatEnvelope,
  beatPeriod,
  clampBpm,
  ecgSample,
  normalizeMode,
  parseRemoteState,
} from "@/lib/malcolm/signal";

describe("Malcolm signal rules", () => {
  it("90 BPM pulses 1.5 times per second", () => {
    expect(1 / beatPeriod(90)).toBeCloseTo(1.5);
  });

  it("maps backend state names to the three screens", () => {
    expect(normalizeMode("idle")).toBe("idle");
    expect(normalizeMode("heart_rate")).toBe("heart");
    expect(normalizeMode("patient_stats")).toBe("patient");
    expect(normalizeMode("something-else")).toBe("idle");
  });

  it("clamps BPM to 30–220 and rejects non-numbers", () => {
    expect(clampBpm(300)).toBe(220);
    expect(clampBpm("89.6")).toBe(90);
    expect(clampBpm("abc")).toBeNull();
  });

  it("ring pulse and ECG R-peak land on the same beat phase", () => {
    expect(beatEnvelope(0)).toBe(1);
    expect(ecgSample(0)).toBeGreaterThan(ecgSample(0.5) + 0.8);
  });

  it("parses the { state, data } payload from the Python backend", () => {
    expect(
      parseRemoteState({
        state: "patient_stats",
        data: { bpm: 91, transcript: "Pulling up the chart", patient: { name: "Aino Virtanen" } },
      }),
    ).toEqual({
      mode: "patient",
      bpm: 91,
      caption: "Pulling up the chart",
      patient: { name: "Aino Virtanen" },
    });
  });

  it("setMalcolmState-style call merges a separate data argument", () => {
    expect(parseRemoteState({ state: "heart_rate" }, { bpm: 88 })).toEqual({
      mode: "heart",
      bpm: 88,
    });
  });
});

