import { createFileRoute } from "@tanstack/react-router";
import { useEffect } from "react";
import { cn } from "@/lib/utils";
import { WaveBackground } from "@/components/malcolm/WaveBackground";
import { RadialRing } from "@/components/malcolm/RadialRing";
import { PatientPane } from "@/components/malcolm/PatientPane";
import { Caption, MicroDots, VoiceDock, useModeHotkeys } from "@/components/malcolm/Controls";
import { installMalcolmBridge } from "@/lib/malcolm/bridge";
import { useMalcolm } from "@/lib/malcolm/store";

export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      { title: "Malcolm — Clinical triage assistant" },
      {
        name: "description",
        content:
          "Malcolm, a voice triage assistant for clinicians: live heart rate monitoring and patient summaries at a glance.",
      },
      { property: "og:title", content: "Malcolm — Clinical triage assistant" },
      {
        property: "og:description",
        content: "Live heart rate monitoring and patient summaries for clinical teams.",
      },
      { property: "og:type", content: "website" },
      { name: "twitter:card", content: "summary_large_image" },
    ],
  }),
  component: Index,
});

const SPLIT = "duration-700 ease-[cubic-bezier(0.22,1,0.36,1)]";

function Index() {
  useEffect(() => installMalcolmBridge(), []);
  useModeHotkeys();
  const { mode, patient, bpm } = useMalcolm();
  const stats = mode === "patient";

  return (
    <main className="flex h-dvh w-screen overflow-hidden bg-background">
      <h1 className="sr-only">Malcolm clinical triage assistant</h1>

      {/* Left: wave canvas + circle. Full width, or compressed to 38vw for patient stats. */}
      <section
        className={cn(
          "relative h-full shrink-0 overflow-hidden border-r transition-[width,border-color]",
          SPLIT,
          stats ? "w-[38vw] border-frame" : "w-screen border-transparent",
        )}
      >
        <WaveBackground />
        <div className="relative z-10 flex h-full w-full flex-col items-center justify-between px-6 pt-6 pb-20">
          <div className="flex min-h-0 w-full flex-1 items-center justify-center">
            <div
              className={cn(
                "transition-[width,transform]",
                SPLIT,
                stats ? "w-[min(240px,28vw,38vh)]" : "w-[min(300px,46vmin)]",
              )}
            >
              <RadialRing />
            </div>
          </div>
          <Caption />
        </div>
      </section>

      {/* Right: edge-to-edge clinical pane. */}
      <section
        aria-hidden={!stats}
        className={cn(
          "h-full shrink-0 overflow-hidden bg-pane transition-[width,opacity]",
          SPLIT,
          stats ? "w-[62vw] opacity-100" : "w-0 opacity-0",
        )}
      >
        {/* Fixed inner width so text doesn't reflow while the pane slides open */}
        <div className="h-full w-[62vw]">
          <PatientPane patient={patient} bpm={bpm} />
        </div>
      </section>

      <VoiceDock />
      <MicroDots />
    </main>
  );
}
