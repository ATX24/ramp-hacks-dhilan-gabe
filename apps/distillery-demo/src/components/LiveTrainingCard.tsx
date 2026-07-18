"use client";

import { StatusBadge } from "@/components/StatusBadge";
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { adaptLiveTrainingGlance } from "@/lib/trainingEvents";
import type {
  DistillationPlan,
  DistillationRunView,
  ModelArtifactMeta,
  TrainingTelemetry,
} from "@/lib/types";

function formatTimestamp(timestamp: string | null): string {
  if (!timestamp) return "unknown time";
  return new Intl.DateTimeFormat("en-US", {
    dateStyle: "medium",
    timeStyle: "medium",
    timeZone: "UTC",
  }).format(new Date(timestamp));
}

export function LiveTrainingCard({
  run,
  plan,
  telemetry,
  artifact,
}: {
  run: DistillationRunView;
  plan: DistillationPlan;
  telemetry: TrainingTelemetry;
  artifact: ModelArtifactMeta | null;
}) {
  const glance = adaptLiveTrainingGlance({ run, plan, telemetry, artifact });

  return (
    <Card
      className="border-border/80 bg-card/90 shadow-none"
      data-testid="live-training-card"
      data-origin={glance.origin}
      data-live={glance.isLive ? "true" : "false"}
    >
      <CardHeader className="gap-3 border-b border-border/70 pb-4">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <CardTitle className="font-serif text-xl font-normal tracking-tight">
            Teaching status
          </CardTitle>
          <div className="flex flex-wrap gap-2">
            <StatusBadge tone={glance.isLive ? "pass" : "precomputed"}>
              {glance.originLabel}
            </StatusBadge>
            <StatusBadge
              tone={
                glance.status === "failed"
                  ? "fail"
                  : glance.status === "teaching" || glance.status === "checking"
                    ? "pass"
                    : glance.status === "finished"
                      ? "precomputed"
                      : "pending"
              }
            >
              {glance.statusLabel}
            </StatusBadge>
          </div>
        </div>
        <p className="text-sm text-muted-foreground">
          At-a-glance progress for the sealed teaching job. Saved previews and earlier
          runs are labeled so they are never mistaken for a live cloud job.
        </p>
      </CardHeader>
      <CardContent className="grid gap-4 pt-4">
        <div className="grid gap-3 sm:grid-cols-2">
          <GlanceStat label="Progress" value={glance.progressLabel} />
          <GlanceStat label="ETA" value={glance.etaLabel} />
          <GlanceStat label="Spend" value={glance.spendLabel} />
          <GlanceStat label="Current experiment" value={glance.experimentLabel} />
        </div>

        {glance.progressPercent !== null ? (
          <div className="grid gap-2" data-testid="live-training-progress">
            <Progress value={glance.progressPercent} aria-label="Teaching progress" />
          </div>
        ) : null}

        <div
          className="rounded-xl border border-border bg-soft/40 px-3 py-3"
          data-testid="live-training-recent-event"
        >
          <p className="text-kicker text-muted-foreground">Most recent event</p>
          <p className="mt-1 font-serif text-base leading-snug">
            {glance.recentEvent.summary}
          </p>
          <p className="mt-1 text-xs text-muted-foreground">
            {formatTimestamp(glance.recentEvent.timestamp)}
            {glance.recentEvent.state && glance.isLive
              ? ` · ${glance.recentEvent.state}`
              : ""}
            {` · ${glance.recentEvent.origin === "live" ? "live" : glance.recentEvent.origin === "precomputed_prior_run" ? "saved earlier run" : "saved preview"}`}
          </p>
        </div>

        <Accordion type="single" collapsible>
          <AccordionItem value="events" className="border-border">
            <AccordionTrigger className="text-sm">
              Advanced · full event log
            </AccordionTrigger>
            <AccordionContent>
              <ul className="list-plain space-y-2 text-sm">
                {glance.events.map((event, index) => (
                  <li key={`${event.timestamp ?? "na"}-${event.state ?? index}`}>
                    <strong>{event.summary}</strong>
                    {event.technical ? (
                      <span className="block text-muted-foreground">
                        Technical: {event.technical}
                      </span>
                    ) : null}
                  </li>
                ))}
              </ul>
            </AccordionContent>
          </AccordionItem>
        </Accordion>
      </CardContent>
    </Card>
  );
}

function GlanceStat({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-xl border border-border bg-background/60 px-3 py-3">
      <p className="text-kicker text-muted-foreground">{label}</p>
      <p className="mt-1 text-sm leading-snug text-foreground">{value}</p>
    </div>
  );
}
