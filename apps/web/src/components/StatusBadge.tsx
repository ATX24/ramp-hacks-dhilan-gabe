import type { ReactNode } from "react";
import { Badge } from "@/components/ui/badge";
import type { DemoModelArmId } from "@/lib/demo/types";

type Tone = "pass" | "fail" | "warn" | "pending" | "unavailable" | "projected" | "precomputed";

const CLASS_BY_TONE: Record<Tone, string> = {
  pass:
    "border-[color-mix(in_oklab,var(--pass)_28%,transparent)] bg-[color-mix(in_oklab,var(--pass)_10%,transparent)] text-[var(--pass)]",
  fail:
    "border-[color-mix(in_oklab,var(--fail)_28%,transparent)] bg-[color-mix(in_oklab,var(--fail)_10%,transparent)] text-[var(--fail)]",
  warn:
    "border-[color-mix(in_oklab,var(--warn)_28%,transparent)] bg-[color-mix(in_oklab,var(--warn)_10%,transparent)] text-[var(--warn)]",
  pending:
    "border-border bg-secondary/60 text-muted-foreground",
  unavailable:
    "border-[color-mix(in_oklab,var(--unavailable)_28%,transparent)] bg-[color-mix(in_oklab,var(--unavailable)_10%,transparent)] text-[var(--unavailable)]",
  projected:
    "border-[color-mix(in_oklab,var(--orange)_28%,transparent)] bg-[color-mix(in_oklab,var(--orange)_10%,transparent)] text-foreground",
  precomputed:
    "border-border bg-secondary/60 text-foreground",
};

export function StatusBadge({
  tone,
  children,
}: {
  tone: Tone;
  children: ReactNode;
}) {
  return (
    <Badge
      variant="outline"
      className={`rounded-full font-normal normal-case ${CLASS_BY_TONE[tone]}`}
    >
      {children}
    </Badge>
  );
}

export function armTone(armId: DemoModelArmId): Tone {
  switch (armId) {
    case "student_base":
      return "pending";
    case "oracle_sft":
      return "warn";
    case "sequence_kd":
    case "logit_kd":
      return "projected";
    case "ce_ablation":
      return "unavailable";
    case "promoted_winner":
      return "pass";
  }
}

export function armBadgeLabel(armId: DemoModelArmId): string {
  switch (armId) {
    case "student_base":
      return "Base";
    case "oracle_sft":
      return "SFT";
    case "sequence_kd":
      return "Sequence KD";
    case "logit_kd":
      return "Logit KD";
    case "ce_ablation":
      return "Ablation";
    case "promoted_winner":
      return "Promoted";
  }
}

export function gateTone(
  status: "pass" | "fail" | "pending" | "unavailable",
): Tone {
  switch (status) {
    case "pass":
      return "pass";
    case "fail":
      return "fail";
    case "pending":
      return "pending";
    case "unavailable":
      return "unavailable";
    default: {
      const _exhaustive: never = status;
      return _exhaustive;
    }
  }
}

export function proofTone(
  status:
    | "proved"
    | "do_not_distill"
    | "failed_quality"
    | "failed_economics"
    | "insufficient_evidence",
): Tone {
  switch (status) {
    case "proved":
      return "pass";
    case "do_not_distill":
      return "warn";
    case "failed_quality":
    case "failed_economics":
      return "fail";
    case "insufficient_evidence":
      return "unavailable";
    default: {
      const _exhaustive: never = status;
      return _exhaustive;
    }
  }
}
