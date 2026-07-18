import type { ReactNode } from "react";

type Tone = "pass" | "fail" | "warn" | "pending" | "unavailable" | "projected" | "precomputed";

const CLASS_BY_TONE: Record<Tone, string> = {
  pass: "badge badge-pass",
  fail: "badge badge-fail",
  warn: "badge badge-warn",
  pending: "badge badge-pending",
  unavailable: "badge badge-unavailable",
  projected: "badge badge-projected",
  precomputed: "badge badge-precomputed",
};

export function StatusBadge({
  tone,
  children,
}: {
  tone: Tone;
  children: ReactNode;
}) {
  return <span className={CLASS_BY_TONE[tone]}>{children}</span>;
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
