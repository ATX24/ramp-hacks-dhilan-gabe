"use client";

import { CodeBlock, CodeBlockCopyButton } from "@/components/ai-elements/code-block";
import {
  Context,
  ContextContent,
  ContextContentBody,
  ContextContentHeader,
  ContextTrigger,
} from "@/components/ai-elements/context";
import { Message, MessageContent } from "@/components/ai-elements/message";
import {
  StatusBadge,
  armBadgeLabel,
  armTone,
} from "@/components/StatusBadge";
import { Button } from "@/components/ui/button";
import type {
  DemoInferenceResponse,
  DemoModelEntry,
} from "@/lib/demo/types";

export function DemoResultCard({
  result,
  model,
}: {
  result: DemoInferenceResponse;
  model: DemoModelEntry | null;
}) {
  if (result.status === "unavailable" || result.status === "error") {
    return (
      <article
        className="rounded-[14px] border border-border bg-card p-4"
        data-testid={`demo-result-${result.model_id}`}
        data-status={result.status}
        role="alert"
      >
        <header className="mb-2 flex flex-wrap items-center gap-2">
          <strong className="font-serif text-lg font-normal">
            {model?.display_name ?? result.model_id}
          </strong>
          {model ? (
            <StatusBadge tone={armTone(model.arm_id)}>
              {armBadgeLabel(model.arm_id)}
            </StatusBadge>
          ) : null}
          <StatusBadge tone="unavailable">Not available</StatusBadge>
        </header>
        <p className="text-sm">{result.message}</p>
        <p className="mt-2 text-xs text-muted-foreground">
          Error code: <code>{result.code}</code>
        </p>
      </article>
    );
  }

  const confidence =
    typeof result.structured_output.confidence === "number"
      ? `${Math.round(result.structured_output.confidence * 100)}% in this output`
      : "Not provided";
  const usedTokens = (result.prompt_tokens ?? 0) + (result.completion_tokens ?? 0);

  return (
    <article
      className="rounded-[14px] border border-border bg-card p-4"
      data-testid={`demo-result-${result.model_id}`}
      data-status="ok"
      data-provenance={result.provenance}
    >
      <header className="mb-3 flex flex-wrap items-center gap-2">
        <strong className="font-serif text-lg font-normal">
          {model?.display_name ?? result.model_id}
        </strong>
        {model ? (
          <StatusBadge tone={armTone(model.arm_id)}>
            {armBadgeLabel(model.arm_id)}
          </StatusBadge>
        ) : null}
        <StatusBadge
          tone={result.provenance === "fixture_preview" ? "precomputed" : "pass"}
        >
          {result.provenance === "fixture_preview"
            ? "Saved demo. Not live."
            : "Live output"}
        </StatusBadge>
      </header>

      <div className="grid gap-2 rounded-[12px] bg-secondary/35 p-3 text-sm">
        <Fact term="Decision" value={plainOutcome(result.structured_output)} />
        <Fact term="Why" value={plainReason(result.structured_output)} />
        <Fact term="Confidence" value={confidence} />
        <Fact
          term="Quality"
          value={
            result.score === null
              ? "Not scored"
              : result.score === 1
                ? "Matched the known answer"
                : "Did not match the known answer"
          }
        />
        <Fact
          term="Speed"
          value={
            result.latency_ms === null
              ? "Not measured"
              : `${result.latency_ms} ms in this saved demo`
          }
        />
        <Fact term="Cost" value="Not measured in a live run" />
      </div>

      <div className="mt-3 flex flex-wrap gap-3 text-xs text-muted-foreground">
        <Context usedTokens={Math.max(usedTokens, 1)} maxTokens={4096}>
          <ContextTrigger asChild>
            <Button type="button" variant="outline" size="sm" className="rounded-full">
              Tokens: {usedTokens || "Unknown"}
            </Button>
          </ContextTrigger>
          <ContextContent>
            <ContextContentHeader />
            <ContextContentBody>
              <p>
                Input:{" "}
                {result.prompt_tokens === null ? "Unknown" : result.prompt_tokens}
              </p>
              <p>
                Output:{" "}
                {result.completion_tokens === null
                  ? "Unknown"
                  : result.completion_tokens}
              </p>
            </ContextContentBody>
          </ContextContent>
        </Context>
        <span>
          Format:{" "}
          {result.validation === "valid"
            ? "Passed"
            : result.validation === "invalid"
              ? "Failed"
              : "Unknown"}
        </span>
      </div>
      {result.validation_detail ? (
        <p className="mt-2 text-sm text-muted-foreground">
          {result.validation_detail}
        </p>
      ) : null}

      <Message from="assistant" className="mt-3">
        <MessageContent className="w-full max-w-none">
          <details>
            <summary className="min-h-11 cursor-pointer py-3 font-medium">
              Advanced output details
            </summary>
            <div className="grid gap-4 pt-2">
              <div data-testid={`demo-structured-${result.model_id}`}>
                <h4 className="mb-2 font-serif text-base">Parsed result</h4>
                <CodeBlock
                  code={JSON.stringify(result.structured_output, null, 2)}
                  language="json"
                  className="rounded-[14px] border border-border"
                >
                  <CodeBlockCopyButton />
                </CodeBlock>
              </div>
              <div data-testid={`demo-raw-${result.model_id}`}>
                <h4 className="mb-2 font-serif text-base">Raw result (JSON)</h4>
                <CodeBlock
                  code={result.raw_json}
                  language="json"
                  className="rounded-[14px] border border-border"
                >
                  <CodeBlockCopyButton />
                </CodeBlock>
              </div>
            </div>
          </details>
        </MessageContent>
      </Message>
    </article>
  );
}

function Fact({ term, value }: { term: string; value: string }) {
  return (
    <div className="grid grid-cols-[7rem_1fr] gap-2">
      <span className="text-muted-foreground">{term}</span>
      <span>{value}</span>
    </div>
  );
}

function plainOutcome(output: Record<string, unknown>): string {
  const action = output.policy_action;
  if (typeof action === "string") {
    if (action === "approve") return "Approve the transaction.";
    if (action === "reject") return "Reject the transaction.";
    return "Send the transaction for review.";
  }
  const direction = output.direction;
  if (typeof direction === "string") {
    return direction === "favorable"
      ? "The variance is favorable."
      : "The variance is unfavorable.";
  }
  const status = output.status;
  if (typeof status === "string") {
    return status === "balanced"
      ? "The cash balances match."
      : "Review the cash exceptions.";
  }
  return "Review the structured result.";
}

function plainReason(output: Record<string, unknown>): string {
  const evidence = output.evidence;
  if (Array.isArray(evidence)) {
    if (typeof evidence[0] === "string") {
      return evidence[0];
    }
    const first = evidence[0];
    if (
      first &&
      typeof first === "object" &&
      "field" in first &&
      "value" in first
    ) {
      return `The output cites ${String(first.field)} with value ${String(first.value)}.`;
    }
  }
  const topDrivers = output.top_drivers;
  if (Array.isArray(topDrivers) && topDrivers.length > 0) {
    return "The result names the largest saved driver first.";
  }
  const exceptions = output.exceptions;
  if (Array.isArray(exceptions)) {
    return exceptions.length > 0
      ? "The result found entries that need review."
      : "The result found no unmatched entries.";
  }
  return "The saved result does not include a plain-language reason.";
}
