import { getDemoExample } from "@/lib/demo/examples";
import {
  buildFixturePreviewOutput,
  fixturePreviewLatencyMs,
} from "@/lib/demo/fixturePreview";
import { findRegistryModel } from "@/lib/demo/registry";
import { validateStructuredOutput } from "@/lib/demo/schemas";
import { scoreAgainstGold } from "@/lib/demo/score";
import type {
  DemoInferenceRequest,
  DemoInferenceResponse,
  DemoLiveHealthHttpOk,
  DemoLiveInferHttpOk,
  DemoLiveInferHttpRequest,
  DemoModelArmId,
  DemoModelEntry,
  DemoModelRegistry,
} from "@/lib/demo/types";

/**
 * Typed web-side gateway for Demo/Playground inference.
 * Fixture preview is local and labeled. Live mode calls a serving contract
 * and must surface unavailable/error — never fabricate live outputs.
 */
export interface DemoInferenceGateway {
  infer(
    registry: DemoModelRegistry,
    request: DemoInferenceRequest,
  ): Promise<DemoInferenceResponse>;
}

export type LiveFetch = (
  input: string,
  init?: RequestInit,
) => Promise<Response>;

export function resolveLiveInferenceBaseUrl(
  env: Record<string, string | undefined> = typeof process !== "undefined"
    ? (process.env as Record<string, string | undefined>)
    : {},
): string | null {
  const raw = env.NEXT_PUBLIC_DISTILLERY_INFERENCE_URL?.trim();
  return raw ? raw.replace(/\/$/, "") : null;
}

export class DistilleryDemoGateway implements DemoInferenceGateway {
  constructor(
    private readonly options: {
      liveBaseUrl?: string | null;
      fetchImpl?: LiveFetch;
    } = {},
  ) {}

  async infer(
    registry: DemoModelRegistry,
    request: DemoInferenceRequest,
  ): Promise<DemoInferenceResponse> {
    const model = findRegistryModel(registry, request.model_id);
    if (!model) {
      return {
        status: "unavailable",
        provenance: "none",
        model_id: request.model_id,
        task: request.task,
        example_id: request.example_id,
        code: "MODEL_NOT_IN_REGISTRY",
        message: `Model ${request.model_id} is not present in the registry payload.`,
      };
    }

    if (request.mode === "fixture_preview") {
      return this.fixturePreview(model.arm_id, request);
    }

    return this.liveInfer(model, request);
  }

  private fixturePreview(
    armId: DemoModelArmId,
    request: DemoInferenceRequest,
  ): DemoInferenceResponse {
    const example = request.example_id ? getDemoExample(request.example_id) : null;
    const gold = example?.task === request.task ? example.gold_output : null;
    const structured = buildFixturePreviewOutput(
      armId,
      request.task,
      gold,
      request.input,
    );
    const validation = validateStructuredOutput(request.task, structured);
    const scored = scoreAgainstGold(structured, gold);
    const latency = fixturePreviewLatencyMs(request.model_id, request.example_id);
    return {
      status: "ok",
      provenance: "fixture_preview",
      model_id: request.model_id,
      task: request.task,
      example_id: request.example_id,
      structured_output: structured,
      raw_json: JSON.stringify(structured, null, 2),
      validation: validation.state,
      validation_detail: validation.detail,
      latency_ms: latency,
      prompt_tokens: null,
      completion_tokens: null,
      score: scored.score,
      score_detail: scored.detail,
      label: "Fixture preview — not live model inference",
    };
  }

  private async liveInfer(
    model: DemoModelEntry,
    request: DemoInferenceRequest,
  ): Promise<DemoInferenceResponse> {
    const baseUrl =
      this.options.liveBaseUrl === undefined
        ? resolveLiveInferenceBaseUrl()
        : this.options.liveBaseUrl;

    if (!baseUrl) {
      return {
        status: "unavailable",
        provenance: "none",
        model_id: request.model_id,
        task: request.task,
        example_id: request.example_id,
        code: "SERVING_ENDPOINT_MISSING",
        message:
          "No live serving endpoint is configured (NEXT_PUBLIC_DISTILLERY_INFERENCE_URL).",
      };
    }

    if (
      model.serving.availability !== "live" ||
      !model.serving.artifact_id ||
      !model.serving.endpoint_id
    ) {
      return {
        status: "unavailable",
        provenance: "none",
        model_id: request.model_id,
        task: request.task,
        example_id: request.example_id,
        code: "ARTIFACT_NOT_SERVABLE",
        message:
          model.serving.reason ??
          "Registry does not advertise a live-servable artifact for this model.",
      };
    }

    const payload: DemoLiveInferHttpRequest = {
      model_id: model.model_id,
      artifact_id: model.serving.artifact_id,
      task: request.task,
      example_id: request.example_id,
      input: request.input,
    };

    const fetchImpl = this.options.fetchImpl ?? fetch;
    try {
      const response = await fetchImpl(`${baseUrl}/v1/demo/infer`, {
        method: "POST",
        headers: {
          Accept: "application/json",
          "Content-Type": "application/json",
        },
        body: JSON.stringify(payload),
      });
      if (!response.ok) {
        return {
          status: "error",
          provenance: "none",
          model_id: request.model_id,
          task: request.task,
          example_id: request.example_id,
          code: "LIVE_TRANSPORT_ERROR",
          message: `Live inference failed with HTTP ${response.status}.`,
          retryable: response.status >= 500,
        };
      }
      const body = (await response.json()) as DemoLiveInferHttpOk;
      if (!body || typeof body !== "object" || !body.structured_output) {
        return {
          status: "error",
          provenance: "none",
          model_id: request.model_id,
          task: request.task,
          example_id: request.example_id,
          code: "LIVE_RESPONSE_INVALID",
          message: "Live inference response did not include structured_output.",
          retryable: false,
        };
      }
      const structured = body.structured_output;
      const validation = validateStructuredOutput(request.task, structured);
      const example = request.example_id ? getDemoExample(request.example_id) : null;
      const gold = example?.task === request.task ? example.gold_output : null;
      const scored = scoreAgainstGold(structured, gold);
      return {
        status: "ok",
        provenance: "live",
        model_id: request.model_id,
        task: request.task,
        example_id: request.example_id,
        structured_output: structured,
        raw_json: JSON.stringify(structured, null, 2),
        validation: validation.state,
        validation_detail: validation.detail,
        latency_ms: body.latency_ms,
        prompt_tokens: body.prompt_tokens,
        completion_tokens: body.completion_tokens,
        score: scored.score,
        score_detail: scored.detail,
        label: "Live model inference",
      };
    } catch (error) {
      return {
        status: "error",
        provenance: "none",
        model_id: request.model_id,
        task: request.task,
        example_id: request.example_id,
        code: "LIVE_TRANSPORT_ERROR",
        message: error instanceof Error ? error.message : "Live inference transport failed.",
        retryable: true,
      };
    }
  }
}

export async function probeLiveDemoHealth(
  baseUrl: string | null,
  fetchImpl: LiveFetch = fetch,
): Promise<DemoLiveHealthHttpOk | null> {
  if (!baseUrl) return null;
  try {
    const response = await fetchImpl(`${baseUrl.replace(/\/$/, "")}/v1/demo/health`, {
      method: "GET",
      headers: { Accept: "application/json" },
    });
    if (!response.ok) return null;
    return (await response.json()) as DemoLiveHealthHttpOk;
  } catch {
    return null;
  }
}

export function createDemoGateway(options?: {
  liveBaseUrl?: string | null;
  fetchImpl?: LiveFetch;
}): DemoInferenceGateway {
  return new DistilleryDemoGateway(options);
}
