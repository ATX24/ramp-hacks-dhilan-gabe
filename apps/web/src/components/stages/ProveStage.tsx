import { ErrorBanner } from "@/components/ErrorBanner";
import { StatusBadge, proofTone } from "@/components/StatusBadge";
import { STAGE_PLAIN } from "@/lib/plainLanguage";
import type { ErrorPayload, ProofReportView } from "@/lib/types";

function formatIndex(value: number | null): string {
  return value === null ? "—" : value.toFixed(3);
}

export function ProveStage({
  proof,
  error,
}: {
  proof: ProofReportView | null;
  error: ErrorPayload | null;
}) {
  if (!proof) {
    return (
      <section aria-labelledby="prove-heading">
        <div className="panel">
          <p className="text-kicker">Check the result</p>
          <h2 id="prove-heading">Prove</h2>
          <ErrorBanner error={error} />
          <p data-testid="prove-empty">
            No proof report is available yet. Complete curated data and a sealed run, or
            open precomputed / insufficient_evidence fixture modes.
          </p>
        </div>
      </section>
    );
  }

  const servingEconomicsLabel = proof.economics.serving_cost_projected
    ? "Projected serving economics"
    : "Measured serving economics";
  const servingValueLabel = proof.economics.serving_cost_projected
    ? "Projected $/request"
    : "Measured $/request";

  return (
    <section aria-labelledby="prove-heading">
      <div className="panel">
        <p className="text-kicker">{STAGE_PLAIN.prove.plain}</p>
        <h2 id="prove-heading">Prove</h2>
        <p>{STAGE_PLAIN.prove.description}</p>
        <p className="text-sm text-muted-foreground">
          Why this matters: {STAGE_PLAIN.prove.why} Advanced confidence intervals,
          OOD retention, systems metrics, and economics stay available below.
        </p>
        <ErrorBanner error={error} />
        <div className="meta-row">
          <span>
            Report <code>{proof.report_id}</code>
          </span>
          <span>
            Protocol <code>{proof.protocol_sha256}</code>
          </span>
          <StatusBadge tone={proofTone(proof.proof_status)}>
            {proof.proof_status}
          </StatusBadge>
          {proof.precomputed ? (
            <StatusBadge tone="precomputed">Precomputed</StatusBadge>
          ) : null}
          <StatusBadge
            tone={proof.economics.serving_cost_projected ? "projected" : "pass"}
          >
            {servingEconomicsLabel}
          </StatusBadge>
        </div>
        {proof.first_failed_gate ? (
          <p>
            First failed gate: <code>{proof.first_failed_gate}</code>
          </p>
        ) : (
          <p>No failed gate recorded on this fixture report.</p>
        )}
        {proof.unevaluated_gates.length > 0 ? (
          <p>
            Unevaluated gates:{" "}
            {proof.unevaluated_gates.map((gate) => (
              <code key={gate} style={{ marginRight: "0.4rem" }}>
                {gate}
              </code>
            ))}
          </p>
        ) : null}
      </div>

      <div className="panel">
        <h3>Arm comparison</h3>
        <div className="table-wrap">
          <table className="data">
            <thead>
              <tr>
                <th scope="col">Arm</th>
                <th scope="col">Purpose</th>
                <th scope="col">Primary index</th>
                <th scope="col">95% CI</th>
                <th scope="col">OOD retention</th>
                <th scope="col">Status</th>
              </tr>
            </thead>
            <tbody>
              {proof.arms.map((arm) => (
                <tr key={arm.arm_id}>
                  <td>
                    <code>{arm.arm_id}</code>
                  </td>
                  <td>{arm.purpose}</td>
                  <td>{formatIndex(arm.primary_index)}</td>
                  <td>
                    {arm.ci_low === null || arm.ci_high === null
                      ? "—"
                      : `[${arm.ci_low.toFixed(3)}, ${arm.ci_high.toFixed(3)}]`}
                  </td>
                  <td>{formatIndex(arm.ood_retention)}</td>
                  <td>
                    {arm.excluded ? (
                      <StatusBadge tone="unavailable">excluded</StatusBadge>
                    ) : (
                      <StatusBadge tone="pass">included</StatusBadge>
                    )}
                    {arm.exclusion_reason ? (
                      <div style={{ marginTop: "0.35rem" }}>{arm.exclusion_reason}</div>
                    ) : null}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div className="grid-2">
        <div className="panel">
          <h3>Systems metrics</h3>
          {proof.systems ? (
            <div className="table-wrap">
              <table className="data">
                <thead>
                  <tr>
                    <th scope="col">Metric</th>
                    <th scope="col">Value</th>
                    <th scope="col">Source</th>
                  </tr>
                </thead>
                <tbody>
                  {[
                    ["p50 latency", `${proof.systems.p50_latency_ms} ms`],
                    ["p95 latency", `${proof.systems.p95_latency_ms} ms`],
                    [
                      "Throughput batch1",
                      `${proof.systems.throughput_rps_batch1} rps`,
                    ],
                    [
                      "Throughput batch8",
                      `${proof.systems.throughput_rps_batch8} rps`,
                    ],
                    ["Peak VRAM", `${proof.systems.peak_vram_gib} GiB`],
                    ["GPU hours", String(proof.systems.gpu_hours)],
                    ["Hardware", proof.systems.hardware],
                  ].map(([metric, value]) => (
                    <tr key={metric}>
                      <td>{metric}</td>
                      <td>{value}</td>
                      <td>
                        {proof.systems?.measurement_source ===
                        "precomputed_prior_run" ? (
                          <StatusBadge tone="precomputed">
                            Prior-run precomputed measurement
                          </StatusBadge>
                        ) : (
                          <StatusBadge tone="pending">
                            Static fixture measurement
                          </StatusBadge>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <p>Systems profile unavailable.</p>
          )}
        </div>

        <div className="panel">
          <h3>
            Economics{" "}
            <StatusBadge
              tone={proof.economics.serving_cost_projected ? "projected" : "pass"}
            >
              {proof.economics.serving_cost_projected ? "Projected serving" : "Measured serving"}
            </StatusBadge>
          </h3>
          <ul className="list-plain">
            <li>
              Gross experiment cost: ${proof.economics.gross_experiment_cost_usd.toFixed(2)}
            </li>
            <li>
              Quality retention:{" "}
              {proof.economics.quality_retention === null
                ? "—"
                : proof.economics.quality_retention.toFixed(3)}
            </li>
            <li>
              Recovered teacher gap:{" "}
              {proof.economics.recovered_teacher_gap === null
                ? "—"
                : proof.economics.recovered_teacher_gap.toFixed(3)}
            </li>
            <li>
              Break-even requests:{" "}
              {proof.economics.break_even_requests === null
                ? "—"
                : String(proof.economics.break_even_requests)}
            </li>
          </ul>
          <p>{proof.economics.note}</p>
          <div className="table-wrap">
            <table className="data">
              <thead>
                <tr>
                  <th scope="col">Utilization</th>
                  <th scope="col">{servingValueLabel}</th>
                </tr>
              </thead>
              <tbody>
                {proof.economics.utilization_sensitivity.map((row) => (
                  <tr key={row.utilization}>
                    <td>{Math.round(row.utilization * 100)}%</td>
                    <td>${row.cost_per_request_usd.toFixed(4)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>

      <div className="grid-2">
        <div className="panel">
          <h3>Limitations</h3>
          <ul className="list-plain">
            {proof.limitations.map((item) => (
              <li key={item}>{item}</li>
            ))}
          </ul>
        </div>
        <div className="panel">
          <h3>Artifact download metadata</h3>
          <ul className="list-plain">
            {proof.artifact_downloads.map((item) => (
              <li key={item.name}>
                <strong>{item.name}</strong>
                <br />
                <code>{item.uri}</code>
                <br />
                <span className="hash">{item.sha256}</span>
              </li>
            ))}
          </ul>
        </div>
      </div>
    </section>
  );
}
