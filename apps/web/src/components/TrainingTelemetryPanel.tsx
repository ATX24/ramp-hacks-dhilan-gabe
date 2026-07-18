import { StatusBadge } from "@/components/StatusBadge";
import type { TrainingTelemetry } from "@/lib/types";

function formatTimestamp(timestamp: string): string {
  return new Intl.DateTimeFormat("en-US", {
    dateStyle: "medium",
    timeStyle: "medium",
    timeZone: "UTC",
  }).format(new Date(timestamp));
}

export function TrainingTelemetryPanel({
  telemetry,
}: {
  telemetry: TrainingTelemetry;
}) {
  switch (telemetry.provenance) {
    case "not_started":
      return (
        <div className="grid gap-3 pt-3" data-testid="training-telemetry-not-started">
          <h3>What happened during the job</h3>
          <div className="banner banner-info" role="status">
            <strong>The job has not started</strong>
            <p style={{ margin: 0 }}>{telemetry.message}</p>
          </div>
        </div>
      );

    case "error":
      return (
        <div className="grid gap-3 pt-3" data-testid="training-telemetry-error">
          <h3>What happened during the job</h3>
          <div className="banner banner-error" role="alert">
            <strong>There are no measurements</strong>
            <p style={{ margin: 0 }}>{telemetry.message}</p>
          </div>
          <h4>Saved setup event</h4>
          <ul className="list-plain">
            {telemetry.events.map((event) => (
              <li key={`${event.timestamp}-${event.state}`}>
                <time dateTime={event.timestamp}>{formatTimestamp(event.timestamp)}</time>{" "}
                . <code>{event.state}</code>. {event.message}
              </li>
            ))}
          </ul>
        </div>
      );

    case "precomputed_prior_run":
      return (
        <div className="grid gap-3 pt-3" data-testid="training-telemetry-prior">
          <div className="controls" style={{ justifyContent: "space-between" }}>
            <h3>What happened during the job</h3>
            <StatusBadge tone="precomputed">Saved run record</StatusBadge>
          </div>
          <p>{telemetry.message}</p>
          <p>
            The event log lists each saved state change. The measurements below show
            how the earlier training job changed over time.
          </p>
          <h4>Saved event log</h4>
          <div className="table-wrap">
            <table className="data">
              <thead>
                <tr>
                  <th scope="col">Time (UTC)</th>
                  <th scope="col">State</th>
                  <th scope="col">What happened</th>
                </tr>
              </thead>
              <tbody>
                {telemetry.events.map((event) => (
                  <tr key={`${event.timestamp}-${event.state}`}>
                    <td>
                      <time dateTime={event.timestamp}>
                        {formatTimestamp(event.timestamp)}
                      </time>
                    </td>
                    <td>
                      <code>{event.state}</code>
                    </td>
                    <td>{event.message}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <h4>Saved training measurements (metrics)</h4>
          <div className="table-wrap">
            <table className="data">
              <thead>
                <tr>
                  <th scope="col">Step</th>
                  <th scope="col">Time (UTC)</th>
                  <th scope="col">Measurement</th>
                  <th scope="col">Value</th>
                </tr>
              </thead>
              <tbody>
                {telemetry.metrics.map((metric) => (
                  <tr key={`${metric.step}-${metric.name}`}>
                    <td>{metric.step}</td>
                    <td>
                      <time dateTime={metric.timestamp}>
                        {formatTimestamp(metric.timestamp)}
                      </time>
                    </td>
                    <td>
                      <code>{metric.name}</code>
                    </td>
                    <td>
                      {metric.value}
                      {metric.unit ? ` ${metric.unit}` : ""}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      );

    default: {
      const _exhaustive: never = telemetry;
      return _exhaustive;
    }
  }
}
