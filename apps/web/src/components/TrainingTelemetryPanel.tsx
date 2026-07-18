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
        <div className="panel" data-testid="training-telemetry-not-started">
          <h3>Logs & metrics</h3>
          <div className="banner banner-info" role="status">
            <strong>Not started · no metrics</strong>
            <p style={{ margin: 0 }}>{telemetry.message}</p>
          </div>
        </div>
      );

    case "error":
      return (
        <div className="panel" data-testid="training-telemetry-error">
          <h3>Logs & metrics</h3>
          <div className="banner banner-error" role="alert">
            <strong>Metrics unavailable</strong>
            <p style={{ margin: 0 }}>{telemetry.message}</p>
          </div>
          <h4>Recorded preparation event</h4>
          <ul className="list-plain">
            {telemetry.events.map((event) => (
              <li key={`${event.timestamp}-${event.state}`}>
                <time dateTime={event.timestamp}>{formatTimestamp(event.timestamp)}</time>{" "}
                · <code>{event.state}</code> · {event.message}
              </li>
            ))}
          </ul>
        </div>
      );

    case "precomputed_prior_run":
      return (
        <div className="panel" data-testid="training-telemetry-prior">
          <div className="controls" style={{ justifyContent: "space-between" }}>
            <h3>Logs & metrics</h3>
            <StatusBadge tone="precomputed">Immutable prior-run record</StatusBadge>
          </div>
          <p>{telemetry.message}</p>
          <h4>Prior-run event log</h4>
          <div className="table-wrap">
            <table className="data">
              <thead>
                <tr>
                  <th scope="col">Timestamp (UTC)</th>
                  <th scope="col">State</th>
                  <th scope="col">Event</th>
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
          <h4>Prior-run training metrics</h4>
          <div className="table-wrap">
            <table className="data">
              <thead>
                <tr>
                  <th scope="col">Step</th>
                  <th scope="col">Timestamp (UTC)</th>
                  <th scope="col">Metric</th>
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
