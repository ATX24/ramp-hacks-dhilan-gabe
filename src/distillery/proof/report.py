"""JSON and self-contained HTML proof report rendering."""

from __future__ import annotations

import html
import json
from typing import Any

from distillery.contracts.hashing import content_sha256
from distillery.contracts.proof import ProofReport
from distillery.proof.evaluate import ProofEvaluationResult
from distillery.proof.systems import SystemsSummary


def render_json_report(
    result: ProofEvaluationResult | ProofReport,
    *,
    systems: dict[str, SystemsSummary] | None = None,
) -> dict[str, Any]:
    """Render an immutable JSON report document."""
    if isinstance(result, ProofEvaluationResult):
        report = result.report
        systems = systems or result.systems
        gate = result.gate_evaluation.to_dict()
        uncertainty = result.uncertainty
        economics = result.economics
    else:
        report = result
        gate = {
            "proof_status": report.proof_status.value,
            "first_failed_gate": report.first_failed_gate,
            "unevaluated_gates": list(report.unevaluated_gates),
            "quality_gates": [g.model_dump(mode="json") for g in report.quality_gates],
            "evidence_needed": [],
        }
        uncertainty = report.uncertainty
        economics = report.economics

    payload: dict[str, Any] = {
        "schema_version": report.schema_version,
        "report_id": report.report_id,
        "run_ids": list(report.run_ids),
        "protocol_id": report.protocol_id,
        "protocol_sha256": report.protocol_sha256,
        "proof_status": report.proof_status.value,
        "first_failed_gate": report.first_failed_gate,
        "unevaluated_gates": list(report.unevaluated_gates),
        "arm_results": [a.model_dump(mode="json") for a in report.arm_results],
        "quality_gates": [g.model_dump(mode="json") for g in report.quality_gates],
        "gate_evaluation": gate,
        "uncertainty": uncertainty,
        "economics": economics,
        "systems": {k: v.to_dict() for k, v in (systems or {}).items()},
        "exclusions": list(report.exclusions),
        "limitations": list(report.limitations),
        "created_at": report.created_at.isoformat(),
        "resource_hash": report.resource_hash(),
    }
    payload["document_sha256"] = content_sha256(
        {k: v for k, v in payload.items() if k != "document_sha256"}
    )
    return payload


def _esc(value: Any) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def render_html_report(
    result: ProofEvaluationResult | ProofReport,
    *,
    systems: dict[str, SystemsSummary] | None = None,
) -> str:
    """Self-contained HTML report (no external CSS/JS/assets)."""
    doc = render_json_report(result, systems=systems)
    status = doc["proof_status"]
    arms_rows = []
    for arm in doc["arm_results"]:
        arms_rows.append(
            "<tr>"
            f"<td>{_esc(arm['arm_id'])}</td>"
            f"<td>{_esc(arm.get('primary_index'))}</td>"
            f"<td>{_esc(arm.get('excluded'))}</td>"
            f"<td><code>{_esc((arm.get('prediction_sha256') or '')[:16])}</code></td>"
            "</tr>"
        )
    gate_rows = []
    for g in doc["quality_gates"]:
        gate_rows.append(
            "<tr>"
            f"<td>{_esc(g['gate_id'])}</td>"
            f"<td>{_esc(g['evaluated'])}</td>"
            f"<td>{_esc(g['passed'])}</td>"
            f"<td>{_esc(g['detail'])}</td>"
            "</tr>"
        )

    eco = doc.get("economics") or {}
    util_rows = []
    for row in eco.get("utilization_rows") or []:
        student = row.get("student_cost_per_request_usd") or {}
        util_rows.append(
            "<tr>"
            f"<td>{_esc(row.get('utilization'))}</td>"
            f"<td>{_esc(student.get('amount_usd'))} "
            f"<em>({_esc(student.get('kind'))})</em></td>"
            f"<td>{_esc(row.get('savings_per_request_usd'))}</td>"
            f"<td>{_esc(row.get('break_even_requests'))}</td>"
            "</tr>"
        )

    systems_blocks = []
    for arm_id, sys in (doc.get("systems") or {}).items():
        systems_blocks.append(
            f"<h3>Systems · {_esc(arm_id)}</h3>"
            "<ul>"
            f"<li>hardware: {_esc(sys.get('hardware'))}</li>"
            f"<li>batch: {_esc(sys.get('batch_size'))}</li>"
            f"<li>p50 latency: {_esc((sys.get('latency_p50_ms') or {}).get('value'))} ms "
            f"({_esc((sys.get('latency_p50_ms') or {}).get('kind'))})</li>"
            f"<li>p95 latency: {_esc((sys.get('latency_p95_ms') or {}).get('value'))} ms "
            f"({_esc((sys.get('latency_p95_ms') or {}).get('kind'))})</li>"
            f"<li>throughput: {_esc((sys.get('requests_per_second') or {}).get('value'))} "
            f"({_esc((sys.get('requests_per_second') or {}).get('kind'))})</li>"
            f"<li>GPU hours: {_esc((sys.get('gpu_hours') or {}).get('value'))} "
            f"({_esc((sys.get('gpu_hours') or {}).get('kind'))})</li>"
            "</ul>"
        )

    limitations = "".join(f"<li>{_esc(x)}</li>" for x in doc.get("limitations") or [])
    evidence = "".join(
        f"<li>{_esc(x)}</li>"
        for x in (doc.get("gate_evaluation") or {}).get("evidence_needed") or []
    )

    # Embed full JSON for immutability / offline re-parse.
    embedded = html.escape(json.dumps(doc, sort_keys=True, indent=2), quote=False)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>ProofReport {_esc(doc['report_id'])}</title>
<style>
:root {{
  --bg: #f6f3ee;
  --ink: #1c1a17;
  --muted: #5c574f;
  --line: #d9d2c5;
  --ok: #1f6b3a;
  --bad: #8b1e1e;
  --warn: #8a5a00;
  --card: #fffdf8;
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0;
  font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
  color: var(--ink);
  background:
    radial-gradient(1200px 500px at 10% -10%, #e7efe6 0%, transparent 55%),
    radial-gradient(900px 400px at 100% 0%, #efe6da 0%, transparent 50%),
    var(--bg);
  line-height: 1.45;
}}
main {{ max-width: 960px; margin: 0 auto; padding: 2.5rem 1.25rem 4rem; }}
header h1 {{
  font-family: "IBM Plex Serif", Georgia, serif;
  font-weight: 600;
  font-size: clamp(1.8rem, 3vw, 2.4rem);
  margin: 0 0 0.35rem;
}}
.sub {{ color: var(--muted); margin-bottom: 1.5rem; }}
.status {{
  display: inline-block;
  font-weight: 700;
  letter-spacing: 0.02em;
  padding: 0.35rem 0.7rem;
  border: 2px solid var(--ink);
  background: var(--card);
}}
.status.proved {{ color: var(--ok); }}
.status.do_not_distill {{ color: var(--warn); }}
.status.failed_quality, .status.failed_economics {{ color: var(--bad); }}
.status.insufficient_evidence {{ color: var(--warn); }}
section {{
  margin-top: 1.75rem;
  padding-top: 1rem;
  border-top: 1px solid var(--line);
}}
h2 {{ font-size: 1.15rem; margin: 0 0 0.75rem; }}
table {{ width: 100%; border-collapse: collapse; font-size: 0.92rem; }}
th, td {{
  text-align: left;
  padding: 0.45rem 0.4rem;
  border-bottom: 1px solid var(--line);
  vertical-align: top;
}}
th {{ color: var(--muted); font-weight: 600; }}
code, pre {{
  font-family: "IBM Plex Mono", ui-monospace, monospace;
  font-size: 0.84rem;
}}
pre {{
  background: var(--card);
  border: 1px solid var(--line);
  padding: 0.75rem;
  overflow: auto;
  max-height: 320px;
}}
.note {{ color: var(--muted); font-size: 0.9rem; }}
.badge {{
  font-size: 0.75rem;
  text-transform: uppercase;
  letter-spacing: 0.04em;
  color: var(--muted);
}}
</style>
</head>
<body>
<main>
<header>
  <div class="badge">Distillery · ProofReport</div>
  <h1>TinyFable proof</h1>
  <p class="sub">Report <code>{_esc(doc['report_id'])}</code> · protocol
  <code>{_esc(doc['protocol_id'])}</code></p>
  <div class="status { _esc(status) }">{_esc(status)}</div>
</header>

<section>
  <h2>Gate outcome</h2>
  <p>First failed gate: <strong>{_esc(doc.get('first_failed_gate') or 'none')}</strong></p>
  <p>Unevaluated gates:
    <strong>{_esc(', '.join(doc.get('unevaluated_gates') or []) or 'none')}</strong>
  </p>
  <table>
    <thead><tr><th>Gate</th><th>Evaluated</th><th>Passed</th><th>Detail</th></tr></thead>
    <tbody>{''.join(gate_rows)}</tbody>
  </table>
  <h3>Evidence needed to change status</h3>
  <ul>{evidence or '<li>None recorded</li>'}</ul>
</section>

<section>
  <h2>Arms</h2>
  <table>
    <thead><tr><th>Arm</th><th>Primary index</th><th>Excluded</th><th>Pred SHA</th></tr></thead>
    <tbody>{''.join(arms_rows)}</tbody>
  </table>
  <p class="note">Cash reconciliation is diagnostic and excluded from the primary index
  unless promoted by a pre-run decision-log entry.</p>
</section>

<section>
  <h2>Economics</h2>
  <p>Gross experiment cost:
    <strong>{_esc((eco.get('gross_experiment_cost_usd') or {}).get('amount_usd'))}</strong>
    <em>({_esc((eco.get('gross_experiment_cost_usd') or {}).get('kind'))})</em>
  </p>
  <p>Quality retention: <strong>{_esc(eco.get('quality_retention'))}</strong> ·
     Recovered teacher gap: <strong>{_esc(eco.get('recovered_teacher_gap'))}</strong>
     (defined={_esc(eco.get('recovered_teacher_gap_defined'))})</p>
  <p>Break-even @25% util:
     <strong>{_esc((eco.get('break_even_at_25pct') or {}).get('break_even_requests'))}</strong>
     · student CPR kind:
     <em>{_esc((eco.get('break_even_at_25pct') or {}).get('student_cost_kind'))}</em></p>
  <table>
    <thead>
      <tr>
        <th>Utilization</th><th>Student CPR</th>
        <th>Savings</th><th>Break-even</th>
      </tr>
    </thead>
    <tbody>{''.join(util_rows) or '<tr><td colspan="4">No utilization rows</td></tr>'}</tbody>
  </table>
  <p class="note">Serving costs are <strong>projected</strong>, not measured production savings.</p>
</section>

<section>
  <h2>Systems</h2>
  {''.join(systems_blocks) or '<p class="note">No systems profiles attached.</p>'}
</section>

<section>
  <h2>Limitations</h2>
  <ul>{limitations or '<li>None</li>'}</ul>
</section>

<section>
  <h2>Immutable JSON</h2>
  <p class="note">document_sha256=<code>{_esc(doc.get('document_sha256'))}</code></p>
  <pre>{embedded}</pre>
</section>
</main>
</body>
</html>
"""
