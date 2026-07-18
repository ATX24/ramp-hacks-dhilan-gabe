import { StatusBadge, gateTone } from "@/components/StatusBadge";
import type { PreflightGate } from "@/lib/types";

export function GateList({ gates }: { gates: PreflightGate[] }) {
  return (
    <div className="table-wrap" role="region" aria-label="Preflight gates">
      <table className="data">
        <thead>
          <tr>
            <th scope="col">Gate</th>
            <th scope="col">Status</th>
            <th scope="col">Detail</th>
          </tr>
        </thead>
        <tbody>
          {gates.map((gate) => (
            <tr key={gate.gate_id}>
              <td>{gate.label}</td>
              <td>
                <StatusBadge tone={gateTone(gate.status)}>{gate.status}</StatusBadge>
              </td>
              <td>{gate.detail}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
