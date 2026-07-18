import { StatusBadge, gateTone } from "@/components/StatusBadge";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import type { PreflightGate } from "@/lib/types";

export function GateList({ gates }: { gates: PreflightGate[] }) {
  return (
    <div className="table-wrap" role="region" aria-label="Safety checks">
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>Check</TableHead>
            <TableHead>Status</TableHead>
            <TableHead>What it means</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {gates.map((gate) => (
            <TableRow key={gate.gate_id}>
              <TableCell>{gate.label}</TableCell>
              <TableCell>
                <StatusBadge tone={gateTone(gate.status)}>
                  {gate.status === "pass"
                    ? "Passed"
                    : gate.status === "fail"
                      ? "Failed"
                      : gate.status === "pending"
                        ? "Waiting"
                        : "Not available"}
                </StatusBadge>
              </TableCell>
              <TableCell>{gate.detail}</TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </div>
  );
}
