import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  formatCi,
  formatCount,
  formatDurationSeconds,
  formatIndex,
  formatRatio,
  formatUnknown,
  formatUsd,
} from "@/lib/demo/format";
import type { DemoModelEntry } from "@/lib/demo/types";
import { cn } from "@/lib/utils";

export function DemoStatsTable({ model }: { model: DemoModelEntry }) {
  const rows: Array<[string, string, string]> = [
    [
      "Model size",
      formatCount(model.stats.advertised_parameter_count),
      "This is the full number of model parameters. Smaller models usually need less memory.",
    ],
    [
      "Adapter size",
      formatCount(model.stats.adapter_parameter_count),
      "This is the number of extra trained parameters. The default keeps the saved value.",
    ],
    [
      "Compression",
      formatRatio(model.stats.compression_ratio),
      "This compares the source and smaller model sizes. A larger ratio means a smaller serving model.",
    ],
    [
      "Training method (recipe)",
      formatUnknown(model.stats.recipe),
      "This controls how the smaller model learned. Auto selected the saved method.",
    ],
    [
      "Source model (teacher)",
      model.stats.teacher
        ? `${model.stats.teacher.id}@${model.stats.teacher.revision.slice(0, 12)}`
        : "Unknown",
      "This is the larger model that supplied answers. The saved version keeps comparisons repeatable.",
    ],
    [
      "Smaller model (student)",
      model.stats.student
        ? `${model.stats.student.id}@${model.stats.student.revision.slice(0, 12)}`
        : "Unknown",
      "This is the model being checked. The saved version will not change on its own.",
    ],
    [
      "Random seed",
      formatUnknown(model.stats.seed),
      "This makes repeated runs easier to compare. Auto sets it when a run starts.",
    ],
    [
      "Data fingerprint",
      formatUnknown(model.stats.data_hash),
      "This identifies the exact data used. Auto records it so the run can be audited.",
    ],
    [
      "Run plan fingerprint",
      formatUnknown(model.stats.manifest_hash),
      "This identifies the exact run setup. Auto records it before training.",
    ],
    [
      "Model file fingerprint",
      formatUnknown(model.stats.artifact_hash),
      "This identifies the saved model file. It changes if the file changes.",
    ],
    [
      "Training time",
      formatDurationSeconds(model.stats.training_duration_seconds),
      "This is the saved run duration. It does not predict every future run.",
    ],
    [
      "Training cost",
      formatUsd(model.stats.training_cost_usd),
      "This is the saved experiment cost. A live run will show its own spending.",
    ],
    [
      "Main score on familiar data (IID)",
      formatIndex(model.stats.iid_primary_index),
      "This measures examples similar to training. Auto checks it against the saved target.",
    ],
    [
      "Likely score range (95% CI)",
      formatCi(model.stats.iid_ci_low, model.stats.iid_ci_high),
      "This range shows sampling uncertainty. A wider range means less certainty.",
    ],
    [
      "Score kept on unfamiliar data (OOD)",
      formatIndex(model.stats.ood_retention),
      "This measures examples that differ from training. Auto uses it to catch brittle gains.",
    ],
    [
      "Result status",
      formatUnknown(model.stats.proof_status),
      "This records whether the model passed every saved check. Unknown values stay unknown.",
    ],
    [
      "Promotion status",
      model.stats.promotion_status,
      "This shows whether the saved checks recommended this model. Auto never promotes an unknown result.",
    ],
  ];

  return (
    <div className="overflow-x-auto rounded-[14px] border border-border">
      <Table className="text-sm">
        <TableHeader>
          <TableRow>
            <TableHead>Field</TableHead>
            <TableHead>Value</TableHead>
            <TableHead>What it changes</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {rows.map(([label, value, helper]) => {
            const unknown = value.toLowerCase() === "unknown";
            return (
              <TableRow key={label} data-testid="advanced-setting">
                <TableCell className="min-w-48 text-muted-foreground">
                  {label}
                </TableCell>
                <TableCell>
                  <span
                    className={cn(
                      unknown
                        ? "demo-unknown italic text-[var(--unavailable)]"
                        : "font-mono text-xs",
                    )}
                    data-unknown={unknown ? "true" : "false"}
                  >
                    {value}
                  </span>
                </TableCell>
                <TableCell className="min-w-64 text-muted-foreground">
                  {helper}
                </TableCell>
              </TableRow>
            );
          })}
        </TableBody>
      </Table>
    </div>
  );
}
