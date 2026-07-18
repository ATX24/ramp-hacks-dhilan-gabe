"use client";

import { useMemo, useState } from "react";
import data from "./compare-data.json";

type Out = {
  raw: string;
  schema_valid: boolean;
  key_fields_correct: boolean;
  latency_s: number;
};

type Case = {
  i: number;
  task: string;
  difficulty: string;
  input_preview: string;
  expected: Record<string, unknown>;
  base: Out;
  distilled: Out;
};

const cases = data as Case[];

function Badge({ ok, yes, no }: { ok: boolean; yes: string; no: string }) {
  return (
    <span
      className={`inline-block border px-2 py-0.5 font-mono text-[11px] uppercase tracking-wider ${
        ok ? "border-green-700 text-green-700" : "border-[#d75a3d] text-[#d75a3d]"
      }`}
    >
      {ok ? yes : no}
    </span>
  );
}

function Card({ title, out }: { title: string; out: Out }) {
  return (
    <div className="border border-black/80 bg-[#fbfaf6] p-5">
      <h3 className="mb-3 font-serif text-lg">{title}</h3>
      <div className="mb-3 flex flex-wrap gap-2">
        <Badge ok={out.schema_valid} yes="schema valid" no="schema invalid" />
        <Badge
          ok={out.key_fields_correct}
          yes="decision fields correct"
          no="decision fields wrong"
        />
        <span className="inline-block border border-black/20 px-2 py-0.5 font-mono text-[11px] text-black/60">
          {out.latency_s}s
        </span>
      </div>
      <pre className="max-h-80 overflow-auto whitespace-pre-wrap bg-black/5 p-3 font-mono text-[12px] leading-5">
        {out.raw}
      </pre>
    </div>
  );
}

export function ModelCompare() {
  const [taskFilter, setTaskFilter] = useState("all");
  const [i, setI] = useState(0);

  const filtered = useMemo(
    () => cases.filter((c) => taskFilter === "all" || c.task === taskFilter),
    [taskFilter],
  );
  const c = filtered[Math.min(i, filtered.length - 1)];

  const tally = useMemo(() => {
    const t = { base_valid: 0, base_ok: 0, dist_valid: 0, dist_ok: 0 };
    for (const x of cases) {
      t.base_valid += +x.base.schema_valid;
      t.base_ok += +x.base.key_fields_correct;
      t.dist_valid += +x.distilled.schema_valid;
      t.dist_ok += +x.distilled.key_fields_correct;
    }
    return t;
  }, []);

  if (!c) return null;

  return (
    <div className="mt-12">
      <div className="mb-8 grid gap-3 border border-black/80 bg-[#fbfaf6] p-5 md:grid-cols-4">
        {[
          ["base, schema valid", `${tally.base_valid}/${cases.length}`],
          ["base, decisions correct", `${tally.base_ok}/${cases.length}`],
          ["distilled, schema valid", `${tally.dist_valid}/${cases.length}`],
          ["distilled, decisions correct", `${tally.dist_ok}/${cases.length}`],
        ].map(([label, v], idx) => (
          <div key={label}>
            <p className="font-mono text-[10px] uppercase tracking-[0.2em] text-black/50">
              {label}
            </p>
            <p
              className={`mt-1 font-serif text-3xl ${idx >= 2 ? "text-[#d75a3d]" : ""}`}
            >
              {v}
            </p>
          </div>
        ))}
      </div>

      <div className="mb-6 flex flex-wrap items-center gap-3">
        <select
          className="border border-black/40 bg-[#fbfaf6] px-3 py-2 font-mono text-[12px]"
          value={taskFilter}
          onChange={(e) => {
            setTaskFilter(e.target.value);
            setI(0);
          }}
        >
          <option value="all">all tasks</option>
          <option value="transaction_review">transaction review</option>
          <option value="variance_analysis">variance analysis</option>
          <option value="cash_reconciliation">cash reconciliation</option>
        </select>
        <select
          className="max-w-[60%] border border-black/40 bg-[#fbfaf6] px-3 py-2 font-mono text-[12px]"
          value={Math.min(i, filtered.length - 1)}
          onChange={(e) => setI(+e.target.value)}
        >
          {filtered.map((x, j) => (
            <option key={x.i} value={j}>
              #{x.i} {x.task.replace(/_/g, " ")} ({x.difficulty})
            </option>
          ))}
        </select>
        <button
          className="border border-black bg-black px-4 py-2 font-mono text-[12px] uppercase tracking-wider text-[#f1eee6] hover:bg-[#d75a3d] hover:border-[#d75a3d]"
          onClick={() => setI((i + 1) % filtered.length)}
        >
          Next prompt
        </button>
      </div>

      <div className="mb-6 border border-black/20 bg-black/5 p-4">
        <p className="mb-2 font-mono text-[10px] uppercase tracking-[0.2em] text-black/50">
          input
        </p>
        <pre className="max-h-40 overflow-auto whitespace-pre-wrap font-mono text-[12px] leading-5 text-black/70">
          {c.input_preview}…
        </pre>
      </div>

      <div className="grid gap-6 md:grid-cols-2">
        <Card title="Base Qwen2.5-0.5B (adapter off)" out={c.base} />
        <Card title="Distilled student (adapter on)" out={c.distilled} />
      </div>

      <div className="mt-6 border border-green-800/40 bg-[#fbfaf6] p-5">
        <h3 className="mb-3 font-serif text-lg">Oracle expected output</h3>
        <pre className="max-h-64 overflow-auto whitespace-pre-wrap bg-black/5 p-3 font-mono text-[12px] leading-5">
          {JSON.stringify(c.expected, null, 1)}
        </pre>
      </div>
    </div>
  );
}
