"use client";

import { useState } from "react";
import demoData from "./demo-data.json";

type ModelOut = {
  raw: string;
  schema_valid: boolean;
  errors: string[];
  key_fields_correct: boolean;
  latency_s: number;
} | null;

type DemoCase = {
  task: string;
  difficulty: string;
  input_preview: string;
  expected: Record<string, unknown>;
  base: ModelOut;
  distilled: ModelOut;
};

const cases = demoData as DemoCase[];

function Badge({ ok, yes, no }: { ok: boolean; yes: string; no: string }) {
  return (
    <span className={`demo-badge ${ok ? "demo-ok" : "demo-bad"}`}>
      {ok ? yes : no}
    </span>
  );
}

function ModelCard({ title, out }: { title: string; out: ModelOut }) {
  return (
    <div className="demo-card">
      <h3>{title}</h3>
      {out ? (
        <>
          <div>
            <Badge ok={out.schema_valid} yes="schema valid" no="schema INVALID" />
            <Badge
              ok={out.key_fields_correct}
              yes="decision fields correct"
              no="decision fields wrong"
            />
            <span className="demo-badge">{out.latency_s}s</span>
          </div>
          <pre>{out.raw}</pre>
        </>
      ) : (
        <p className="demo-pending">
          Training run completing; output captured by the same harness once the
          adapter lands.
        </p>
      )}
    </div>
  );
}

export function CompareDemo() {
  const [i, setI] = useState(0);
  const [ran, setRan] = useState(false);
  const c = cases[i];
  return (
    <section className="distillery-feature" id="demo">
      <div className="feature-meta">
        <span>Playground</span>
        <span>Same 0.5B weights, adapter off vs on</span>
      </div>
      <div className="prose-block">
        <p className="section-kicker">Old model vs distilled model</p>
        <p>
          This is the Distillery playground: pick a held-out finance task
          neither model trained on and run it through the student before and
          after distillation. Outputs are real captures from the evaluation
          harness; the base column is the student with the LoRA adapter
          disabled, the distilled column is the identical weights with the
          adapter enabled. Every response is validated against the executable
          oracle.
        </p>
      </div>
      <div className="demo-runbar">
        <select
          className="demo-select"
          value={i}
          onChange={(e) => {
            setI(+e.target.value);
            setRan(false);
          }}
        >
          {cases.map((x, j) => (
            <option key={x.task} value={j}>
              #{j} {x.task.replace(/_/g, " ")} ({x.difficulty})
            </option>
          ))}
        </select>
        <button className="demo-run" onClick={() => setRan(true)}>
          Run both
        </button>
      </div>
      <div className="demo-input">
        <span>INPUT</span>
        <pre>{c.input_preview}…</pre>
      </div>
      {ran ? (
        <>
          <div className="demo-cols">
            <ModelCard title="Base Qwen2.5-0.5B (not distilled)" out={c.base} />
            <ModelCard
              title="Distilled student (LoRA adapter on)"
              out={c.distilled}
            />
          </div>
          <div className="demo-card demo-expected">
            <h3>Oracle expected output</h3>
            <pre>{JSON.stringify(c.expected, null, 1)}</pre>
          </div>
        </>
      ) : (
        <p className="demo-pending">
          Press Run both to compare the two models on this task.
        </p>
      )}
    </section>
  );
}
