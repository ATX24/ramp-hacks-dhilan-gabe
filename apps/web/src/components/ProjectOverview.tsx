"use client";

import Link from "next/link";
import { useEffect, useMemo, useRef, useState } from "react";
import {
  CheckIcon,
  ChevronDownIcon,
  CircleIcon,
  FileUpIcon,
  RefreshCwIcon,
} from "lucide-react";
import { CodeBlock, CodeBlockCopyButton } from "@/components/ai-elements/code-block";
import { Loader } from "@/components/ai-elements/loader";
import {
  PromptInput,
  PromptInputBody,
  PromptInputFooter,
  PromptInputSubmit,
  PromptInputTextarea,
} from "@/components/ai-elements/prompt-input";
import { Suggestion, Suggestions } from "@/components/ai-elements/suggestion";
import { ModeSwitcher } from "@/components/ModeSwitcher";
import { StatusBadge } from "@/components/StatusBadge";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Panel,
  PanelBody,
  PanelDescription,
  PanelHeader,
  PanelTitle,
} from "@/components/ui/panel";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import { defaultExampleForTask } from "@/lib/demo/examples";
import { createDemoGateway } from "@/lib/demo/gateway";
import { buildDemoModelRegistry } from "@/lib/demo/registry";
import type {
  DemoInferenceResponse,
  DemoModelEntry,
  DemoPortfolioEntry,
} from "@/lib/demo/types";
import { buildStageHref } from "@/lib/navigation";
import {
  DISTILLATION_PRIORITIES,
  SAFE_EXAMPLE_GOALS,
  resolveAutoDistillationPlan,
  type DistillationPriority,
} from "@/lib/modelPortfolio";
import type { StageBundle } from "@/lib/types";
import { cn } from "@/lib/utils";

type DataChoice = "sample" | "upload";

const SAMPLE_DATA = [
  {
    id: "finance_mix",
    name: "Mixed finance sample",
    detail: "560 made-up records across the supported finance jobs.",
  },
  {
    id: "transaction_review",
    name: "Transaction review sample",
    detail: "Made-up purchases, policy checks, and account choices.",
  },
  {
    id: "variance_analysis",
    name: "Budget variance sample",
    detail: "Made-up budget, actual, price, volume, and currency changes.",
  },
] as const;

const JOURNEY = [
  {
    title: "Reading your goal",
    detail: "Distillery is turning your sentence into a testable job.",
  },
  {
    title: "Picking a starting model",
    detail: "The generalist stays selected unless you chose a specialist.",
  },
  {
    title: "Checking the data and limit",
    detail: "Distillery is checking the sample and the spending cap.",
  },
  {
    title: "Writing the check plan",
    detail: "Distillery is deciding how it will compare quality and running cost.",
  },
  {
    title: "The sample plan is ready",
    detail: "This walkthrough is complete. It did not start a paid job.",
  },
] as const;

export function ProjectOverview({
  bundle,
  onRefresh,
}: {
  bundle: StageBundle;
  onRefresh: () => Promise<void>;
}) {
  const registry = useMemo(() => buildDemoModelRegistry(bundle), [bundle]);
  const [goal, setGoal] = useState<string>(SAFE_EXAMPLE_GOALS[0]);
  const [priority, setPriority] = useState<DistillationPriority>("quality");
  const [dataChoice, setDataChoice] = useState<DataChoice>("sample");
  const [sampleId, setSampleId] = useState<string>(SAMPLE_DATA[0].id);
  const [fileName, setFileName] = useState<string | null>(null);
  const [portfolioId, setPortfolioId] = useState("tinyfable_generalist");
  const [advancedOpen, setAdvancedOpen] = useState(false);
  const [phaseIndex, setPhaseIndex] = useState(-1);
  const [goalError, setGoalError] = useState<string | null>(null);
  const [refreshing, setRefreshing] = useState(false);
  const [lastChecked, setLastChecked] = useState<string | null>(null);
  const [comparisonResults, setComparisonResults] = useState<
    DemoInferenceResponse[]
  >([]);
  const [comparisonPending, setComparisonPending] = useState(false);
  const comparisonStarted = useRef(false);

  const selectedPortfolio =
    registry.portfolio.find((item) => item.portfolio_id === portfolioId) ??
    registry.portfolio[0] ??
    null;
  const autoPlan = resolveAutoDistillationPlan(bundle, priority, portfolioId);
  const demoGateway = useMemo(() => createDemoGateway(), []);
  const comparisonCandidates = useMemo(() => {
    const base = registry.models.find((model) => model.arm_id === "student_base");
    const trained =
      registry.models.find((model) => model.arm_id === "promoted_winner") ??
      registry.models.find((model) => model.arm_id === "sequence_kd") ??
      registry.models.find((model) => model.arm_id === "oracle_sft");
    return [base, trained].filter(
      (model): model is DemoModelEntry => model !== undefined,
    );
  }, [registry.models]);
  const running = phaseIndex >= 0 && phaseIndex < JOURNEY.length - 1;
  const complete = phaseIndex === JOURNEY.length - 1;

  useEffect(() => {
    if (!running) return;
    const timer = window.setTimeout(() => {
      setPhaseIndex((current) => Math.min(current + 1, JOURNEY.length - 1));
    }, 650);
    return () => window.clearTimeout(timer);
  }, [phaseIndex, running]);

  useEffect(() => {
    if (!complete || comparisonStarted.current) return;
    comparisonStarted.current = true;
    let cancelled = false;
    const financeExample = defaultExampleForTask("transaction_review");
    setComparisonPending(true);
    void Promise.all(
      comparisonCandidates.map((model) =>
        demoGateway.infer(registry, {
          model_id: model.model_id,
          task: financeExample.task,
          example_id: financeExample.example_id,
          input: financeExample.input,
          mode: "fixture_preview",
        }),
      ),
    ).then((next) => {
      if (cancelled) return;
      setComparisonResults(next);
      setComparisonPending(false);
    });
    return () => {
      cancelled = true;
    };
  }, [
    comparisonCandidates,
    complete,
    demoGateway,
    registry,
  ]);

  function startWalkthrough() {
    if (goal.trim().length < 12) {
      setGoalError("Tell us a little more about the job. One short sentence is enough.");
      return;
    }
    setGoalError(null);
    setComparisonResults([]);
    setComparisonPending(true);
    comparisonStarted.current = false;
    setPhaseIndex(0);
    setLastChecked(
      new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
    );
  }

  async function refreshProject() {
    setRefreshing(true);
    try {
      await onRefresh();
      setLastChecked(
        new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
      );
    } finally {
      setRefreshing(false);
    }
  }

  const sample = SAMPLE_DATA.find((item) => item.id === sampleId) ?? SAMPLE_DATA[0];
  const sourceDescription =
    dataChoice === "sample"
      ? sample.detail
      : fileName
        ? `${fileName} is selected in this browser. The file has not been sent.`
        : "Choose a CSV, JSON, or JSONL file. The sample walkthrough will only show its name.";
  const sdkGoal = goal.trim() || SAFE_EXAMPLE_GOALS[0];
  const sdkCode = [
    'data = distillery.datasets.create("./finance.jsonl")',
    `run = distillery.distill(data, goal=${JSON.stringify(sdkGoal)}, auto=True)`,
    "result = run.wait()",
  ].join("\n");

  return (
    <div className="grid gap-4">
      <Panel>
        <PanelHeader className="pb-3">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <p className="text-kicker text-[var(--orange)]">New project</p>
            <StatusBadge tone="precomputed">Sample only. No live job.</StatusBadge>
          </div>
          <h1 className="max-w-3xl font-serif text-4xl leading-[0.98] tracking-[-0.05em] sm:text-5xl">
            What do you want your smaller model to do?
          </h1>
          <PanelDescription className="max-w-2xl text-base">
            Describe one useful job. Distillery will pick a safe starting point and
            spending limit. It will also decide how to check the result.
          </PanelDescription>
        </PanelHeader>
        <PanelBody className="grid gap-6 lg:grid-cols-[minmax(0,1.35fr)_minmax(18rem,0.65fr)]">
          <div className="grid gap-5">
            <SettingGroup
              label="Goal"
              helper="This tells the model what job to learn. A specific sentence makes the final check more useful. If you keep the example, Distillery will use transaction review."
            >
              <PromptInput
                className="rounded-[14px] border-border bg-card"
                onSubmit={(_message, event) => {
                  event.preventDefault();
                  startWalkthrough();
                }}
              >
                <PromptInputBody>
                  <PromptInputTextarea
                    aria-label="What should the smaller model do?"
                    data-testid="project-goal"
                    value={goal}
                    onChange={(event) => setGoal(event.target.value)}
                    placeholder="For example: Review finance transactions and flag policy exceptions"
                    className="min-h-28 text-base"
                  />
                </PromptInputBody>
                <PromptInputFooter>
                  <span className="text-xs text-muted-foreground">
                    Use a job description, not private data.
                  </span>
                  <PromptInputSubmit
                    className="ml-auto min-h-11 rounded-full px-5"
                    size="sm"
                    status={running ? "submitted" : undefined}
                    disabled={running}
                    data-testid="distill-action"
                    onClick={(event) => {
                      event.preventDefault();
                      startWalkthrough();
                    }}
                  >
                    {running ? "Preparing your plan..." : "Distill my model"}
                  </PromptInputSubmit>
                </PromptInputFooter>
              </PromptInput>
              {goalError ? (
                <Alert
                  role="alert"
                  className="border-[color-mix(in_oklab,var(--fail)_30%,transparent)] bg-[color-mix(in_oklab,var(--fail)_8%,transparent)]"
                >
                  <AlertDescription>{goalError}</AlertDescription>
                </Alert>
              ) : null}
              <Suggestions className="pb-1">
                {SAFE_EXAMPLE_GOALS.map((example) => (
                  <Suggestion
                    key={example}
                    suggestion={example}
                    onClick={setGoal}
                    variant={goal === example ? "default" : "outline"}
                  />
                ))}
              </Suggestions>
            </SettingGroup>

            <div className="grid gap-5 sm:grid-cols-2">
              <SettingGroup
                label="Data"
                helper="This changes what Distillery uses for the walkthrough. The safe sample is made up and sends nothing. Choose a file only if you want to see its name in the plan."
              >
                <ToggleGroup
                  type="single"
                  value={dataChoice}
                  onValueChange={(value) => {
                    if (value === "sample" || value === "upload") {
                      setDataChoice(value);
                    }
                  }}
                  className="justify-start"
                  aria-label="Choose sample data or a local file"
                >
                  <ToggleGroupItem
                    value="sample"
                    className="min-h-11 rounded-full px-4"
                    data-testid="data-sample"
                  >
                    Safe sample
                  </ToggleGroupItem>
                  <ToggleGroupItem
                    value="upload"
                    className="min-h-11 rounded-full px-4"
                    data-testid="data-upload"
                  >
                    <FileUpIcon className="size-4" />
                    Choose a file
                  </ToggleGroupItem>
                </ToggleGroup>
                {dataChoice === "sample" ? (
                  <Select value={sampleId} onValueChange={setSampleId}>
                    <SelectTrigger
                      aria-label="Choose a safe sample"
                      data-testid="sample-select"
                    >
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      {SAMPLE_DATA.map((item) => (
                        <SelectItem key={item.id} value={item.id}>
                          {item.name}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                ) : (
                  <Input
                    type="file"
                    accept=".csv,.json,.jsonl"
                    aria-label="Choose a local data file"
                    onChange={(event) => {
                      setFileName(event.target.files?.[0]?.name ?? null);
                    }}
                  />
                )}
                <p className="text-sm text-muted-foreground">{sourceDescription}</p>
              </SettingGroup>

              <SettingGroup
                label="What matters most?"
                helper="This changes the spending limit and how auto weighs its choices. Pick quality if you are unsure. Distillery will still check speed and cost."
              >
                <ToggleGroup
                  type="single"
                  value={priority}
                  onValueChange={(value) => {
                    if (value === "quality" || value === "speed" || value === "cost") {
                      setPriority(value);
                    }
                  }}
                  className="flex-wrap justify-start"
                  aria-label="Choose what matters most"
                >
                  {DISTILLATION_PRIORITIES.map((item) => (
                    <ToggleGroupItem
                      key={item.id}
                      value={item.id}
                      className="min-h-11 rounded-full px-4"
                      data-testid={`priority-${item.id}`}
                    >
                      {item.label}
                    </ToggleGroupItem>
                  ))}
                </ToggleGroup>
                <p className="text-sm text-muted-foreground">
                  {DISTILLATION_PRIORITIES.find((item) => item.id === priority)
                    ?.description}
                </p>
              </SettingGroup>
            </div>

            <div className="grid gap-3 rounded-[14px] border border-border bg-secondary/35 p-4">
              <div>
                <h2 className="font-serif text-lg">Candidates for this check</h2>
                <p className="mt-1 text-sm text-muted-foreground">
                  The same made-up transaction goes to both models. Distillery will
                  show what changed before it recommends the next check.
                </p>
              </div>
              <div className="grid gap-2 sm:grid-cols-2">
                {comparisonCandidates.map((model) => (
                  <CandidatePreview key={model.model_id} model={model} />
                ))}
              </div>
              <p className="text-sm text-muted-foreground">
                TinyFable Generalist covers every supported finance job. Task
                specialists are backup choices. Distillery will not switch to one
                unless you choose it under Advanced.
              </p>
            </div>

            <p className="text-sm text-muted-foreground">
              Auto will choose the starting model, how to teach it, the spending
              limit, and the final checks. You can review those choices before any live
              work starts.
            </p>
          </div>

          <aside className="grid self-start content-start gap-3 rounded-[18px] bg-[#171714] p-5 text-[#f1eee6]">
            <p className="text-kicker text-[#a6a59f]">The same three steps in code</p>
            <h2 className="font-serif text-2xl tracking-tight">
              The SDK uses the same setup in three lines.
            </h2>
            <CodeBlock
              code={sdkCode}
              language="python"
              className="border-[#3a3b37] bg-[#23231f] [&_pre]:whitespace-pre-wrap [&_code]:break-words"
            >
              <CodeBlockCopyButton />
            </CodeBlock>
            <p className="text-sm text-[#b6b5ae]">
              The page and the SDK use the same auto plan. Neither path hides a model
              switch.
            </p>
          </aside>
        </PanelBody>
      </Panel>

      {phaseIndex >= 0 ? (
        <RunJourney
          bundle={bundle}
          complete={complete}
          currentPhase={phaseIndex}
          lastChecked={lastChecked}
          onRefresh={refreshProject}
          refreshing={refreshing}
          selectedPortfolio={selectedPortfolio}
          comparisonCandidates={comparisonCandidates}
          comparisonPending={comparisonPending}
          comparisonResults={comparisonResults}
        />
      ) : null}

      <Panel>
        <PanelHeader>
          <button
            type="button"
            className="flex w-full items-center justify-between gap-3 text-left"
            aria-expanded={advancedOpen}
            aria-controls="advanced-settings"
            onClick={() => setAdvancedOpen((open) => !open)}
            data-testid="advanced-toggle"
          >
            <span>
              <PanelTitle className="text-xl">Advanced</PanelTitle>
              <PanelDescription className="mt-1">
                Open this to choose a specific model or inspect the full plan. If you
                leave it closed, auto keeps the recommended defaults.
              </PanelDescription>
            </span>
            <ChevronDownIcon
              className={cn("size-5 transition-transform", advancedOpen && "rotate-180")}
            />
          </button>
        </PanelHeader>
        {advancedOpen ? (
          <PanelBody id="advanced-settings" data-testid="advanced-settings">
            <div className="grid gap-6">
              <AdvancedPortfolio
                portfolio={registry.portfolio}
                selectedId={portfolioId}
                onSelect={setPortfolioId}
              />

              <div className="grid gap-3">
                <h3 className="font-serif text-lg">How auto set up this plan</h3>
                <p className="text-sm text-muted-foreground">
                  These values control the run and the final checks. Distillery keeps
                  them here so the main path stays short.
                </p>
                <div className="grid gap-3 md:grid-cols-2">
                  <AdvancedSetting
                    label="Model ID"
                    value={selectedPortfolio?.base_model_id ?? "Not available"}
                    helper="This identifies the small model that receives the new behavior. The generalist is the default because it covers every supported job."
                  />
                  <AdvancedSetting
                    label="Training method (recipe)"
                    value={`auto (${autoPlan.resolvedTechnique})`}
                    helper="This changes how the model learns. Auto checks the available methods and keeps the safe supported choice."
                  />
                  <AdvancedSetting
                    label="Training objective"
                    value="Not listed in this sample"
                    helper="This tells training what error to reduce. Distillery will show the exact objective before a live job starts."
                  />
                  <AdvancedSetting
                    label="Random seed"
                    value="Set when a live run starts"
                    helper="This makes repeated runs easier to compare. Distillery sets it for live work, so leaving the default is usually best."
                  />
                  <AdvancedSetting
                    label="Token limit"
                    value="Not set in this sample"
                    helper="This caps the length of each training example. Auto will use the model's safe limit unless you change it."
                  />
                  <AdvancedSetting
                    label="Hardware"
                    value={`${bundle.plan.planned_job.instance_type} (${bundle.plan.planned_job.backend})`}
                    helper="This changes run time and cost. Auto uses the smallest supported machine that fits the plan."
                  />
                  <AdvancedSetting
                    label="Data fingerprint"
                    value={bundle.dataset.content_sha256}
                    helper="This code identifies the exact data used. It matters when you need to repeat or audit a run. Auto records it for you."
                  />
                  <AdvancedSetting
                    label="Proof fingerprint"
                    value={bundle.proof?.protocol_sha256 ?? "Created after the run"}
                    helper="This code identifies the exact checks used for the result. Auto records it so the checks cannot change after training."
                  />
                </div>
              </div>

              <div className="grid gap-3">
                <h3 className="font-serif text-lg">Limits and final checks</h3>
                <div className="grid gap-3 md:grid-cols-3">
                  <AdvancedSetting
                    label="Spending limit"
                    value={`$${autoPlan.budgetCeilingUsd.toFixed(2)}`}
                    helper="This is the most a live run may spend. Your priority changes it. Auto stops before it exceeds the limit."
                  />
                  <AdvancedSetting
                    label="Time estimate"
                    value={`${autoPlan.etaMinutes.low} to ${autoPlan.etaMinutes.high} minutes`}
                    helper="This is the planned run time, not a promise. Leaving the default lets auto choose the bounded job."
                  />
                  <AdvancedSetting
                    label="Proof plan"
                    value={autoPlan.proofProtocol}
                    helper="These checks decide whether the smaller model is good enough. Auto compares quality and cost before it can recommend a result."
                  />
                </div>
              </div>

              <div className="grid gap-3">
                <h3 className="font-serif text-lg">Saved state</h3>
                <p className="text-sm text-muted-foreground">
                  This changes which saved example the evidence pages show. Leave the
                  default if you only want the setup walkthrough.
                </p>
                <ModeSwitcher current={bundle.mode} />
              </div>

              <EvidenceLinks bundle={bundle} />
            </div>
          </PanelBody>
        ) : null}
      </Panel>
    </div>
  );
}

function SettingGroup({
  label,
  helper,
  children,
}: {
  label: string;
  helper: string;
  children: React.ReactNode;
}) {
  return (
    <div className="grid gap-2">
      <Label className="font-serif text-lg font-normal">{label}</Label>
      <p className="text-sm text-muted-foreground">{helper}</p>
      {children}
    </div>
  );
}

function AdvancedSetting({
  label,
  value,
  helper,
}: {
  label: string;
  value: string;
  helper: string;
}) {
  return (
    <div
      className="rounded-[14px] border border-border bg-card p-3"
      data-testid="advanced-setting"
    >
      <p className="text-sm text-muted-foreground">{label}</p>
      <p className="mt-1 break-words font-mono text-xs text-foreground">{value}</p>
      <p className="mt-2 text-sm text-muted-foreground">{helper}</p>
    </div>
  );
}

function AdvancedPortfolio({
  portfolio,
  selectedId,
  onSelect,
}: {
  portfolio: readonly DemoPortfolioEntry[];
  selectedId: string;
  onSelect: (id: string) => void;
}) {
  return (
    <fieldset className="grid gap-3">
      <legend className="font-serif text-lg">Choose the model shape</legend>
      <p className="text-sm text-muted-foreground">
        This changes whether one model handles every job or one adapter handles a
        single job. The generalist is the default. Specialists only run when you pick
        one here.
      </p>
      <div className="grid gap-2 md:grid-cols-2">
        {portfolio.map((item) => {
          const checked = item.portfolio_id === selectedId;
          return (
            <label
              key={item.portfolio_id}
              className={cn(
                "grid cursor-pointer gap-2 rounded-[14px] border border-border bg-card p-4",
                checked && "border-[var(--orange)] bg-secondary/35",
              )}
            >
              <span className="flex items-center gap-2">
                <input
                  type="radio"
                  name="model-portfolio"
                  value={item.portfolio_id}
                  checked={checked}
                  onChange={() => onSelect(item.portfolio_id)}
                  data-testid={`portfolio-${item.portfolio_id}`}
                />
                <span className="font-serif text-base">{item.display_name}</span>
                {item.recommended ? (
                  <Badge variant="outline" className="rounded-full">
                    Recommended
                  </Badge>
                ) : null}
              </span>
              <span className="text-sm text-muted-foreground">{item.purpose}</span>
              <span className="text-xs text-muted-foreground">
                {item.role === "generalist"
                  ? "Auto picks this unless you make another choice."
                  : "Auto treats this as a backup and will not route to it on its own."}
              </span>
            </label>
          );
        })}
      </div>
    </fieldset>
  );
}

function CandidatePreview({ model }: { model: DemoModelEntry }) {
  const generalist = model.arm_id !== "student_base";
  return (
    <div className="rounded-[12px] border border-border bg-card p-3">
      <div className="flex flex-wrap items-center gap-2">
        <span className="font-serif">
          {generalist ? "TinyFable Generalist" : "Current base model"}
        </span>
        <Badge variant="outline" className="rounded-full">
          {generalist ? "Recommended" : "Starting point"}
        </Badge>
      </div>
      <p className="mt-1 text-sm text-muted-foreground">
        {generalist
          ? "This saved candidate learned the finance job."
          : "This shows what the smaller model does before training."}
      </p>
      <p className="mt-1 text-xs text-muted-foreground">
        Saved demo output is available. No live endpoint will be called.
      </p>
    </div>
  );
}

function RunJourney({
  bundle,
  comparisonCandidates,
  comparisonPending,
  comparisonResults,
  complete,
  currentPhase,
  lastChecked,
  onRefresh,
  refreshing,
  selectedPortfolio,
}: {
  bundle: StageBundle;
  comparisonCandidates: readonly DemoModelEntry[];
  comparisonPending: boolean;
  comparisonResults: readonly DemoInferenceResponse[];
  complete: boolean;
  currentPhase: number;
  lastChecked: string | null;
  onRefresh: () => Promise<void>;
  refreshing: boolean;
  selectedPortfolio: DemoPortfolioEntry | null;
}) {
  const current = JOURNEY[currentPhase] ?? JOURNEY[0];
  return (
    <Panel data-testid="project-run">
      <PanelHeader>
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div>
            <p className="text-kicker text-[var(--orange)]">Project status</p>
            <PanelTitle className="text-3xl">
              {complete ? "Your sample plan is ready" : current.title}
            </PanelTitle>
            <PanelDescription className="mt-1" aria-live="polite">
              {current.detail}
            </PanelDescription>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <StatusBadge tone={complete ? "pass" : "pending"}>
              {complete ? "Ready" : "Working"}
            </StatusBadge>
            <StatusBadge tone="precomputed">Sample only. Not live.</StatusBadge>
          </div>
        </div>
      </PanelHeader>
      <PanelBody className="grid gap-5">
        <div className="grid gap-2 sm:grid-cols-4">
          <RunStat
            label="Current action"
            value={complete ? "Waiting for your review" : current.title}
          />
          <RunStat
            label="Time left"
            value={complete ? "Ready" : "About 18 to 32 minutes for a live run"}
          />
          <RunStat label="Spent" value="$0.00" />
          <RunStat
            label="Model"
            value={selectedPortfolio?.display_name ?? "TinyFable Generalist"}
          />
        </div>

        <ol className="grid gap-2" aria-label="Plan progress">
          {JOURNEY.map((step, index) => {
            const done = index < currentPhase || (complete && index === currentPhase);
            const active = index === currentPhase && !complete;
            return (
              <li
                key={step.title}
                className={cn(
                  "grid grid-cols-[auto_1fr] gap-3 rounded-[14px] border border-border px-4 py-3",
                  active && "bg-secondary/35",
                )}
              >
                {done ? (
                  <CheckIcon className="mt-0.5 size-4 text-[var(--pass)]" />
                ) : active ? (
                  <Loader className="mt-0.5" size={16} />
                ) : (
                  <CircleIcon className="mt-0.5 size-4 text-muted-foreground" />
                )}
                <div>
                  <p className="text-sm font-medium">{step.title}</p>
                  <p className="text-sm text-muted-foreground">{step.detail}</p>
                </div>
              </li>
            );
          })}
        </ol>

        {complete ? (
          <OutcomeComparison
            bundle={bundle}
            candidates={comparisonCandidates}
            pending={comparisonPending}
            results={comparisonResults}
          />
        ) : null}

        <div className="flex flex-wrap items-center justify-between gap-3">
          <p className="text-sm text-muted-foreground">
            {lastChecked ? `Last checked at ${lastChecked}.` : "Not checked yet."}
            {" "}Refresh reads the saved project state. It does not start work.
          </p>
          <Button
            type="button"
            variant="outline"
            className="rounded-full"
            onClick={() => void onRefresh()}
            disabled={refreshing}
            data-testid="refresh-project"
          >
            {refreshing ? (
              <Loader size={16} />
            ) : (
              <RefreshCwIcon className="size-4" />
            )}
            {refreshing ? "Checking..." : "Refresh"}
          </Button>
        </div>
      </PanelBody>
    </Panel>
  );
}

function OutcomeComparison({
  bundle,
  candidates,
  pending,
  results,
}: {
  bundle: StageBundle;
  candidates: readonly DemoModelEntry[];
  pending: boolean;
  results: readonly DemoInferenceResponse[];
}) {
  if (pending) {
    return (
      <div
        className="flex items-center gap-2 rounded-[14px] border border-border p-4"
        role="status"
      >
        <Loader size={18} />
        Comparing the saved finance example...
      </div>
    );
  }

  const completed = results.filter(
    (result): result is Extract<DemoInferenceResponse, { status: "ok" }> =>
      result.status === "ok",
  );
  if (completed.length < 2) {
    return (
      <Alert role="alert" data-testid="project-result">
        <AlertDescription>
          The saved comparison could not finish. No live model was called. Open Try it
          to inspect each available output.
        </AlertDescription>
      </Alert>
    );
  }

  const ranked = [...completed].sort((left, right) => {
    const scoreDifference = (right.score ?? -1) - (left.score ?? -1);
    if (scoreDifference !== 0) return scoreDifference;
    return (left.latency_ms ?? Number.MAX_SAFE_INTEGER) -
      (right.latency_ms ?? Number.MAX_SAFE_INTEGER);
  });
  const best = ranked[0]!;
  const bestModel = candidates.find((model) => model.model_id === best.model_id);
  const bestName =
    bestModel?.arm_id === "student_base"
      ? "the current base model"
      : "TinyFable Generalist";
  const equalQuality = completed.every(
    (result) => result.score === completed[0]?.score,
  );
  const equalScore = completed[0]?.score;
  const decision = equalQuality
    ? equalScore === 1
      ? "Take both candidates to the final checks"
      : "Keep both candidates out of rollout for now"
    : `Take ${bestName} to the final checks`;
  const why = equalQuality
    ? equalScore === 1
      ? "Both candidates matched the known answer in this saved example. The full checks need more examples before choosing one."
      : "Neither candidate matched the known answer in this saved example. Fix the result before choosing a model."
    : `${bestName} matched the known answer more closely in this saved example.`;

  return (
    <section
      className="grid gap-4 rounded-[16px] border border-[color-mix(in_oklab,var(--pass)_35%,var(--border))] bg-[color-mix(in_oklab,var(--pass)_6%,var(--card))] p-4"
      data-testid="project-result"
      aria-labelledby="project-decision"
    >
      <div>
        <p className="text-kicker text-[var(--pass)]">Decision</p>
        <h3 id="project-decision" className="mt-1 font-serif text-2xl">
          {decision}
        </h3>
        <p className="mt-2 text-sm">{why}</p>
        <p className="mt-1 text-sm text-muted-foreground">
          Confidence is low. This is one saved finance example, not a live run.
        </p>
      </div>

      <div className="grid gap-3 sm:grid-cols-2">
        {completed.map((result) => {
          const model = candidates.find((item) => item.model_id === result.model_id);
          const generalist = model?.arm_id !== "student_base";
          const outputConfidence =
            typeof result.structured_output.confidence === "number"
              ? `${Math.round(result.structured_output.confidence * 100)}% in the saved output`
              : "Not provided";
          return (
            <article
              key={result.model_id}
              className="rounded-[14px] border border-border bg-card p-3"
            >
              <div className="flex flex-wrap items-center gap-2">
                <h4 className="font-serif text-lg">
                  {generalist ? "TinyFable Generalist" : "Current base model"}
                </h4>
                <StatusBadge tone="precomputed">Saved demo. Not live.</StatusBadge>
              </div>
              <dl className="mt-3 grid gap-2 text-sm">
                <ComparisonFact
                  term="Human outcome"
                  value={plainOutcome(result.structured_output)}
                />
                <ComparisonFact
                  term="Why"
                  value={plainReason(result.structured_output)}
                />
                <ComparisonFact term="Output confidence" value={outputConfidence} />
                <ComparisonFact
                  term="Quality"
                  value={
                    result.score === null
                      ? "Not scored"
                      : result.score === 1
                        ? "Matched the known answer"
                        : "Did not match the known answer"
                  }
                />
                <ComparisonFact
                  term="Speed"
                  value={
                    result.latency_ms === null
                      ? "Not measured"
                      : `${result.latency_ms} ms in the saved demo`
                  }
                />
                <ComparisonFact term="Cost" value="Not measured in a live run" />
              </dl>
            </article>
          );
        })}
      </div>

      <div className="flex flex-wrap items-center justify-between gap-3">
        <p className="text-sm text-muted-foreground">
          No training ran, no money was spent, and no file left this browser.
        </p>
        <Button asChild className="h-11 rounded-full px-5">
          <Link
            href={buildStageHref("/prove", bundle.mode, bundle.run.run_id)}
            data-testid="review-proof-action"
          >
            Review the saved checks
          </Link>
        </Button>
      </div>
    </section>
  );
}

function ComparisonFact({ term, value }: { term: string; value: string }) {
  return (
    <div className="grid grid-cols-[8rem_1fr] gap-2">
      <dt className="text-muted-foreground">{term}</dt>
      <dd>{value}</dd>
    </div>
  );
}

function plainOutcome(output: Record<string, unknown>): string {
  const action = output.policy_action;
  if (typeof action === "string") {
    if (action === "approve") return "Approve the transaction.";
    if (action === "reject") return "Reject the transaction.";
    return "Send the transaction for review.";
  }
  return "Review the structured result.";
}

function plainReason(output: Record<string, unknown>): string {
  const evidence = output.evidence;
  if (Array.isArray(evidence)) {
    if (typeof evidence[0] === "string") {
      return evidence[0];
    }
    const first = evidence[0];
    if (
      first &&
      typeof first === "object" &&
      "field" in first &&
      "value" in first
    ) {
      return `The saved output cites ${String(first.field)} with value ${String(first.value)}.`;
    }
  }
  return "The saved result does not include a plain-language reason.";
}

function RunStat({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-[14px] border border-border bg-card px-3 py-3">
      <p className="text-sm text-muted-foreground">{label}</p>
      <p className="mt-1 font-serif text-lg leading-tight">{value}</p>
    </div>
  );
}

function EvidenceLinks({ bundle }: { bundle: StageBundle }) {
  const links = [
    {
      route: "/curate" as const,
      label: "See the data check",
      detail: "Inspect sources, splits, and fingerprints.",
    },
    {
      route: "/synthesize" as const,
      label: "See how gaps are filled",
      detail: "Inspect provided and generated answers.",
    },
    {
      route: "/train" as const,
      label: "See the run plan",
      detail: "Inspect the method, machine, limits, and logs.",
    },
    {
      route: "/prove" as const,
      label: "See the final checks",
      detail: "Inspect quality, speed, cost, and saved evidence.",
    },
    {
      route: "/demo" as const,
      label: "Try the result",
      detail: "Compare saved outputs or use a live endpoint when one exists.",
    },
  ];
  return (
    <div className="grid gap-3">
      <h3 className="font-serif text-lg">Evidence pages</h3>
      <p className="text-sm text-muted-foreground">
        These pages contain the technical record. You can ignore them for the basic
        flow. Open them when you need to audit a choice or compare a result.
      </p>
      <div className="grid gap-2 md:grid-cols-2">
        {links.map((item) => (
          <Link
            key={item.route}
            href={buildStageHref(item.route, bundle.mode, bundle.run.run_id)}
            className="rounded-[14px] border border-border bg-card p-3 transition-colors hover:border-[var(--orange)]"
          >
            <span className="font-serif text-base">{item.label}</span>
            <span className="mt-1 block text-sm text-muted-foreground">
              {item.detail}
            </span>
          </Link>
        ))}
      </div>
    </div>
  );
}
