import type { Metadata } from "next";
import Link from "next/link";
import { SiteFooter, SiteHeader } from "../components/SiteChrome";
import { CompareDemo } from "./CompareDemo";

export const metadata: Metadata = {
  title: "The Distillation Experiment | Anthropic 2",
  description:
    "Distilling a large model into a small one for financial workflows with Distillery, including a side-by-side demo of the student before and after distillation.",
};

export default function ExperimentPage() {
  return (
    <>
      <SiteHeader ctaHref="/distillery" ctaLabel="Open Distillery" />
      <main>
        <section className="distillery-hero">
          <div className="release-meta">
            <span>EXPERIMENT WRITEUP</span>
            <span>RAMP HACKATHON 2026</span>
          </div>
          <h1>Distilling large to small, and proving it</h1>
          <div className="distillery-deck">
            <p>Why distillation is the open-source path, and what happened when we ran it.</p>
            <a href="#demo">Jump to the demo ↓</a>
          </div>
        </section>

        <section className="distillery-feature" id="writeup">
          <div className="feature-meta">
            <span>The thesis</span>
            <span>Distillation as strategy</span>
          </div>
          <div className="prose-block">
            <p>
              Realistically, the only way open source AI companies will be able to
              build models to compete with frontier labs is via distillation.
            </p>
            <p>
              Thinking Machines built Tinker, which made fine-tuning incredibly easy
              for researchers. We wanted to take their API a step further and
              specifically build for distillation. Distillery lets you
              programmatically describe your distillation process: which models to
              use, data formation (we provide a synthetic option as well), and
              actual hardware for training. We also let users define specific
              configs for new and experimental distillation techniques.
            </p>
            <p>
              To prove the capabilities of this API, we ran an experiment where we
              attempted to distill a large Nova teacher into a much smaller
              student. With Distillery, we synthesized a dataset of 400 financial
              traces and trained the student on the teacher&apos;s outputs. We
              achieved a working large-to-small distillation loop end to end:
              teacher labeling with full provenance, two competing training runs
              (a standard recipe and a custom rejection-sampling recipe), and a
              sealed held-out evaluation where the teacher itself scores 57.5%
              schema validity and 25% decision-field accuracy, numbers the student
              has to be judged against honestly.
            </p>
          </div>
        </section>

        <section className="distillery-feature" id="what-is-distillery">
          <div className="feature-meta">
            <span>The system</span>
            <span>Curate, synthesize, train, prove</span>
          </div>
          <div className="prose-block">
            <p className="section-kicker">What Distillery is</p>
            <p>
              Distillery is a distillation platform: it takes expensive repetitive
              calls to a large model and turns them into a cheap small model you
              can actually trust. The workflow is Curate, Synthesize, Train,
              Prove, exposed as an API with a three-line SDK happy path.
            </p>
            <p>
              Trace synthesis covers the first two stages. Datasets are immutable,
              hashed resources with strict train, validation, and test splits.
              Labels come from a teacher model filling only missing training
              responses; the teacher structurally never sees test prompts. Every
              generated label carries provenance and must pass deterministic
              schema validation before acceptance. Any relabeling or curation
              produces a new immutable dataset, never a mutation.
            </p>
            <p>
              Recipes are the distillation methods. Built-ins include
              sequence-level knowledge distillation for black-box API teachers and
              full-logit distillation for white-box pairs, which fails loudly with
              RECIPE_INCOMPATIBLE when its requirements are not met. There are no
              silent downgrades, and an auto resolver explains every choice it
              makes, including recommending do-not-distill. Users can also define
              their own recipes: a distillation method written as a small Python
              class against synthesis primitives, registered and run through the
              same resolver, capability gates, and API as the built-ins.
            </p>
            <p>
              Proof is the point of the whole thing. Nothing ships on vibes. Every
              candidate student is evaluated on frozen held-out data the training
              process never touched: schema validity, decision-field accuracy,
              cost ratio versus the teacher, total experiment cost, and break-even
              request count. A candidate below target gets BLOCKED, which is a
              first-class outcome, not a failure. Only a passing model can be
              promoted behind a stable model alias, with automatic teacher
              fallback on invalid outputs and one-call rollback. Recipes,
              including user-defined ones, cannot touch or override this gate;
              that separation is what makes letting anyone invent a distillation
              method safe.
            </p>
            <p className="section-kicker">The experiment</p>
            <p>
              The task is structured financial workflows: transaction review (GL
              account, balanced journal entry, policy action), variance analysis,
              and cash reconciliation, drawn from a deterministic synthetic
              finance world with an executable oracle that knows every
              ground-truth answer. The teacher is Amazon Nova Pro; the student is
              a 0.5B parameter model, roughly a thousand-fold parameter
              reduction, trained with QLoRA.
            </p>
            <p>
              We labeled the training traces with the teacher in about two minutes
              of parallel calls for under a dollar. 113 labels passed schema
              validation. From the same labels we built two datasets: the
              baseline keeps all 113 valid labels, more data but noisier; the
              custom rejection-sampling recipe keeps only the 40 labels whose
              decision fields also match the oracle, three times less data with
              near-zero label noise. Both students train through the identical
              sealed pipeline, and both are scored on the same frozen 40-example
              test set as the teacher. That is the comparison the proof gate
              exists to referee: clean-but-few versus noisy-but-many is a real
              methodological tradeoff, and Distillery measures the winner instead
              of asserting it.
            </p>
          </div>
        </section>

        <CompareDemo />

        <section className="distillery-proof">
          <div>
            <p className="section-kicker">Run it yourself</p>
            <h2>The demo is live code, not a screenshot.</h2>
          </div>
          <p>
            The comparison above mirrors examples/compare_demo.py in the repo: one
            0.5B model held in memory, the LoRA adapter toggled off for the base
            output and on for the distilled output, every response validated
            against the executable oracle. Clone the repo and run it locally to
            drive any of the 40 held-out tasks through both models.
          </p>
          <Link href="/distillery">Read about Distillery ↗</Link>
        </section>
      </main>
      <SiteFooter />
    </>
  );
}
