import type { Metadata } from "next";
import Link from "next/link";
import { SiteFooter, SiteHeader } from "../components/SiteChrome";

export const metadata: Metadata = {
  title: "Distillery | Anthropic 2",
  description:
    "Distillery turns existing traces and synthetic finance examples into a sealed distill-or-don't experiment.",
};

const stages = [
  {
    number: "01",
    title: "Curate",
    copy: "Import valid traces, isolate the test set, and freeze every source and hash.",
  },
  {
    number: "02",
    title: "Synthesize",
    copy: "Ask a teacher only for labels that are missing, rejected, or deliberately augmented.",
  },
  {
    number: "03",
    title: "Train",
    copy: "Resolve sequence or logit distillation openly, then run one finite sealed job.",
  },
  {
    number: "04",
    title: "Prove",
    copy: "Compare the student with rules, its base, its teacher, and a cheap off-the-shelf model.",
  },
];

export default function DistilleryPage() {
  return (
    <>
      <SiteHeader ctaHref="#access" ctaLabel="Request access" />
      <main>
        <section className="distillery-hero">
          <div className="release-meta">
            <span>DISTILLATION PRODUCT</span>
            <span>AVAILABLE NOW</span>
          </div>
          <h1>Distillery</h1>
          <div className="distillery-deck">
            <p>Smaller models. Proven economics.</p>
            <a href="#how-it-works">See how it works ↓</a>
          </div>
        </section>

        <section className="distillery-feature" id="how-it-works">
          <div className="feature-meta">
            <span>The high-level path</span>
            <span>One model, one proof</span>
          </div>
          <div className="distillery-feature-layout">
            <div className="distillery-copy">
              <p className="section-kicker">Decision first</p>
              <h2>Distill, or don&apos;t.</h2>
              <p>
                Distillery turns existing traces and synthetic finance examples
                into TinyFable, then tests whether the smaller model beats the
                practical alternatives on quality and economics. If it does not,
                the product says so.
              </p>
            </div>
            <div className="distill-command">
              <div className="command-title">
                <span>finance_generalist.py</span>
                <span>● complete</span>
              </div>
              <pre>{`distillery = Distillery(api_key=os.environ["DISTILLERY_API_KEY"])
dataset = distillery.datasets.create("./finance_world.jsonl")
run = distillery.distill(dataset, recipe="auto").wait()`}</pre>
              <div className="command-result">
                <span>recipe resolution is always disclosed</span>
                <b>PROVED</b>
              </div>
            </div>
          </div>
        </section>

        <section className="distillery-steps" aria-label="The four Distillery stages">
          {stages.map((stage) => (
            <article key={stage.number}>
              <span>{stage.number}</span>
              <h2>{stage.title}</h2>
              <p>{stage.copy}</p>
            </article>
          ))}
        </section>

        <section className="distillery-proof">
          <div>
            <p className="section-kicker">The first experiment</p>
            <h2>TinyFable is the test, not the testimonial.</h2>
          </div>
          <p>
            One Qwen2.5 1.5B teacher, one 0.5B student, and one finance-generalist
            benchmark. The same TinyFable weights must handle transaction review
            and variance analysis. Rules, base, teacher, a cheap API, oracle SFT,
            sequence KD, and logit KD all get a seat at the table.
          </p>
          <Link href="/#tinyfable">Meet TinyFable ↗</Link>
        </section>

        <section className="outcomes-section">
          <p className="section-kicker">An honest product needs losing states</p>
          <div className="outcome-list">
            <span className="outcome-primary">PROVED</span>
            <span>DO_NOT_DISTILL</span>
            <span>FAILED_QUALITY</span>
            <span>FAILED_ECONOMICS</span>
            <span>INSUFFICIENT_EVIDENCE</span>
          </div>
          <p>
            Distillery does not silently change methods, invent missing evidence,
            or treat a completed training job as success.
          </p>
        </section>

        <section className="access-section" id="access">
          <p>Public research release</p>
          <h2>Bring traces.<br />Keep the proof.</h2>
          <a href="mailto:research@anthropic2.dev?subject=Distillery%20access">
            Request access <span>↗</span>
          </a>
        </section>
      </main>
      <SiteFooter />
    </>
  );
}
