import type { Metadata } from "next";
import Link from "next/link";
import { ArrowUpRight } from "lucide-react";
import { HeroCarousel } from "./components/HeroCarousel";
import { SiteFooter, SiteHeader } from "./components/SiteChrome";
import { Button } from "./components/ui/Button";

export const metadata: Metadata = {
  title: "Anthropic 2 | TinyFable",
  description:
    "TinyFable is a smaller, portable finance-generalist model trained with Distillery.",
};

const papers = [
  {
    index: "01 / Model report",
    title: "TinyFable: distilling a finance generalist",
    href: "/papers/tinyfable-systems.pdf",
  },
  {
    index: "02 / Systems",
    title: "Distillery: transparent method selection for smaller models",
    href: "/papers/durable-promises.pdf",
  },
  {
    index: "03 / Evaluation",
    title: "Evaluating TinyFable: quality, systems, and economics",
    href: "/papers/evaluation-economics.pdf",
  },
];

export default function Home() {
  return (
    <>
      <SiteHeader />
      <main>
        <div id="tinyfable">
          <HeroCarousel />
        </div>

        <section className="model-feature" id="announcement">
          <div className="feature-meta">
            <span>Announcing TinyFable</span>
            <span>July 2026</span>
          </div>
          <div className="feature-layout">
            <div className="feature-copy">
              <p className="section-kicker">One model, two finance jobs</p>
              <h2>A small model that cleared the bar.</h2>
              <p>
                TinyFable is a 0.5B-class model built to handle transaction
                review and variance analysis with the same set of weights. It
                learns from a deterministic synthetic finance world, then faces
                held-out accounting cases it has not seen before.
              </p>
              <div className="feature-actions">
                <Link href="/writing/announcing-tinyfable.md">
                  Read the announcement <span>↗</span>
                </Link>
                <Link href="/papers/tinyfable-systems.pdf">
                  Read the working paper <span>↗</span>
                </Link>
              </div>
            </div>
            <div className="transfer-diagram" aria-label="A 1.5 billion parameter teacher transfers finance behavior to the 0.5 billion parameter TinyFable model">
              <div className="transfer-node transfer-teacher">
                <span>Source model</span>
                <strong>Qwen2.5</strong>
                <em>1.5B teacher</em>
              </div>
              <div className="transfer-lane">
                <span>validated traces</span>
                <i />
                <b>sequence.v1</b>
                <b>logit.v1</b>
                <i />
                <span>portable weights</span>
              </div>
              <div className="transfer-node transfer-student">
                <span>Model 001</span>
                <strong>TinyFable</strong>
                <em>0.5B student</em>
              </div>
              <div className="transfer-tasks">
                <span>transaction_review</span>
                <span>variance_analysis</span>
              </div>
            </div>
          </div>
        </section>

        <section className="research-section" id="research">
          <div className="section-heading">
            <span>01</span>
            <div>
              <p>Research</p>
              <h2>Training is not a result.</h2>
            </div>
          </div>
          <div className="research-copy-grid">
            <p>
              A smaller model only matters if it keeps enough quality and has a
              credible economic reason to exist. So we start with the decision,
              not the training run.
            </p>
            <p>
              Distillery turns traces and synthetic finance examples into a
              sealed experiment. Its final answer can be approved, failed quality,
              failed economics, insufficient evidence, or do not distill.
            </p>
          </div>

          <div className="simple-method four-up">
            <article>
              <span>1</span>
              <h3>Curate</h3>
              <p>Reuse valid traces. Freeze splits, provenance, and hashes.</p>
            </article>
            <article>
              <span>2</span>
              <h3>Synthesize</h3>
              <p>Generate only missing labels and preserve where each answer came from.</p>
            </article>
            <article>
              <span>3</span>
              <h3>Train</h3>
              <p>Resolve the recipe openly and run one finite, sealed job.</p>
            </article>
            <article>
              <span>4</span>
              <h3>Prove</h3>
              <p>Compare quality, uncertainty, systems behavior, and break-even.</p>
            </article>
          </div>
        </section>

        <section className="benchmark-section">
          <div className="benchmark-intro">
            <p className="section-kicker">TinyFable evaluation</p>
            <h2>The chart arrives with the evidence.</h2>
            <p>
              TinyFable cleared the joint quality and economic release gates.
              Every number in the full report comes from immutable predictions,
              manifests, cost records, and matched two-seed comparisons.
            </p>
          </div>
          <div className="benchmark-placeholder" aria-label="TinyFable evaluation report summary">
            <div className="benchmark-topline">
              <span>SEALED EVALUATION REPORT</span>
              <b>APPROVED</b>
            </div>
            <div className="benchmark-row">
              <span>Task quality</span><i /><em>passed</em>
            </div>
            <div className="benchmark-row">
              <span>OOD and uncertainty</span><i /><em>passed</em>
            </div>
            <div className="benchmark-row">
              <span>Latency and VRAM</span><i /><em>measured</em>
            </div>
            <div className="benchmark-row">
              <span>Break-even</span><i /><em>positive</em>
            </div>
            <p>Release status: APPROVED. Exact values are read from the frozen evaluation artifact.</p>
          </div>
        </section>

        <section className="papers-section" id="papers">
          <div className="section-heading">
            <span>02</span>
            <div>
              <p>Working papers</p>
              <h2>Publish the method before the victory lap.</h2>
            </div>
          </div>
          <div className="paper-list">
            {papers.map((paper) => (
              <a key={paper.index} href={paper.href}>
                <span>{paper.index}</span>
                <h3>{paper.title}</h3>
                <em>PDF ↗</em>
              </a>
            ))}
          </div>
        </section>

        <section className="about-section" id="about">
          <p className="section-kicker">About Anthropic 2</p>
          <h2>
            The world&apos;s second most confusingly named AI research lab.
          </h2>
          <div>
            <p>
              Anthropic 2 is an independent hackathon research lab studying when
              a smaller model is actually worth making. We build compact models,
              publish the methods, and let the evidence constrain the story.
            </p>
            <p>
              This is satire. We are not Anthropic. We do admire the cream, the
              serif, and the confidence required to give a navigation bar this
              much breathing room.
            </p>
          </div>
          <Button asChild className="black-link">
            <Link href="/distillery">
              Open Distillery <ArrowUpRight size={15} />
            </Link>
          </Button>
        </section>
      </main>
      <SiteFooter />
    </>
  );
}
