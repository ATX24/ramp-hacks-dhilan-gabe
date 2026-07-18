import Link from "next/link";
import Image from "next/image";
import { ArrowUpRight } from "lucide-react";
import { Button } from "./ui/Button";

export function Brand() {
  return (
    <span className="brand-lockup">
      <Image
        src="/anthropic-logo.svg"
        alt="Anthropic"
        width={590}
        height={68}
        priority
      />
      <sup aria-label="2">2</sup>
    </span>
  );
}

export function SiteHeader({
  ctaHref = "/distillery",
  ctaLabel = "Open Distillery",
}: {
  ctaHref?: string;
  ctaLabel?: string;
}) {
  return (
    <header className="site-header">
      <Link className="site-brand" href="/" aria-label="Anthropic 2 home">
        <Brand />
      </Link>
      <nav className="site-nav" aria-label="Primary navigation">
        <Link href="/#research">Research</Link>
        <Link href="/#papers">Papers</Link>
        <Link href="/#about">About</Link>
        <Link href="/distillery">Distillery</Link>
      </nav>
      <Button asChild className="header-cta">
        <Link href={ctaHref}>
          {ctaLabel}
          <ArrowUpRight size={15} />
        </Link>
      </Button>
    </header>
  );
}

export function SiteFooter() {
  return (
    <footer className="site-footer">
      <div className="footer-brand">
        <Brand />
        <p>
          An independent hackathon research lab. Not affiliated with Anthropic.
          The extra 2 is doing a lot of work.
        </p>
      </div>
      <div className="footer-column">
        <span>Research</span>
        <Link href="/#tinyfable">TinyFable</Link>
        <Link href="/#research">Research approach</Link>
        <Link href="/#papers">Papers</Link>
      </div>
      <div className="footer-column">
        <span>Product</span>
        <Link href="/distillery">Distillery</Link>
        <Link href="/distillery#how-it-works">Curate to evaluate</Link>
        <Link href="/distillery#access">Request access</Link>
      </div>
      <div className="footer-bottom">
        <span>Anthropic 2 © 2026</span>
        <span>Built at Ramp Hackathon</span>
      </div>
    </footer>
  );
}
