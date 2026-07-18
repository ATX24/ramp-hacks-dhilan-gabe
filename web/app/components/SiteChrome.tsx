import Image from "next/image";
import Link from "next/link";
import { ArrowUpRight } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  NavigationMenu,
  NavigationMenuItem,
  NavigationMenuLink,
  NavigationMenuList,
} from "@/components/ui/navigation-menu";

export function Brand({ inverse = false }: { inverse?: boolean }) {
  return (
    <span className="inline-flex items-start gap-1">
      <Image
        src="/anthropic-logo.svg"
        alt="Anthropic"
        width={590}
        height={68}
        priority
        className={inverse ? "h-auto w-[128px] brightness-0 invert" : "h-auto w-[128px]"}
      />
      <sup
        aria-label="2"
        className="-mt-1.5 grid size-4 place-items-center bg-[#d65f45] font-mono text-[9px] font-bold leading-none text-[#faf9f5]"
      >
        2
      </sup>
    </span>
  );
}

const links = [
  ["Research", "/research"],
  ["Papers", "/papers"],
  ["Docs", "/docs"],
  ["Experiment", "/experiment"],
  ["TinyFable", "/tinyfable"],
] as const;

export function SiteHeader({
  ctaHref = "/distillery",
  ctaLabel = "Try Distillery",
}: {
  ctaHref?: string;
  ctaLabel?: string;
}) {
  return (
    <header className="mx-auto flex h-24 w-full max-w-[1600px] items-center justify-between px-6 md:px-10 lg:px-14">
      <Link href="/" aria-label="Anthropic 2 home">
        <Brand />
      </Link>

      <div className="flex items-center gap-8">
        <NavigationMenu className="hidden lg:flex" viewport={false}>
          <NavigationMenuList className="gap-1">
            {links.map(([label, href]) => (
              <NavigationMenuItem key={href}>
                <NavigationMenuLink asChild className="bg-transparent px-3 text-[15px] font-normal hover:bg-black/5">
                  <Link href={href}>{label}</Link>
                </NavigationMenuLink>
              </NavigationMenuItem>
            ))}
          </NavigationMenuList>
        </NavigationMenu>
        <Button asChild size="lg" className="h-11 rounded-xl bg-[#141413] px-4 !text-[#faf9f5] hover:bg-[#2b2b28] hover:!text-white">
          <Link href={ctaHref}>
            {ctaLabel}
            <ArrowUpRight className="size-4" />
          </Link>
        </Button>
      </div>
    </header>
  );
}

export function SiteFooter() {
  return (
    <footer className="mt-28 bg-[#141413] text-[#faf9f5]">
      <div className="mx-auto grid max-w-[1600px] gap-14 px-6 py-16 md:grid-cols-[1.5fr_1fr_1fr] md:px-10 lg:px-14">
        <div>
          <Brand inverse />
          <p className="mt-7 max-w-sm text-sm leading-6 text-white/60">
            An independent research lab studying when smaller models are worth making. Not affiliated with Anthropic.
          </p>
        </div>
        <div className="grid content-start gap-3 text-sm">
          <span className="mb-2 text-white/45">Research</span>
          <Link href="/tinyfable">TinyFable</Link>
          <Link href="/research">Research areas</Link>
          <Link href="/papers">Publications</Link>
        </div>
        <div className="grid content-start gap-3 text-sm">
          <span className="mb-2 text-white/45">Product</span>
          <Link href="/distillery">Distillery</Link>
          <Link href="/company">Company</Link>
          <a href="mailto:research@anthropic2.dev">Contact</a>
        </div>
      </div>
      <div className="mx-auto flex max-w-[1600px] justify-between border-t border-white/15 px-6 py-6 text-xs text-white/45 md:px-10 lg:px-14">
        <span>Anthropic 2 © 2026</span>
        <span>Built at Ramp Hackathon</span>
      </div>
    </footer>
  );
}
