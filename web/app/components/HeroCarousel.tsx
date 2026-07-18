"use client";

import { useCallback, useEffect, useState } from "react";
import useEmblaCarousel from "embla-carousel-react";
import { ArrowDown, ArrowUpRight, ChevronLeft, ChevronRight } from "lucide-react";
import Link from "next/link";
import { SdkSnippet } from "./SdkSnippet";
import { Button } from "./ui/Button";

const slides = ["TinyFable", "Distillery"];

export function HeroCarousel() {
  const [emblaRef, emblaApi] = useEmblaCarousel({ loop: true });
  const [selectedIndex, setSelectedIndex] = useState(0);

  const updateSelection = useCallback(() => {
    if (emblaApi) setSelectedIndex(emblaApi.selectedScrollSnap());
  }, [emblaApi]);

  useEffect(() => {
    if (!emblaApi) return;
    emblaApi.on("select", updateSelection);
    emblaApi.on("reInit", updateSelection);
    return () => {
      emblaApi.off("select", updateSelection);
      emblaApi.off("reInit", updateSelection);
    };
  }, [emblaApi, updateSelection]);

  return (
    <section className="hero-carousel" aria-roledescription="carousel" aria-label="Anthropic 2 releases">
      <div className="hero-viewport" ref={emblaRef}>
        <div className="hero-track">
          <article className="hero-slide hero-tinyfable" aria-label="TinyFable, slide 1 of 2">
            <div className="hero-topline">
              <span>MODEL 001</span>
              <span>RELEASED JULY 2026</span>
            </div>
            <div className="hero-title-wrap">
              <p>Trained with Distillery</p>
              <h1>TinyFable</h1>
            </div>
            <div className="hero-lower">
              <div className="hero-deck">
                <p>
                  A portable finance generalist distilled from a 1.5B teacher
                  into a 0.5B student.
                </p>
                <Button asChild variant="outline">
                  <a href="#announcement">
                    Read the model report <ArrowDown size={15} />
                  </a>
                </Button>
              </div>
              <SdkSnippet compact />
            </div>
          </article>

          <article className="hero-slide hero-distillery" aria-label="Distillery, slide 2 of 2">
            <div className="hero-topline">
              <span>DISTILLATION PRODUCT</span>
              <span>AVAILABLE NOW</span>
            </div>
            <div className="hero-title-wrap">
              <p>From traces to a model in three lines</p>
              <h2>Distillery</h2>
            </div>
            <div className="hero-lower">
              <div className="hero-deck">
                <p>Curate the evidence. Train the candidate. Evaluate whether it was worth it.</p>
                <Button asChild>
                  <Link href="/distillery">
                    Open Distillery <ArrowUpRight size={15} />
                  </Link>
                </Button>
              </div>
              <SdkSnippet compact />
            </div>
          </article>
        </div>
      </div>

      <div className="hero-controls">
        <div className="hero-dots" aria-label="Choose a release">
          {slides.map((slide, index) => (
            <button
              key={slide}
              type="button"
              className={index === selectedIndex ? "hero-dot is-active" : "hero-dot"}
              onClick={() => emblaApi?.scrollTo(index)}
              aria-label={`Show ${slide}`}
              aria-current={index === selectedIndex ? "true" : undefined}
            >
              <span>{String(index + 1).padStart(2, "0")}</span>
              {slide}
            </button>
          ))}
        </div>
        <div className="hero-arrows">
          <Button
            size="icon"
            variant="outline"
            aria-label="Previous release"
            onClick={() => emblaApi?.scrollPrev()}
          >
            <ChevronLeft size={18} />
          </Button>
          <Button
            size="icon"
            variant="outline"
            aria-label="Next release"
            onClick={() => emblaApi?.scrollNext()}
          >
            <ChevronRight size={18} />
          </Button>
        </div>
      </div>
    </section>
  );
}
