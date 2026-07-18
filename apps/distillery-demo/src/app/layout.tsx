import type { Metadata } from "next";
import { Geist_Mono, Source_Sans_3 } from "next/font/google";
import { TooltipProvider } from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";
import "./globals.css";

const sourceSans = Source_Sans_3({
  subsets: ["latin"],
  variable: "--font-sans",
  display: "swap",
});

const geistMono = Geist_Mono({
  subsets: ["latin"],
  variable: "--font-mono",
  display: "swap",
});

export const metadata: Metadata = {
  title: "Distillery · TinyFable",
  description:
    "Smaller models. Proven economics. Curate → Synthesize → Train → Prove → Demo.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html
      lang="en"
      className={cn(sourceSans.variable, geistMono.variable, "font-sans")}
      style={{ ["--font-serif" as string]: 'Georgia, "Times New Roman", serif' }}
    >
      <body>
        <TooltipProvider delayDuration={200}>{children}</TooltipProvider>
      </body>
    </html>
  );
}
