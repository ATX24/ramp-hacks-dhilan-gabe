import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  metadataBase: new URL("https://anthropic2.dev"),
  title: "Anthropic 2 | TinyFable",
  description:
    "Anthropic 2 is an independent research lab studying when smaller, portable models are worth making.",
  openGraph: {
    title: "Anthropic 2 | TinyFable",
    description:
      "TinyFable is one smaller, portable finance generalist. The model, method, and proof publish together.",
    url: "https://anthropic2.dev",
    siteName: "Anthropic 2",
    type: "website",
    images: [
      {
        url: "/og.png",
        width: 1731,
        height: 909,
        alt: "Anthropic 2 presents TinyFable. Smaller models. Proven economics.",
      },
    ],
  },
  twitter: {
    card: "summary_large_image",
    title: "Anthropic 2 | TinyFable",
    description:
      "TinyFable is one smaller, portable finance generalist. The model, method, and proof publish together.",
    images: ["/og.png"],
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
