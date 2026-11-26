import type { Metadata } from "next";
import {
  Poppins,
  Instrument_Serif,
  Lato,
  Limelight,
  Fascinate,
} from "next/font/google";
import "./globals.css";

const poppins = Poppins({
  subsets: ["latin"],
  weight: ["100", "200", "300", "400", "500", "600", "700", "800", "900"],
  style: ["normal", "italic"],
  variable: "--font-poppins",
  display: "swap",
});

const instrumentSerif = Instrument_Serif({
  subsets: ["latin"],
  weight: ["400"],
  style: ["normal", "italic"],
  variable: "--font-instrument",
  display: "swap",
});

const lato = Lato({
  subsets: ["latin"],
  weight: ["100", "300", "400", "700", "900"],
  style: ["normal", "italic"],
  variable: "--font-lato",
  display: "swap",
});

const limelight = Limelight({
  subsets: ["latin"],
  weight: ["400"],
  variable: "--font-limelight",
  display: "swap",
});

const fascinate = Fascinate({
  subsets: ["latin"],
  weight: ["400"],
  variable: "--font-fascinate",
  display: "swap",
});

export const metadata: Metadata = {
  title: "PsycheAI — Multimodal Depression Detection",
  description:
    "PsycheAI: Advanced Multimodal Depression Detection powered by deep learning and cinematic visualization.",
  keywords: ["PsycheAI", "depression detection", "multimodal AI", "mental health AI"],
  openGraph: {
    title: "PsycheAI — Multimodal Depression Detection",
    description: "Advanced Multimodal Depression Detection",
    type: "website",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html
      lang="en"
      className={`${poppins.variable} ${instrumentSerif.variable} ${lato.variable} ${limelight.variable} ${fascinate.variable}`}
    >
      <body className="antialiased" suppressHydrationWarning>
        {children}
      </body>
    </html>
  );
}
