"use client";

import { useEffect, useRef } from "react";
import Lenis from "lenis";
import gsap from "gsap";
import { ScrollTrigger } from "gsap/ScrollTrigger";

/**
 * LenisProvider
 *
 * Initialises Lenis smooth scroll and wires it into GSAP ScrollTrigger
 * via RAF ticker. Renders its children as-is — it is purely a behavioural
 * wrapper with no DOM output.
 */
export default function LenisProvider({
  children,
}: {
  children: React.ReactNode;
}) {
  const lenisRef = useRef<Lenis | null>(null);

  useEffect(() => {
    gsap.registerPlugin(ScrollTrigger);

    const lenis = new Lenis({
      duration: 1.15,
      easing: (t) => Math.min(1, 1.001 - Math.pow(2, -10 * t)), // expo
      touchMultiplier: 1.8,
      infinite: false,
      syncTouch: false,
    });

    lenisRef.current = lenis;

    // Bridge Lenis → ScrollTrigger
    lenis.on("scroll", ScrollTrigger.update);

    // Add Lenis RAF to GSAP ticker for 60fps+ frame lock
    const tickerCallback = (time: number) => {
      lenis.raf(time * 1000);
    };
    gsap.ticker.add(tickerCallback);
    gsap.ticker.lagSmoothing(500, 33);

    return () => {
      lenis.destroy();
      lenisRef.current = null;
      gsap.ticker.remove(tickerCallback);
    };
  }, []);

  return <>{children}</>;
}
