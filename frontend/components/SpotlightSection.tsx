"use client";

import { useLayoutEffect, useRef } from "react";
import Link from "next/link";
import gsap from "gsap";
import { ScrollTrigger } from "gsap/ScrollTrigger";
import { SplitText } from "gsap/SplitText";

/* ─────────────────────────────────────────────────────────────
   VIDEO CATALOGUE & SETTINGS
──────────────────────────────────────────────────────────────── */
const GRID_VIDEOS = [
  "/videos/barcelona.mp4",
  "/videos/berlin-80s.mp4",
  "/videos/france.mp4",
  "/videos/greece.mp4",
  "/videos/iphone15.mp4",
  "/videos/istanbul.mp4",
  "/videos/italy-venice.mp4",
  "/videos/italy-bw.mp4",
  "/videos/london-xt50.mp4",
  "/videos/london-barbican.mp4",
  "/videos/london-summer.mp4",
  "/videos/london-vhs.mp4",
  "/videos/newyork.mp4",
  "/videos/prague.mp4",
  "/videos/berlin-fx30.mp4",
  "/videos/krakow.mp4",
  "/videos/berlin-teufelsberg.mp4",
  "/videos/berlin-summer.mp4",
  "/videos/endless-love.mp4",
  "/videos/memories.mp4",
];

/** The ONLY colored video — emerges from center & stays as continuous background */
const COVER_VIDEO = "/videos/beauty-fx30.mp4";

// 18 Large B&W Video Configurations strictly in the Perimeter (Center Void: [-26vw, +26vw] x [-26vh, +26vh])
const SELECTED_VIDEOS = [
  // ── BATCH 1 (6 Videos, reveals 0.05 -> 0.18) ──
  { src: GRID_VIDEOS[0],  x: -35, y: -28, rotateFrom: -6, rotateTo: -3, baseScale: 1.05, z: 2, batch: 1 }, // Top-Left
  { src: GRID_VIDEOS[1],  x: 35,  y: -28, rotateFrom: 6,  rotateTo: 3,  baseScale: 1.00, z: 2, batch: 1 }, // Top-Right
  { src: GRID_VIDEOS[2],  x: -35, y: 28,  rotateFrom: 5,  rotateTo: 2,  baseScale: 0.95, z: 2, batch: 1 }, // Bottom-Left
  { src: GRID_VIDEOS[3],  x: 35,  y: 28,  rotateFrom: -5, rotateTo: -2, baseScale: 1.02, z: 2, batch: 1 }, // Bottom-Right
  { src: GRID_VIDEOS[4],  x: -38, y: 0,   rotateFrom: -4, rotateTo: -1, baseScale: 1.00, z: 2, batch: 1 }, // Mid-Left
  { src: GRID_VIDEOS[5],  x: 38,  y: 0,   rotateFrom: 4,  rotateTo: 1,  baseScale: 1.00, z: 2, batch: 1 }, // Mid-Right

  // ── BATCH 2 (6 Videos, reveals 0.16 -> 0.29) ──
  { src: GRID_VIDEOS[6],  x: -16, y: -34, rotateFrom: -5, rotateTo: -2, baseScale: 0.98, z: 3, batch: 2 }, // Top-Center-Left
  { src: GRID_VIDEOS[7],  x: 16,  y: -34, rotateFrom: 5,  rotateTo: 2,  baseScale: 0.96, z: 3, batch: 2 }, // Top-Center-Right
  { src: GRID_VIDEOS[8],  x: -16, y: 34,  rotateFrom: 4,  rotateTo: 1,  baseScale: 0.95, z: 3, batch: 2 }, // Bottom-Center-Left
  { src: GRID_VIDEOS[9],  x: 16,  y: 34,  rotateFrom: -4, rotateTo: -1, baseScale: 1.02, z: 3, batch: 2 }, // Bottom-Center-Right
  { src: GRID_VIDEOS[10], x: -40, y: -15, rotateFrom: 7,  rotateTo: 4,  baseScale: 1.04, z: 3, batch: 2 }, // Left-Upper
  { src: GRID_VIDEOS[11], x: 40,  y: -15, rotateFrom: -7, rotateTo: -4, baseScale: 0.98, z: 3, batch: 2 }, // Right-Upper

  // ── BATCH 3 (6 Videos, reveals 0.18 -> 0.28) ──
  { src: GRID_VIDEOS[12], x: -40, y: 15,  rotateFrom: -6, rotateTo: -3, baseScale: 0.96, z: 4, batch: 3 }, // Left-Lower
  { src: GRID_VIDEOS[13], x: 40,  y: 15,  rotateFrom: 6,  rotateTo: 3,  baseScale: 1.02, z: 4, batch: 3 }, // Right-Lower
  { src: GRID_VIDEOS[14], x: 0,   y: -38, rotateFrom: 2,  rotateTo: 0,  baseScale: 0.95, z: 4, batch: 3 }, // Top-Edge Center
  { src: GRID_VIDEOS[15], x: 0,   y: 38,  rotateFrom: -2, rotateTo: 0,  baseScale: 0.95, z: 4, batch: 3 }, // Bottom-Edge Center
  { src: GRID_VIDEOS[16], x: -44, y: -30, rotateFrom: -8, rotateTo: -5, baseScale: 1.00, z: 4, batch: 3 }, // Far Top-Left
  { src: GRID_VIDEOS[17], x: 44,  y: -30, rotateFrom: 8,  rotateTo: 5,  baseScale: 1.00, z: 4, batch: 3 }, // Far Top-Right
];

const BATCH_TIMINGS = {
  1: { start: 0.00, end: 0.12 },
  2: { start: 0.10, end: 0.20 },
  3: { start: 0.18, end: 0.28 },
};

/* helper — lerp */
const lerp = (a: number, b: number, t: number) => a + (b - a) * t;

/* helper — clamp progress between 0 and 1 */
const clamp01 = (v: number) => Math.max(0, Math.min(1, v));

/* helper — ease out cubic */
const easeOutCubic = (t: number) => 1 - Math.pow(1 - t, 3);

/* Conversation messages for the chat demo */
const CHAT_MESSAGES = [
  { author: "ai",   text: "How have you been feeling lately?",                                            revealAt: 0.880 },
  { author: "user", text: "I've been feeling more tired than usual, and I've had less motivation to do things.", revealAt: 0.895 },
  { author: "ai",   text: "How often have you been feeling this way?",                                    revealAt: 0.908 },
  { author: "user", text: "Probably most days over the last couple of weeks.",                            revealAt: 0.920 },
  { author: "ai",   text: "Has it been affecting your sleep, work, studies, or everyday activities?",    revealAt: 0.932 },
  { author: "user", text: "Yeah, I've been finding it harder to focus on my studies.",                    revealAt: 0.944 },
  { author: "ai",   text: "Thanks for sharing that. I'll keep that in mind as we continue.",             revealAt: 0.956 },
];

/* Step labels */
const STEP_LABELS = [
  { num: "01", title: "You Talk",        desc: "Share what's been on your mind through text or voice.", revealAt: 0.962 },
  { num: "02", title: "AI Adapts",       desc: "Questions evolve based on what you tell us.",           revealAt: 0.972 },
  { num: "03", title: "You Understand",  desc: "Receive a clear, plain-language wellbeing snapshot.",   revealAt: 0.982 },
];

/* message reveal window — how much progress it takes each message to fully fade in */
const MSG_FADE_RANGE = 0.010;
/* step label reveal window */
const STEP_FADE_RANGE = 0.008;

export default function SpotlightSection() {
  const sectionRef        = useRef<HTMLElement>(null);
  const titleHeaderRef    = useRef<HTMLHeadingElement>(null);
  const titleSubheaderRef = useRef<HTMLParagraphElement>(null);
  const titleContainerRef = useRef<HTMLDivElement>(null);
  const introHeaderRef    = useRef<HTMLHeadingElement>(null);

  /** The SINGLE main cover video */
  const coverWrapRef    = useRef<HTMLDivElement>(null);
  const coverVideoElRef = useRef<HTMLVideoElement>(null);
  const floatingVideoRefs = useRef<(HTMLDivElement | null)[]>([]);

  /** About Us editorial container ref */
  const aboutSectionRef = useRef<HTMLDivElement>(null);

  /** How It Works refs */
  const howItWorksRef        = useRef<HTMLDivElement>(null);
  const howHeaderRef         = useRef<HTMLDivElement>(null);
  const chatContainerRef     = useRef<HTMLDivElement>(null);
  const chatMessageRefs      = useRef<(HTMLDivElement | null)[]>([]);
  const stepLabelRefs        = useRef<(HTMLDivElement | null)[]>([]);
  const howOverlayRef        = useRef<HTMLDivElement>(null);

  /** Product Dashboard refs */
  const dashboardRef         = useRef<HTMLDivElement>(null);
  const dashboardShellRef    = useRef<HTMLDivElement>(null);
  const dashboardNavRef      = useRef<HTMLDivElement>(null);
  const dashboardHeadRef     = useRef<HTMLDivElement>(null);
  const dashboardCard1Ref    = useRef<HTMLDivElement>(null);
  const dashboardCard2Ref    = useRef<HTMLDivElement>(null);
  const dashboardCard3Ref    = useRef<HTMLDivElement>(null);
  const dashboardInsightsRef = useRef<HTMLDivElement>(null);
  const dashboardPersonalRef = useRef<HTMLDivElement>(null);
  const dashboardIntroRef    = useRef<HTMLDivElement>(null);
  const centerCtaRef         = useRef<HTMLDivElement>(null);

  const introSplitRef = useRef<InstanceType<typeof SplitText> | null>(null);
  const stRef         = useRef<ScrollTrigger | null>(null);

  useLayoutEffect(() => {
    gsap.registerPlugin(ScrollTrigger, SplitText);

    function initAnimations() {
      /* ── Teardown ──────────────────────────────────────────── */
      introSplitRef.current?.revert();    introSplitRef.current = null;
      stRef.current?.kill();              stRef.current = null;

      const introEl = introHeaderRef.current;
      if (!introEl) return;

      /* ── SplitText on Intro ────────────────────────────────── */
      introSplitRef.current = SplitText.create(introEl, { type: "words" });
      gsap.set(introSplitRef.current.words, { opacity: 1 });

      /* ── Initial states setup ── */
      gsap.set(titleContainerRef.current, { opacity: 1 });
      if (titleHeaderRef.current)    gsap.set(titleHeaderRef.current, { opacity: 0, y: 20 });
      if (titleSubheaderRef.current) gsap.set(titleSubheaderRef.current, { opacity: 0, y: 15 });
      if (aboutSectionRef.current) {
        gsap.set(aboutSectionRef.current, { opacity: 0, y: 30, pointerEvents: "none" });
      }

      /* ── How It Works initial states ── */
      if (howItWorksRef.current)    gsap.set(howItWorksRef.current, { opacity: 0, pointerEvents: "none" });
      if (howHeaderRef.current)     gsap.set(howHeaderRef.current, { opacity: 0, y: 24 });
      if (chatContainerRef.current) gsap.set(chatContainerRef.current, { opacity: 0, scale: 0.96, y: 20 });
      if (howOverlayRef.current)    gsap.set(howOverlayRef.current, { opacity: 0 });

      chatMessageRefs.current.forEach((el) => {
        if (el) gsap.set(el, { opacity: 0, y: 10 });
      });
      stepLabelRefs.current.forEach((el) => {
        if (el) gsap.set(el, { opacity: 0, y: 8 });
      });



      /* ── Dashboard initial states ── */
      if (dashboardRef.current)      gsap.set(dashboardRef.current,      { opacity: 0, pointerEvents: "none" });
      if (dashboardIntroRef.current) gsap.set(dashboardIntroRef.current, { opacity: 0, y: 20 });
      if (dashboardShellRef.current) gsap.set(dashboardShellRef.current, { opacity: 0, scale: 0.96, y: 24 });
      if (dashboardNavRef.current)   gsap.set(dashboardNavRef.current,   { opacity: 0, y: 8 });
      if (dashboardHeadRef.current)  gsap.set(dashboardHeadRef.current,  { opacity: 0, y: 12 });
      if (dashboardCard1Ref.current) gsap.set(dashboardCard1Ref.current, { opacity: 0, y: 16 });
      if (dashboardCard2Ref.current) gsap.set(dashboardCard2Ref.current, { opacity: 0, y: 16 });
      if (dashboardCard3Ref.current) gsap.set(dashboardCard3Ref.current, { opacity: 0, y: 16 });
      if (dashboardInsightsRef.current) gsap.set(dashboardInsightsRef.current, { opacity: 0, y: 20 });
      if (dashboardPersonalRef.current) gsap.set(dashboardPersonalRef.current, { opacity: 0, y: 20 });
      if (centerCtaRef.current)         gsap.set(centerCtaRef.current,         { opacity: 0, scale: 0.9, y: 30, pointerEvents: "none" });

      // Main cover video starts tiny at exact center
      gsap.set(coverWrapRef.current, {
        scale: 0.005,
        opacity: 0,
        borderRadius: 20,
        xPercent: -50,
        yPercent: -50,
        left: "50%",
        top: "50%",
        position: "absolute",
      });
      if (coverVideoElRef.current) {
        gsap.set(coverVideoElRef.current, { filter: "none" });
      }

      // Floating B&W videos start deep in z-space strictly along the perimeter
      SELECTED_VIDEOS.forEach((video, idx) => {
        const el = floatingVideoRefs.current[idx];
        if (!el) return;
        gsap.set(el, {
          x: `${video.x}vw`,
          y: `${video.y}vh`,
          scale: 0.15 * video.baseScale,
          opacity: 0,
          filter: "grayscale(100%) contrast(1.05) blur(18px)",
          rotation: video.rotateFrom,
          xPercent: -50,
          yPercent: -50,
          left: "50%",
          top: "50%",
          zIndex: video.z,
          force3D: true,
        });
      });

      /* ── ScrollTrigger ── */
      stRef.current = ScrollTrigger.create({
        trigger: sectionRef.current,
        start:   "top top",
        end:     "+=2600%",   // Extended to 2600% to accommodate Dashboard section
        pin:     true,
        scrub:   0.8,         // Snappier, ultra-smooth response with Lenis
        fastScrollEnd: true,
        onUpdate: (self) => {
          const globalP = self.progress; // 0 → 1
          const p = globalP * (2600 / 1100); // scaled progress for all animations

          /* ─────────────────────────────────────────────────────
             1. B&W VIDEOS PERIMETER DEPTH ENTRANCE (0.00 → 0.35)
             Leaves central void [-26vw, +26vw] x [-26vh, +26vh] clear!
          ───────────────────────────────────────────────────── */
          SELECTED_VIDEOS.forEach((video, idx) => {
            const el = floatingVideoRefs.current[idx];
            if (!el) return;

            const timing = BATCH_TIMINGS[video.batch as 1 | 2 | 3];
            const stagger = (idx % 6) * 0.012;
            const revealStart = timing.start + stagger;
            const revealDuration = 0.10;

            let opacity = 0;
            let scale = 0.15 * video.baseScale;
            let blur = 18;
            let curX = video.x;
            let curY = video.y;
            let curRot = video.rotateFrom;

            if (p >= revealStart) {
              const t_reveal = Math.min(1, (p - revealStart) / revealDuration);
              const easeOut = 1 - Math.pow(1 - t_reveal, 3);

              opacity = easeOut;
              scale = video.baseScale * (0.15 + 0.85 * easeOut);
              blur = 18 * (1 - easeOut);
              curX = video.x;
              curY = video.y;
              curRot = lerp(video.rotateFrom, video.rotateTo, easeOut);
            }

            // Recess into background when main video emerges (p >= 0.40)
            if (p >= 0.40) {
              const t_bg = Math.min(1, (p - 0.40) / 0.12);
              opacity = lerp(1, 0.10, t_bg);
              scale = scale * lerp(1, 0.85, t_bg);
              blur = blur + t_bg * 4;
            }

            gsap.set(el, {
              x: `${curX}vw`,
              y: `${curY}vh`,
              scale: scale,
              opacity: opacity,
              filter: `grayscale(100%) contrast(1.05)${blur > 0 ? ` blur(${blur.toFixed(1)}px)` : ""}`,
              rotation: curRot,
            });
          });

          /* ─────────────────────────────────────────────────────
             2. INTRO TEXT ("A silent struggle...") (0.00 → 0.36)
             Inside the void, fades out completely before main video
          ───────────────────────────────────────────────────── */
          const introSplit = introSplitRef.current;
          if (introSplit && introSplit.words.length > 0) {
            if (p >= 0.26 && p <= 0.36) {
              const introFadeProgress = (p - 0.26) / 0.10;
              const totalWords = introSplit.words.length;

              introSplit.words.forEach((word, index) => {
                const wordFadeProgress = index / totalWords;
                const fadeRange = 0.12;

                if (introFadeProgress >= wordFadeProgress + fadeRange) {
                  gsap.set(word, { opacity: 0 });
                } else if (introFadeProgress <= wordFadeProgress) {
                  gsap.set(word, { opacity: 1 });
                } else {
                  const wordOpacity =
                    1 - (introFadeProgress - wordFadeProgress) / fadeRange;
                  gsap.set(word, { opacity: wordOpacity });
                }
              });
            } else if (p < 0.26) {
              gsap.set(introSplit.words, { opacity: 1 });
            } else if (p > 0.36) {
              gsap.set(introSplit.words, { opacity: 0 });
            }
          }

          /* ─────────────────────────────────────────────────────
             3. MAIN COVER VIDEO EMERGENCE FROM VOID (0.40 → 1.00)
             Starts AFTER intro text has completely faded out
          ───────────────────────────────────────────────────── */
          if (coverWrapRef.current) {
            if (p < 0.40) {
              gsap.set(coverWrapRef.current, {
                scale: 0.005,
                opacity: 0,
                borderRadius: 20,
              });
              if (coverVideoElRef.current) {
                gsap.set(coverVideoElRef.current, { filter: "none" });
              }
            } else if (p >= 0.40 && p <= 0.53) {
              const tm = (p - 0.40) / 0.13;
              const easeMain = 1 - Math.pow(1 - tm, 3.5);

              const scale = 0.005 + 0.995 * easeMain;
              const borderRadius = lerp(20, 0, easeMain);

              gsap.set(coverWrapRef.current, {
                scale: scale,
                opacity: 1,
                borderRadius: borderRadius,
              });

              if (coverVideoElRef.current) {
                gsap.set(coverVideoElRef.current, { filter: "none" });
              }
            } else {
              // Fullscreen continuous background (p > 0.53)
              gsap.set(coverWrapRef.current, {
                scale: 1.00,
                opacity: 1,
                borderRadius: 0,
              });

              // Progressive blur ONLY after reaching fullscreen when PsycheAI reveals
              if (coverVideoElRef.current) {
                let blur = 0;
                if (p >= 0.53 && p <= 0.63) {
                  const t_blur = (p - 0.53) / 0.10;
                  blur = t_blur * 20;
                } else if (p > 0.63) {
                  // Keep blur at 20 through About Us and How It Works
                  // (atmospheric depth for all subsequent sections)
                  blur = 20;
                }
                gsap.set(coverVideoElRef.current, {
                  filter: blur > 0 ? `blur(${blur.toFixed(1)}px)` : "none",
                });
              }
            }
          }

          /* ─────────────────────────────────────────────────────
             4. OUTRO (PsycheAI) HEADER & SUBHEADER (0.53 → 0.68)
          ───────────────────────────────────────────────────── */
          const titleEl    = titleHeaderRef.current;
          const subTitleEl = titleSubheaderRef.current;
          if (titleEl && subTitleEl) {
            if (p < 0.53) {
              gsap.set(titleEl, { opacity: 0, y: 20 });
              gsap.set(subTitleEl, { opacity: 0, y: 15 });
            } else if (p >= 0.53 && p <= 0.61) {
              const t_in = (p - 0.53) / 0.08;
              const easeIn = 1 - Math.pow(1 - t_in, 3);
              gsap.set(titleEl, {
                opacity: easeIn,
                y: 20 * (1 - easeIn),
              });
              const t_sub = Math.max(0, (p - 0.55) / 0.06);
              const easeSub = 1 - Math.pow(1 - t_sub, 3);
              gsap.set(subTitleEl, {
                opacity: easeSub * 0.9,
                y: 15 * (1 - easeSub),
              });
            } else if (p > 0.61 && p <= 0.67) {
              const t_out = (p - 0.61) / 0.06;
              gsap.set(titleEl, {
                opacity: 1 - t_out,
                y: -t_out * 25,
              });
              gsap.set(subTitleEl, {
                opacity: 0.9 * (1 - t_out),
                y: -t_out * 25,
              });
            } else {
              gsap.set([titleEl, subTitleEl], { opacity: 0 });
            }
          }

          /* ─────────────────────────────────────────────────────
             5. ABOUT US EDITORIAL SECTION (0.68 → 0.86)
             Now fades OUT at 0.82–0.88 to transition into How It Works
          ───────────────────────────────────────────────────── */
          if (aboutSectionRef.current) {
            if (p < 0.68) {
              gsap.set(aboutSectionRef.current, {
                opacity: 0,
                y: 30,
                pointerEvents: "none",
              });
            } else if (p >= 0.68 && p <= 0.78) {
              // Fade in
              const t_ab = (p - 0.68) / 0.10;
              const easeAb = easeOutCubic(t_ab);
              gsap.set(aboutSectionRef.current, {
                opacity: easeAb,
                y: (1 - easeAb) * 30,
                pointerEvents: "auto",
              });
            } else if (p > 0.78 && p <= 0.82) {
              // Hold at full opacity
              gsap.set(aboutSectionRef.current, {
                opacity: 1,
                y: 0,
                pointerEvents: "auto",
              });
            } else if (p > 0.82 && p <= 0.88) {
              // Fade out — handing off to How It Works
              const t_out = (p - 0.82) / 0.06;
              gsap.set(aboutSectionRef.current, {
                opacity: 1 - t_out,
                y: -t_out * 20,
                pointerEvents: "none",
              });
            } else {
              // p > 0.88 — fully gone
              gsap.set(aboutSectionRef.current, {
                opacity: 0,
                y: -20,
                pointerEvents: "none",
              });
            }
          }

          /* ─────────────────────────────────────────────────────
             6. HOW IT WORKS — EXTRA DARK OVERLAY (0.82 → 1.00)
             Darkens the video slightly more for chat readability
          ───────────────────────────────────────────────────── */
          if (howOverlayRef.current) {
            if (p < 0.82) {
              gsap.set(howOverlayRef.current, { opacity: 0 });
            } else if (p >= 0.82 && p <= 0.90) {
              const t = (p - 0.82) / 0.08;
              gsap.set(howOverlayRef.current, { opacity: easeOutCubic(t) * 0.45 });
            } else {
              gsap.set(howOverlayRef.current, { opacity: 0.45 });
            }
          }

          /* ─────────────────────────────────────────────────────
             7. HOW IT WORKS — OUTER WRAPPER (0.84 → 1.18)
          ───────────────────────────────────────────────────── */
          if (howItWorksRef.current) {
            if (p < 0.84) {
              gsap.set(howItWorksRef.current, { opacity: 0, pointerEvents: "none" });
            } else if (p >= 0.84 && p <= 1.10) {
              gsap.set(howItWorksRef.current, { opacity: 1, pointerEvents: "auto" });
            } else if (p > 1.10 && p <= 1.18) {
              const t = (p - 1.10) / 0.08;
              gsap.set(howItWorksRef.current, { opacity: 1 - t, pointerEvents: "none" });
            } else {
              gsap.set(howItWorksRef.current, { opacity: 0, pointerEvents: "none" });
            }
          }

          /* ─────────────────────────────────────────────────────
             8. HOW IT WORKS — HEADING (0.84 → 0.92)
          ───────────────────────────────────────────────────── */
          if (howHeaderRef.current) {
            if (p < 0.84) {
              gsap.set(howHeaderRef.current, { opacity: 0, y: 24 });
            } else if (p >= 0.84 && p <= 0.92) {
              const t = clamp01((p - 0.84) / 0.08);
              const e = easeOutCubic(t);
              gsap.set(howHeaderRef.current, { opacity: e, y: 24 * (1 - e) });
            } else {
              gsap.set(howHeaderRef.current, { opacity: 1, y: 0 });
            }
          }

          /* ─────────────────────────────────────────────────────
             9. HOW IT WORKS — CHAT CONTAINER (0.87 → 0.94)
          ───────────────────────────────────────────────────── */
          if (chatContainerRef.current) {
            if (p < 0.87) {
              gsap.set(chatContainerRef.current, { opacity: 0, scale: 0.96, y: 20 });
            } else if (p >= 0.87 && p <= 0.94) {
              const t = clamp01((p - 0.87) / 0.07);
              const e = easeOutCubic(t);
              gsap.set(chatContainerRef.current, { opacity: e, scale: lerp(0.96, 1, e), y: 20 * (1 - e) });
            } else {
              gsap.set(chatContainerRef.current, { opacity: 1, scale: 1, y: 0 });
            }
          }

          /* ─────────────────────────────────────────────────────
             10. CHAT MESSAGES — Sequential reveal (0.880 → 0.970)
          ───────────────────────────────────────────────────── */
          CHAT_MESSAGES.forEach((msg, idx) => {
            const el = chatMessageRefs.current[idx];
            if (!el) return;

            if (p < msg.revealAt) {
              gsap.set(el, { opacity: 0, y: 10 });
            } else {
              const t = clamp01((p - msg.revealAt) / MSG_FADE_RANGE);
              const e = easeOutCubic(t);
              gsap.set(el, { opacity: e, y: 10 * (1 - e) });
            }
          });

          /* ─────────────────────────────────────────────────────
             11. STEP LABELS — Staggered reveal (0.962 → 0.990)
          ───────────────────────────────────────────────────── */
          STEP_LABELS.forEach((label, idx) => {
            const el = stepLabelRefs.current[idx];
            if (!el) return;

            if (p < label.revealAt) {
              gsap.set(el, { opacity: 0, y: 8 });
            } else {
              const t = clamp01((p - label.revealAt) / STEP_FADE_RANGE);
              const e = easeOutCubic(t);
              gsap.set(el, { opacity: e, y: 8 * (1 - e) });
            }
          });

          /* ─────────────────────────────────────────────────────
             12. PRODUCT DASHBOARD (1.18 → 1.80)
          ───────────────────────────────────────────────────── */

          // Outer wrapper (reveals at 1.18, stays visible, then gracefully fades out after 1.75 to make way for center CTA)
          if (dashboardRef.current) {
            if (p < 1.18) {
              gsap.set(dashboardRef.current, { opacity: 0, pointerEvents: "none" });
            } else if (p >= 1.18 && p <= 1.75) {
              gsap.set(dashboardRef.current, { opacity: 1, pointerEvents: "auto", y: 0 });
            } else if (p > 1.75 && p <= 1.95) {
              const t = clamp01((p - 1.75) / 0.20);
              const e = easeOutCubic(t);
              gsap.set(dashboardRef.current, { opacity: 1 - e, y: -40 * e, pointerEvents: "none" });
            } else {
              gsap.set(dashboardRef.current, { opacity: 0, pointerEvents: "none" });
            }
          }

          // Section intro text (eyebrow + heading)
          if (dashboardIntroRef.current) {
            if (p < 1.18) {
              gsap.set(dashboardIntroRef.current, { opacity: 0, y: 20 });
            } else if (p >= 1.18 && p <= 1.28) {
              const t = clamp01((p - 1.18) / 0.10);
              const e = easeOutCubic(t);
              gsap.set(dashboardIntroRef.current, { opacity: e, y: 20 * (1 - e) });
            } else {
              gsap.set(dashboardIntroRef.current, { opacity: 1, y: 0 });
            }
          }

          // Dashboard shell
          if (dashboardShellRef.current) {
            if (p < 1.22) {
              gsap.set(dashboardShellRef.current, { opacity: 0, scale: 0.96, y: 24 });
            } else if (p >= 1.22 && p <= 1.34) {
              const t = clamp01((p - 1.22) / 0.12);
              const e = easeOutCubic(t);
              gsap.set(dashboardShellRef.current, { opacity: e, scale: lerp(0.96, 1, e), y: 24 * (1 - e) });
            } else {
              gsap.set(dashboardShellRef.current, { opacity: 1, scale: 1, y: 0 });
            }
          }

          // Nav
          if (dashboardNavRef.current) {
            if (p < 1.26) {
              gsap.set(dashboardNavRef.current, { opacity: 0, y: 8 });
            } else if (p >= 1.26 && p <= 1.34) {
              const t = clamp01((p - 1.26) / 0.08);
              const e = easeOutCubic(t);
              gsap.set(dashboardNavRef.current, { opacity: e, y: 8 * (1 - e) });
            } else {
              gsap.set(dashboardNavRef.current, { opacity: 1, y: 0 });
            }
          }

          // Heading + CTAs
          if (dashboardHeadRef.current) {
            if (p < 1.30) {
              gsap.set(dashboardHeadRef.current, { opacity: 0, y: 12 });
            } else if (p >= 1.30 && p <= 1.38) {
              const t = clamp01((p - 1.30) / 0.08);
              const e = easeOutCubic(t);
              gsap.set(dashboardHeadRef.current, { opacity: e, y: 12 * (1 - e) });
            } else {
              gsap.set(dashboardHeadRef.current, { opacity: 1, y: 0 });
            }
          }

          // Overview cards (staggered)
          if (dashboardCard1Ref.current) {
            if (p < 1.34) {
              gsap.set(dashboardCard1Ref.current, { opacity: 0, y: 16 });
            } else if (p >= 1.34 && p <= 1.42) {
              const t = clamp01((p - 1.34) / 0.08);
              const e = easeOutCubic(t);
              gsap.set(dashboardCard1Ref.current, { opacity: e, y: 16 * (1 - e) });
            } else {
              gsap.set(dashboardCard1Ref.current, { opacity: 1, y: 0 });
            }
          }
          if (dashboardCard2Ref.current) {
            if (p < 1.38) {
              gsap.set(dashboardCard2Ref.current, { opacity: 0, y: 16 });
            } else if (p >= 1.38 && p <= 1.46) {
              const t = clamp01((p - 1.38) / 0.08);
              const e = easeOutCubic(t);
              gsap.set(dashboardCard2Ref.current, { opacity: e, y: 16 * (1 - e) });
            } else {
              gsap.set(dashboardCard2Ref.current, { opacity: 1, y: 0 });
            }
          }
          if (dashboardCard3Ref.current) {
            if (p < 1.42) {
              gsap.set(dashboardCard3Ref.current, { opacity: 0, y: 16 });
            } else if (p >= 1.42 && p <= 1.50) {
              const t = clamp01((p - 1.42) / 0.08);
              const e = easeOutCubic(t);
              gsap.set(dashboardCard3Ref.current, { opacity: e, y: 16 * (1 - e) });
            } else {
              gsap.set(dashboardCard3Ref.current, { opacity: 1, y: 0 });
            }
          }

          // Conversation insights
          if (dashboardInsightsRef.current) {
            if (p < 1.46) {
              gsap.set(dashboardInsightsRef.current, { opacity: 0, y: 20 });
            } else if (p >= 1.46 && p <= 1.56) {
              const t = clamp01((p - 1.46) / 0.10);
              const e = easeOutCubic(t);
              gsap.set(dashboardInsightsRef.current, { opacity: e, y: 20 * (1 - e) });
            } else {
              gsap.set(dashboardInsightsRef.current, { opacity: 1, y: 0 });
            }
          }

          // Personal insights
          if (dashboardPersonalRef.current) {
            if (p < 1.52) {
              gsap.set(dashboardPersonalRef.current, { opacity: 0, y: 20 });
            } else if (p >= 1.52 && p <= 1.64) {
              const t = clamp01((p - 1.52) / 0.12);
              const e = easeOutCubic(t);
              gsap.set(dashboardPersonalRef.current, { opacity: e, y: 20 * (1 - e) });
            } else {
              gsap.set(dashboardPersonalRef.current, { opacity: 1, y: 0 });
            }
          }

          /* ─────────────────────────────────────────────────────
             13. FINAL HERO CENTER CTA (1.85 → 2.35)
          ───────────────────────────────────────────────────── */
          if (centerCtaRef.current) {
            if (p < 1.85) {
              gsap.set(centerCtaRef.current, { opacity: 0, scale: 0.9, y: 30, pointerEvents: "none" });
            } else if (p >= 1.85 && p <= 2.05) {
              const t = clamp01((p - 1.85) / 0.20);
              const e = easeOutCubic(t);
              gsap.set(centerCtaRef.current, {
                opacity: e,
                scale: lerp(0.9, 1, e),
                y: 30 * (1 - e),
                pointerEvents: "auto",
              });
            } else {
              gsap.set(centerCtaRef.current, { opacity: 1, scale: 1, y: 0, pointerEvents: "auto" });
            }
          }
        },
      });
    }

    initAnimations();

    let resizeTimer: ReturnType<typeof setTimeout>;
    const onResize = () => {
      clearTimeout(resizeTimer);
      resizeTimer = setTimeout(() => {
        ScrollTrigger.refresh();
        initAnimations();
      }, 200);
    };
    window.addEventListener("resize", onResize);

    return () => {
      window.removeEventListener("resize", onResize);
      clearTimeout(resizeTimer);
      introSplitRef.current?.revert();
      stRef.current?.kill();
      ScrollTrigger.getAll().forEach((t) => t.kill());
    };
  }, []);

  return (
    <section
      ref={sectionRef}
      className="spotlight relative w-full h-screen overflow-hidden flex items-center justify-center"
      style={{ background: "#080808" }}
      aria-label="PsycheAI showcase spotlight scroll section"
    >
      {/* ── Radial vignette ── */}
      <div
        className="absolute inset-0 z-10 pointer-events-none"
        style={{
          background:
            "radial-gradient(ellipse 65% 65% at 50% 50%, transparent 25%, rgba(8,8,8,0.9) 100%)",
        }}
      />

      {/* ── INTRO HEADER (Sized strictly inside the central void) ── */}
      <div
        className="spotlight-intro-header absolute z-20 text-center pointer-events-none select-none flex flex-col items-center justify-center px-4"
        style={{
          top: "50%",
          left: "50%",
          transform: "translate(-50%, -50%)",
          width: "clamp(280px, 42vw, 560px)",
          maxWidth: "90vw",
        }}
      >
        <h1
          ref={introHeaderRef}
          className="font-heading tracking-tight"
          style={{
            fontFamily:    "var(--font-heading), serif",
            fontSize:      "clamp(1.8rem, 3.2vw, 3.4rem)",
            fontWeight:    400,
            lineHeight:    1.2,
            letterSpacing: "-0.01em",
            color:         "#ffffff",
            textShadow:    "0 15px 40px rgba(0,0,0,0.9)",
          }}
        >
          A silent struggle, understood through multimodal signals.
        </h1>
      </div>

      {/* ── OUTRO HEADER (PsycheAI Heading + Multimodal Subtitle) ── */}
      <div
        ref={titleContainerRef}
        className="spotlight-outro-header absolute z-20 text-center w-11/12 max-w-5xl pointer-events-none select-none flex flex-col items-center justify-center gap-2 sm:gap-3"
        style={{ top: "50%", left: "50%", transform: "translate(-50%, -50%)" }}
      >
        <h1
          ref={titleHeaderRef}
          className="font-brand tracking-wide"
          style={{
            fontFamily:    "var(--font-brand), cursive",
            fontSize:      "clamp(3.5rem, 8.5vw, 7.8rem)",
            fontWeight:    400,
            lineHeight:    1.05,
            letterSpacing: "0.02em",
            color:         "#ffffff",
            textShadow:    "0 15px 40px rgba(0,0,0,0.8)",
          }}
        >
          PsycheAI
        </h1>
        <p
          ref={titleSubheaderRef}
          className="font-body font-light tracking-wide text-[#e5e0d8]"
          style={{
            fontFamily:    "var(--font-body), sans-serif",
            fontSize:      "clamp(1.1rem, 2.2vw, 2rem)",
            fontWeight:    300,
            letterSpacing: "0.02em",
            opacity:       0.9,
            textShadow:    "0 10px 30px rgba(0,0,0,0.8)",
          }}
        >
          Multimodal depression detection.
        </p>
      </div>

      {/* ── ABOUT US (Clean typography over blurred main video) ── */}
      <div
        ref={aboutSectionRef}
        className="absolute z-30 pointer-events-auto flex flex-col justify-center items-center px-6 sm:px-10 md:px-16 w-full max-w-4xl text-center select-none"
        style={{
          top: "50%",
          left: "50%",
          transform: "translate(-50%, -50%)",
          opacity: 0,
        }}
      >
        <h2
          className="font-heading tracking-tight text-white mb-6 md:mb-8"
          style={{
            fontFamily: "var(--font-heading), serif",
            fontSize: "clamp(2.8rem, 6vw, 4.8rem)",
            fontWeight: 400,
            lineHeight: 1.1,
            letterSpacing: "-0.02em",
            textShadow: "0 10px 30px rgba(0,0,0,0.8)",
          }}
        >
          About <span style={{ color: "#c9a96e", fontWeight: 400, fontStyle: "italic" }}>Us</span>
        </h2>

        <p
          className="text-[#e5e0d8] font-body font-light leading-relaxed md:leading-loose text-base sm:text-lg md:text-xl"
          style={{
            fontFamily: "var(--font-body), sans-serif",
            fontWeight: 300,
            lineHeight: 1.8,
            letterSpacing: "0.01em",
            textShadow: "0 4px 20px rgba(0,0,0,0.9)",
          }}
        >
          We&apos;re an AI-powered conversational platform built to make mental health check-ins simple, accessible, and judgment-free. Instead of clinical forms, we use natural conversation — analyzing both text and voice responses — to give you a clear, plain-language snapshot of your mental wellbeing. Every question adapts based on what you share, and crisis resources are always just one step away if you need them. This isn&apos;t a diagnosis — it&apos;s a low-friction first step toward self-awareness and, when needed, professional help. Your data stays encrypted and fully in your control, because when it comes to mental health, trust matters most.
        </p>
      </div>

      {/* ── HOW IT WORKS — Extra overlay for chat readability ── */}
      <div
        ref={howOverlayRef}
        className="absolute inset-0 pointer-events-none"
        style={{
          zIndex: 31,
          background: "rgba(4, 4, 8, 0)",
        }}
      />

      {/* ── HOW IT WORKS ── */}
      <div
        ref={howItWorksRef}
        className="absolute z-[32] w-full h-full flex items-center justify-center"
        style={{ opacity: 0, pointerEvents: "none", padding: "0 clamp(20px, 5vw, 80px)" }}
        aria-label="How It Works section"
      >
        {/* ── Two-column grid: Left = text + steps | Right = chat ── */}
        <div
          style={{
            display:             "grid",
            gridTemplateColumns: "1fr 1fr",
            gap:                 "clamp(32px, 5vw, 72px)",
            alignItems:          "center",
            width:               "100%",
            maxWidth:            "1200px",
          }}
        >
          {/* ══ LEFT COLUMN — Text content + step labels ══ */}
          <div
            ref={howHeaderRef}
            className="flex flex-col select-none"
            style={{ opacity: 0 }}
          >
            {/* Eyebrow */}
            <p
              style={{
                fontFamily:    "var(--font-inter, system-ui, sans-serif)",
                fontSize:      "clamp(0.6rem, 0.85vw, 0.72rem)",
                fontWeight:    500,
                letterSpacing: "0.28em",
                textTransform: "uppercase",
                color:         "#c9a96e",
                marginBottom:  "1rem",
                textShadow:    "0 2px 12px rgba(0,0,0,0.8)",
              }}
            >
              How It Works
            </p>

            {/* Main heading */}
            <h2
              style={{
                fontFamily:    "var(--font-inter, system-ui, sans-serif)",
                fontSize:      "clamp(1.6rem, 3.2vw, 3rem)",
                fontWeight:    300,
                lineHeight:    1.08,
                letterSpacing: "-0.035em",
                color:         "#f0ede8",
                textShadow:    "0 10px 36px rgba(0,0,0,0.85)",
                marginBottom:  "1rem",
              }}
            >
              A conversation,<br />not a questionnaire.
            </h2>

            {/* Supporting text */}
            <p
              style={{
                fontFamily:    "var(--font-inter, system-ui, sans-serif)",
                fontSize:      "clamp(0.78rem, 1.2vw, 0.98rem)",
                fontWeight:    300,
                lineHeight:    1.7,
                letterSpacing: "-0.005em",
                color:         "rgba(229, 224, 216, 0.72)",
                textShadow:    "0 4px 20px rgba(0,0,0,0.9)",
                marginBottom:  "clamp(28px, 4vw, 48px)",
              }}
            >
              Talk naturally. Our AI adapts to your responses,
              analyzes relevant language and speech cues, and
              turns the conversation into a simple snapshot of
              your mental wellbeing.
            </p>

            {/* Step labels — stacked vertically on left */}
            <div className="flex flex-col gap-5">
              {STEP_LABELS.map((label, idx) => (
                <div
                  key={idx}
                  ref={(el) => { stepLabelRefs.current[idx] = el; }}
                  className="flex items-start gap-4"
                  style={{ opacity: 0 }}
                >
                  {/* Step number */}
                  <span
                    style={{
                      fontFamily:    "var(--font-inter, system-ui, sans-serif)",
                      fontSize:      "0.6rem",
                      letterSpacing: "0.22em",
                      color:         "rgba(201,169,110,0.5)",
                      fontWeight:    500,
                      paddingTop:    "2px",
                      flexShrink:    0,
                      minWidth:      "24px",
                    }}
                  >
                    {label.num}
                  </span>

                  <div className="flex flex-col">
                    {/* Title */}
                    <span
                      style={{
                        fontFamily:    "var(--font-inter, system-ui, sans-serif)",
                        fontSize:      "clamp(0.82rem, 1.15vw, 0.95rem)",
                        fontWeight:    400,
                        letterSpacing: "-0.01em",
                        color:         "#f0ede8",
                        marginBottom:  "3px",
                        textShadow:    "0 2px 12px rgba(0,0,0,0.8)",
                      }}
                    >
                      {label.title}
                    </span>

                    {/* Description */}
                    <span
                      style={{
                        fontFamily:    "var(--font-inter, system-ui, sans-serif)",
                        fontSize:      "clamp(0.67rem, 0.95vw, 0.78rem)",
                        fontWeight:    300,
                        color:         "rgba(229,224,216,0.48)",
                        lineHeight:    1.5,
                        letterSpacing: "-0.003em",
                        textShadow:    "0 2px 10px rgba(0,0,0,0.7)",
                      }}
                    >
                      {label.desc}
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* ══ RIGHT COLUMN — Chat interface ══ */}
          <div
            ref={chatContainerRef}
            className="select-none"
            style={{
              background:           "rgba(10, 10, 14, 0.72)",
              backdropFilter:       "blur(24px)",
              WebkitBackdropFilter: "blur(24px)",
              border:               "1px solid rgba(255,255,255,0.09)",
              borderRadius:         "18px",
              boxShadow:            "0 24px 80px rgba(0,0,0,0.6), 0 0 0 1px rgba(255,255,255,0.04)",
              padding:              "clamp(16px, 2.5vw, 28px)",
              opacity:              0,
              width:                "100%",
            }}
            role="log"
            aria-label="AI conversation demonstration"
            aria-live="polite"
          >
            {/* Chat header bar */}
            <div
              className="flex items-center gap-3 mb-4 pb-3"
              style={{ borderBottom: "1px solid rgba(255,255,255,0.07)" }}
            >
              {/* Dots */}
              <div className="flex gap-1.5">
                <div style={{ width: 6, height: 6, borderRadius: "50%", background: "rgba(255,255,255,0.15)" }} />
                <div style={{ width: 6, height: 6, borderRadius: "50%", background: "rgba(255,255,255,0.15)" }} />
                <div style={{ width: 6, height: 6, borderRadius: "50%", background: "rgba(255,255,255,0.15)" }} />
              </div>
              {/* Label */}
              <span
                style={{
                  fontFamily:    "var(--font-inter, system-ui, sans-serif)",
                  fontSize:      "0.62rem",
                  letterSpacing: "0.16em",
                  textTransform: "uppercase",
                  color:         "rgba(201,169,110,0.7)",
                  fontWeight:    500,
                }}
              >
                PsycheAI · Wellbeing Check-in
              </span>
              {/* Live indicator */}
              <div className="ml-auto flex items-center gap-1.5" style={{ opacity: 0.6 }}>
                <div
                  style={{
                    width:        6,
                    height:       6,
                    borderRadius: "50%",
                    background:   "#6ee7b7",
                    boxShadow:    "0 0 6px #6ee7b7",
                    animation:    "psycheai-pulse 2s ease-in-out infinite",
                  }}
                />
                <span
                  style={{
                    fontFamily:    "var(--font-inter, system-ui, sans-serif)",
                    fontSize:      "0.58rem",
                    letterSpacing: "0.12em",
                    color:         "rgba(229,224,216,0.6)",
                    textTransform: "uppercase",
                  }}
                >
                  Live
                </span>
              </div>
            </div>

            {/* Messages */}
            <div className="flex flex-col gap-3">
              {CHAT_MESSAGES.map((msg, idx) => (
                <div
                  key={idx}
                  ref={(el) => { chatMessageRefs.current[idx] = el; }}
                  className="flex flex-col"
                  style={{
                    alignItems: msg.author === "user" ? "flex-end" : "flex-start",
                    opacity: 0,
                  }}
                >
                  {/* Author label */}
                  <span
                    style={{
                      fontFamily:    "var(--font-inter, system-ui, sans-serif)",
                      fontSize:      "0.58rem",
                      letterSpacing: "0.18em",
                      textTransform: "uppercase",
                      color:         msg.author === "ai" ? "rgba(201,169,110,0.65)" : "rgba(180,180,200,0.5)",
                      fontWeight:    500,
                      marginBottom:  "4px",
                      paddingLeft:   msg.author === "ai" ? "2px" : "0",
                      paddingRight:  msg.author === "user" ? "2px" : "0",
                    }}
                  >
                    {msg.author === "ai" ? "AI" : "You"}
                  </span>

                  {/* Bubble */}
                  <div
                    style={{
                      maxWidth:     "88%",
                      background:   msg.author === "ai"
                        ? "rgba(255,255,255,0.07)"
                        : "rgba(201,169,110,0.14)",
                      border:       msg.author === "ai"
                        ? "1px solid rgba(255,255,255,0.08)"
                        : "1px solid rgba(201,169,110,0.22)",
                      borderRadius: msg.author === "ai"
                        ? "4px 14px 14px 14px"
                        : "14px 4px 14px 14px",
                      padding:      "9px 14px",
                      fontFamily:   "var(--font-inter, system-ui, sans-serif)",
                      fontSize:     "clamp(0.75rem, 1.1vw, 0.88rem)",
                      fontWeight:   300,
                      lineHeight:   1.55,
                      color:        msg.author === "ai"
                        ? "rgba(240, 237, 232, 0.9)"
                        : "rgba(240, 237, 232, 0.85)",
                      letterSpacing: "-0.004em",
                    }}
                  >
                    {msg.text}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>


      </div>
      {/* ── PRODUCT DASHBOARD ── */}
      <div
        ref={dashboardRef}
        className="absolute z-[34] w-full h-full flex flex-col items-center justify-center px-4 sm:px-6 overflow-hidden select-none"
        style={{ opacity: 0, pointerEvents: "none" }}
        aria-label="Product dashboard preview section"
      >
        {/* Section intro text */}
        <div
          ref={dashboardIntroRef}
          className="flex flex-col items-center text-center mb-3 md:mb-5"
          style={{ opacity: 0 }}
        >
          <p
            style={{
              fontFamily:    "var(--font-inter, system-ui, sans-serif)",
              fontSize:      "clamp(0.6rem, 0.85vw, 0.72rem)",
              fontWeight:    500,
              letterSpacing: "0.28em",
              textTransform: "uppercase",
              color:         "#c9a96e",
              marginBottom:  "0.5rem",
              textShadow:    "0 0 20px rgba(201,169,110,0.4)",
            }}
          >
            Your Wellbeing, At a Glance
          </p>
          <h2
            style={{
              fontFamily:    "var(--font-inter, system-ui, sans-serif)",
              fontSize:      "clamp(1.3rem, 2.6vw, 2.4rem)",
              fontWeight:    300,
              lineHeight:    1.15,
              letterSpacing: "-0.035em",
              color:         "#f0ede8",
              textShadow:    "0 10px 36px rgba(0,0,0,0.85)",
              marginBottom:  "0.4rem",
            }}
          >
            Turn a conversation into
            <br className="hidden sm:block" /> something you can understand.
          </h2>
        </div>


        {/* Dashboard shell */}
        <div
          ref={dashboardShellRef}
          className="dashboard-shell w-full"
          style={{
            maxWidth:             "1160px",
            background:           "rgba(14, 15, 18, 0.75)",
            backdropFilter:       "blur(40px) saturate(1.4)",
            WebkitBackdropFilter: "blur(40px) saturate(1.4)",
            border:               "1px solid rgba(201,169,110,0.18)",
            borderRadius:         "20px",
            boxShadow:            "0 0 0 1px rgba(255,255,255,0.04), 0 32px 80px rgba(0,0,0,0.60), 0 0 80px rgba(201,169,110,0.08), inset 0 1px 0 rgba(255,255,255,0.08)",
            overflow:             "hidden",
            opacity:              0,
          }}
          role="img"
          aria-label="Mental Wellbeing product dashboard preview"
        >
          {/* ── Dashboard Top Bar ── */}
          <div
            style={{
              display:       "flex",
              alignItems:    "center",
              padding:       "12px 20px",
              borderBottom:  "1px solid rgba(255,255,255,0.06)",
              background:    "rgba(10, 11, 14, 0.80)",
              gap:           "12px",
            }}
          >
            {/* Logo mark */}
            <div style={{ display: "flex", alignItems: "center", gap: "8px", flexShrink: 0 }}>
              <div style={{
                width: 28, height: 28, borderRadius: "8px",
                background: "linear-gradient(135deg, #c9a96e 0%, #8c6b32 100%)",
                display: "flex", alignItems: "center", justifyContent: "center",
                boxShadow: "0 2px 8px rgba(201,169,110,0.25)",
              }}>
                <svg width="14" height="14" viewBox="0 0 24 24" fill="#080808">
                  <path d="M12 0l2 8 8 2-8 2-2 8-2-8-8-2 8-2 2-8z"/>
                </svg>
              </div>
              <span style={{
                fontFamily: "var(--font-inter, system-ui, sans-serif)",
                fontSize: "0.82rem", fontWeight: 600, color: "#f0ede8",
                letterSpacing: "-0.02em",
              }}>PsycheAI</span>
            </div>

            {/* Nav tabs */}
            <div
              ref={dashboardNavRef}
              className="dashboard-nav"
              style={{
                display: "flex", gap: "4px", marginLeft: "16px",
                background: "rgba(255,255,255,0.04)", borderRadius: "10px",
                padding: "3px", opacity: 0, flexShrink: 0,
              }}
            >
              {["Overview", "Check-ins", "Analysis", "Insights", "History"].map((tab, i) => (
                <button
                  key={tab}
                  id={`dashboard-nav-${tab.toLowerCase()}`}
                  style={{
                    fontFamily:    "var(--font-inter, system-ui, sans-serif)",
                    fontSize:      "0.68rem",
                    fontWeight:    i === 0 ? 500 : 400,
                    letterSpacing: "-0.01em",
                    color:         i === 0 ? "#f0ede8" : "rgba(229,224,216,0.45)",
                    background:    i === 0 ? "rgba(201,169,110,0.15)" : "transparent",
                    border:        i === 0 ? "1px solid rgba(201,169,110,0.28)" : "1px solid transparent",
                    borderRadius:  "7px",
                    padding:       "5px 12px",
                    cursor:        "default",
                    whiteSpace:    "nowrap",
                  }}
                >
                  {tab}
                </button>
              ))}
            </div>

            {/* Spacer */}
            <div style={{ flex: 1 }} />

            {/* Right side: notification + avatar */}
            <div style={{ display: "flex", alignItems: "center", gap: "10px", flexShrink: 0 }}>
              <div style={{
                width: 30, height: 30, borderRadius: "8px",
                background: "rgba(255,255,255,0.06)",
                border: "1px solid rgba(255,255,255,0.08)",
                display: "flex", alignItems: "center", justifyContent: "center",
              }}>
                <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="rgba(229,224,216,0.5)" strokeWidth="2">
                  <path d="M18 8A6 6 0 0 0 6 8c0 7-3 9-3 9h18s-3-2-3-9"/>
                  <path d="M13.73 21a2 2 0 0 1-3.46 0"/>
                </svg>
              </div>
              <div style={{
                width: 30, height: 30, borderRadius: "50%",
                background: "linear-gradient(135deg, #c9a96e 0%, #e8d5b0 100%)",
                border: "1.5px solid rgba(201,169,110,0.4)",
              }} />
            </div>
          </div>

          {/* ── Dashboard Body ── */}
          <div style={{ padding: "18px 20px", display: "flex", flexDirection: "column", gap: "14px" }}>

            {/* ── Row 1: heading + CTA ── */}
            <div
              ref={dashboardHeadRef}
              style={{
                display: "flex", alignItems: "flex-start", justifyContent: "space-between",
                gap: "12px", flexWrap: "wrap", opacity: 0,
              }}
            >
              <div>
                <h3 style={{
                  fontFamily: "var(--font-inter, system-ui, sans-serif)",
                  fontSize: "clamp(1.05rem, 1.6vw, 1.45rem)",
                  fontWeight: 300, letterSpacing: "-0.03em",
                  color: "#f0ede8", margin: 0,
                }}>Mental Wellbeing</h3>
                <p style={{
                  fontFamily: "var(--font-inter, system-ui, sans-serif)",
                  fontSize: "0.74rem", fontWeight: 300,
                  color: "rgba(229,224,216,0.5)", margin: "3px 0 0",
                }}>Track your check-ins and understand patterns over time.</p>
              </div>
              <div style={{ display: "flex", gap: "8px", alignItems: "center", flexShrink: 0 }}>
                <Link
                  href="/dashboard"
                  id="dashboard-cta-history"
                  style={{
                    fontFamily: "var(--font-inter, system-ui, sans-serif)",
                    fontSize: "0.72rem", fontWeight: 400,
                    color: "rgba(229,224,216,0.7)",
                    background: "rgba(255,255,255,0.05)",
                    border: "1px solid rgba(255,255,255,0.08)",
                    borderRadius: "8px", padding: "7px 14px",
                    cursor: "pointer", whiteSpace: "nowrap",
                    textDecoration: "none",
                    display: "inline-block",
                  }}
                >View History</Link>
                <Link
                  href="/dashboard"
                  id="dashboard-cta-checkin"
                  style={{
                    fontFamily: "var(--font-inter, system-ui, sans-serif)",
                    fontSize: "0.72rem", fontWeight: 500,
                    color: "#080808",
                    background: "linear-gradient(135deg, #c9a96e 0%, #e8d5b0 100%)",
                    border: "1px solid rgba(201,169,110,0.4)",
                    borderRadius: "8px", padding: "7px 16px",
                    cursor: "pointer", whiteSpace: "nowrap",
                    boxShadow: "0 4px 16px rgba(201,169,110,0.25)",
                    textDecoration: "none",
                    display: "inline-block",
                  }}
                >Start a Check-in →</Link>
              </div>
            </div>

            {/* ── Row 2: Overview cards ── */}
            <div className="db-cards-row" style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: "12px" }}>

              {/* Card 1: Today's Check-in */}
              <div
                ref={dashboardCard1Ref}
                style={{
                  background: "rgba(18,20,26,0.85)",
                  border: "1px solid rgba(201,169,110,0.12)",
                  borderRadius: "14px",
                  padding: "14px 16px",
                  opacity: 0,
                  boxShadow: "0 8px 32px rgba(0,0,0,0.4)",
                }}
              >
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: "10px" }}>
                  <span style={{
                    fontFamily: "var(--font-inter, system-ui, sans-serif)",
                    fontSize: "0.65rem", fontWeight: 500,
                    letterSpacing: "0.14em", textTransform: "uppercase",
                    color: "rgba(229,224,216,0.45)",
                  }}>Today&apos;s Check-in</span>
                  <div style={{
                    width: 7, height: 7, borderRadius: "50%",
                    background: "#c9a96e",
                    boxShadow: "0 0 8px rgba(201,169,110,0.6)",
                  }} />
                </div>
                <div style={{ display: "flex", flexDirection: "column", gap: "6px" }}>
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                    <span style={{ fontSize: "0.7rem", color: "rgba(229,224,216,0.5)", fontFamily: "var(--font-inter, system-ui, sans-serif)" }}>Last check-in</span>
                    <span style={{ fontSize: "0.7rem", color: "#f0ede8", fontFamily: "var(--font-inter, system-ui, sans-serif)", fontWeight: 400 }}>Today</span>
                  </div>
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                    <span style={{ fontSize: "0.7rem", color: "rgba(229,224,216,0.5)", fontFamily: "var(--font-inter, system-ui, sans-serif)" }}>Status</span>
                    <span style={{
                      fontSize: "0.65rem", fontFamily: "var(--font-inter, system-ui, sans-serif)",
                      color: "#c9a96e", background: "rgba(201,169,110,0.10)",
                      border: "1px solid rgba(201,169,110,0.25)",
                      borderRadius: "6px", padding: "2px 8px",
                    }}>Completed</span>
                  </div>
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                    <span style={{ fontSize: "0.7rem", color: "rgba(229,224,216,0.5)", fontFamily: "var(--font-inter, system-ui, sans-serif)" }}>Streak</span>
                    <span style={{ fontSize: "0.7rem", color: "#e8d5b0", fontFamily: "var(--font-inter, system-ui, sans-serif)" }}>5 days ✦</span>
                  </div>
                </div>
              </div>

              {/* Card 2: Wellbeing Snapshot */}
              <div
                ref={dashboardCard2Ref}
                style={{
                  background: "rgba(18,20,26,0.85)",
                  border: "1px solid rgba(201,169,110,0.12)",
                  borderRadius: "14px",
                  padding: "14px 16px",
                  opacity: 0,
                  boxShadow: "0 8px 32px rgba(0,0,0,0.4)",
                }}
              >
                <div style={{ marginBottom: "10px" }}>
                  <span style={{
                    fontFamily: "var(--font-inter, system-ui, sans-serif)",
                    fontSize: "0.65rem", fontWeight: 500,
                    letterSpacing: "0.14em", textTransform: "uppercase",
                    color: "rgba(229,224,216,0.45)",
                  }}>Wellbeing Snapshot</span>
                </div>
                {/* Waveform visualization */}
                <div style={{ position: "relative", marginBottom: "10px" }}>
                  <svg viewBox="0 0 160 40" style={{ width: "100%", height: "34px" }}>
                    <defs>
                      <linearGradient id="db-wave-grad" x1="0%" y1="0%" x2="100%" y2="0%">
                        <stop offset="0%" stopColor="#c9a96e" stopOpacity="0.4" />
                        <stop offset="50%" stopColor="#e8d5b0" stopOpacity="0.95" />
                        <stop offset="100%" stopColor="#c9a96e" stopOpacity="0.5" />
                      </linearGradient>
                    </defs>
                    <path d="M0,20 Q8,10 16,20 T32,20 T48,20 T64,16 T80,20 T96,22 T112,20 T128,18 T144,20 T160,20"
                      fill="none" stroke="url(#db-wave-grad)" strokeWidth="2" strokeLinecap="round" />
                    <path d="M0,20 Q8,14 16,20 T32,22 T48,18 T64,20 T80,20 T96,20 T112,20 T128,20 T144,20 T160,20"
                      fill="none" stroke="rgba(255,255,255,0.08)" strokeWidth="1" strokeLinecap="round" />
                  </svg>
                </div>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                  <span style={{ fontSize: "0.7rem", color: "rgba(229,224,216,0.5)", fontFamily: "var(--font-inter, system-ui, sans-serif)" }}>Current snapshot</span>
                  <span style={{
                    fontSize: "0.75rem", fontFamily: "var(--font-inter, system-ui, sans-serif)",
                    fontWeight: 500, color: "#c9a96e",
                    background: "rgba(201,169,110,0.10)",
                    border: "1px solid rgba(201,169,110,0.25)",
                    borderRadius: "6px", padding: "2px 10px",
                  }}>Steady</span>
                </div>
              </div>

              {/* Card 3: Recent Pattern */}
              <div
                ref={dashboardCard3Ref}
                style={{
                  background: "rgba(18,20,26,0.85)",
                  border: "1px solid rgba(201,169,110,0.12)",
                  borderRadius: "14px",
                  padding: "14px 16px",
                  opacity: 0,
                  boxShadow: "0 8px 32px rgba(0,0,0,0.4)",
                }}
              >
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: "8px" }}>
                  <span style={{
                    fontFamily: "var(--font-inter, system-ui, sans-serif)",
                    fontSize: "0.65rem", fontWeight: 500,
                    letterSpacing: "0.14em", textTransform: "uppercase",
                    color: "rgba(229,224,216,0.45)",
                  }}>Check-in Patterns</span>
                  <span style={{ fontSize: "0.62rem", color: "rgba(229,224,216,0.35)", fontFamily: "var(--font-inter)" }}>30d</span>
                </div>
                {/* Mini sparkline */}
                <svg viewBox="0 0 120 36" style={{ width: "100%", height: "34px", marginBottom: "6px" }}>
                  <defs>
                    <linearGradient id="db-spark-area" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor="#c9a96e" stopOpacity="0.35" />
                      <stop offset="100%" stopColor="#c9a96e" stopOpacity="0.0" />
                    </linearGradient>
                  </defs>
                  <path d="M0,28 L15,22 L30,24 L45,18 L60,20 L75,14 L90,16 L105,12 L120,10 L120,36 L0,36 Z"
                    fill="url(#db-spark-area)" />
                  <path d="M0,28 L15,22 L30,24 L45,18 L60,20 L75,14 L90,16 L105,12 L120,10"
                    fill="none" stroke="#c9a96e" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
                  {[{ x: 15, y: 22 }, { x: 45, y: 18 }, { x: 75, y: 14 }, { x: 105, y: 12 }, { x: 120, y: 10 }].map((pt, i) => (
                    <circle key={i} cx={pt.x} cy={pt.y} r="2" fill="#e8d5b0" opacity={0.9} />
                  ))}
                </svg>
                <p style={{ fontSize: "0.68rem", color: "rgba(229,224,216,0.45)", fontFamily: "var(--font-inter, system-ui, sans-serif)", margin: 0 }}>
                  Consistent check-ins this month
                </p>
              </div>
            </div>

            {/* ── Row 3: Insights panels ── */}
            <div className="db-insights-row" style={{ display: "grid", gridTemplateColumns: "1.5fr 1fr", gap: "12px" }}>

              {/* Conversation Insights */}
              <div
                ref={dashboardInsightsRef}
                style={{
                  background: "rgba(18,20,26,0.85)",
                  border: "1px solid rgba(201,169,110,0.12)",
                  borderRadius: "14px",
                  padding: "16px 18px",
                  opacity: 0,
                  boxShadow: "0 8px 32px rgba(0,0,0,0.4)",
                }}
              >
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "12px" }}>
                  <span style={{
                    fontFamily: "var(--font-inter, system-ui, sans-serif)",
                    fontSize: "0.78rem", fontWeight: 500,
                    color: "#f0ede8", letterSpacing: "-0.01em",
                  }}>Recent Conversation Insights</span>
                  <span style={{
                    fontSize: "0.62rem", color: "rgba(229,224,216,0.4)",
                    fontFamily: "var(--font-inter, system-ui, sans-serif)",
                  }}>Today</span>
                </div>

                {/* Quote from check-in */}
                <div style={{
                  background: "rgba(201,169,110,0.06)",
                  border: "1px solid rgba(201,169,110,0.15)",
                  borderRadius: "10px",
                  padding: "10px 14px",
                  marginBottom: "12px",
                  borderLeft: "2px solid rgba(201,169,110,0.5)",
                }}>
                  <p style={{
                    fontFamily: "var(--font-inter, system-ui, sans-serif)",
                    fontSize: "0.75rem", fontWeight: 300, fontStyle: "italic",
                    color: "rgba(240,237,232,0.85)", lineHeight: 1.6, margin: 0,
                  }}>
                    &ldquo;I&apos;ve been finding it harder to focus lately.&rdquo;
                  </p>
                  <span style={{
                    fontSize: "0.6rem", color: "rgba(229,224,216,0.4)",
                    fontFamily: "var(--font-inter, system-ui, sans-serif)",
                    display: "block", marginTop: "4px",
                  }}>From your recent check-in</span>
                </div>

                {/* Extracted signals */}
                <div style={{ display: "flex", gap: "8px", flexWrap: "wrap" }}>
                  {[
                    { label: "Focus",  color: "#c9a96e", bg: "rgba(201,169,110,0.10)",  border: "rgba(201,169,110,0.25)" },
                    { label: "Energy", color: "#e8d5b0", bg: "rgba(232,213,176,0.10)",  border: "rgba(232,213,176,0.25)" },
                    { label: "Mood",   color: "#d4af37", bg: "rgba(212,175,55,0.10)",   border: "rgba(212,175,55,0.25)" },
                  ].map(sig => (
                    <span key={sig.label} style={{
                      fontFamily: "var(--font-inter, system-ui, sans-serif)",
                      fontSize: "0.67rem", fontWeight: 400,
                      color: sig.color,
                      background: sig.bg,
                      border: `1px solid ${sig.border}`,
                      borderRadius: "20px", padding: "3px 12px",
                    }}>{sig.label}</span>
                  ))}
                  <span style={{
                    fontFamily: "var(--font-inter, system-ui, sans-serif)",
                    fontSize: "0.62rem", color: "rgba(229,224,216,0.35)",
                    alignSelf: "center", marginLeft: "4px",
                  }}>Patterns observed, not diagnosed</span>
                </div>

                {/* Mini timeline */}
                <div style={{ marginTop: "12px", paddingTop: "10px", borderTop: "1px solid rgba(255,255,255,0.05)" }}>
                  <span style={{
                    fontSize: "0.62rem", color: "rgba(229,224,216,0.35)",
                    fontFamily: "var(--font-inter, system-ui, sans-serif)",
                    letterSpacing: "0.12em", textTransform: "uppercase",
                    display: "block", marginBottom: "6px",
                  }}>Recent check-ins</span>
                  <div style={{ display: "flex", gap: "6px", alignItems: "flex-end" }}>
                    {[0.6, 0.8, 0.5, 0.9, 0.7, 1.0, 0.85].map((h, i) => (
                      <div key={i} style={{
                        flex: 1,
                        height: `${Math.round(h * 26)}px`,
                        background: i === 6
                          ? "linear-gradient(180deg, #e8d5b0 0%, #c9a96e 100%)"
                          : `rgba(201,169,110,${0.15 + h * 0.25})`,
                        borderRadius: "3px 3px 2px 2px",
                        transition: "height 0.3s",
                      }} />
                    ))}
                  </div>
                </div>
              </div>

              {/* Personal Insights */}
              <div
                ref={dashboardPersonalRef}
                style={{
                  background: "rgba(18,20,26,0.85)",
                  border: "1px solid rgba(201,169,110,0.12)",
                  borderRadius: "14px",
                  padding: "16px 18px",
                  display: "flex", flexDirection: "column",
                  opacity: 0,
                  boxShadow: "0 8px 32px rgba(0,0,0,0.4)",
                }}
              >
                <span style={{
                  fontFamily: "var(--font-inter, system-ui, sans-serif)",
                  fontSize: "0.78rem", fontWeight: 500,
                  color: "#f0ede8", letterSpacing: "-0.01em",
                  marginBottom: "12px", display: "block",
                }}>Personal Insights</span>

                {/* Insight cards */}
                {[
                  {
                    icon: (
                      <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="#c9a96e" strokeWidth="2">
                        <circle cx="12" cy="12" r="10"/><path d="M12 6v6l4 2"/>
                      </svg>
                    ),
                    accentColor: "#c9a96e",
                    accentBg: "rgba(201,169,110,0.08)",
                    accentBorder: "rgba(201,169,110,0.18)",
                    title: "Take a short pause",
                    desc: "A gentle reminder based on recent check-in patterns.",
                  },
                  {
                    icon: (
                      <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="#e8d5b0" strokeWidth="2">
                        <path d="M12 2l3 7h7l-5.5 4 2 7L12 16l-6.5 4 2-7L2 9h7z"/>
                      </svg>
                    ),
                    accentColor: "#e8d5b0",
                    accentBg: "rgba(232,213,176,0.08)",
                    accentBorder: "rgba(232,213,176,0.18)",
                    title: "Keep your routine consistent",
                    desc: "A neutral self-care suggestion.",
                  },
                  {
                    icon: (
                      <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="#c9a96e" strokeWidth="2">
                        <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>
                      </svg>
                    ),
                    accentColor: "#c9a96e",
                    accentBg: "rgba(201,169,110,0.08)",
                    accentBorder: "rgba(201,169,110,0.18)",
                    title: "Check in again",
                    desc: "Continue tracking your wellbeing.",
                  },
                ].map((insight, i) => (
                  <div key={i} style={{
                    display: "flex", gap: "10px", alignItems: "flex-start",
                    padding: "9px 11px",
                    background: insight.accentBg,
                    border: `1px solid ${insight.accentBorder}`,
                    borderRadius: "10px",
                    marginBottom: i < 2 ? "7px" : 0,
                  }}>
                    <div style={{
                      width: 26, height: 26, borderRadius: "8px",
                      background: "rgba(255,255,255,0.05)",
                      border: `1px solid ${insight.accentBorder}`,
                      display: "flex", alignItems: "center", justifyContent: "center",
                      flexShrink: 0,
                    }}>
                      {insight.icon}
                    </div>
                    <div>
                      <span style={{
                        fontFamily: "var(--font-inter, system-ui, sans-serif)",
                        fontSize: "0.72rem", fontWeight: 500,
                        color: insight.accentColor,
                        display: "block", marginBottom: "1px",
                      }}>{insight.title}</span>
                      <span style={{
                        fontFamily: "var(--font-inter, system-ui, sans-serif)",
                        fontSize: "0.62rem", fontWeight: 300,
                        color: "rgba(229,224,216,0.50)", lineHeight: 1.35,
                      }}>{insight.desc}</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>




      </div>

      {/* ── CENTER GET STARTED HERO (Reveals prominently after Wellbeing dashboard fades out) ── */}
      <div
        ref={centerCtaRef}
        className="absolute z-[36] inset-0 flex flex-col items-center justify-center text-center px-4 sm:px-6 pointer-events-none select-none"
        style={{ opacity: 0 }}
      >
        <div className="max-w-2xl mx-auto flex flex-col items-center">
          <span
            style={{
              fontFamily:    "var(--font-inter, system-ui, sans-serif)",
              fontSize:      "clamp(0.65rem, 0.9vw, 0.78rem)",
              fontWeight:    600,
              letterSpacing: "0.26em",
              textTransform: "uppercase",
              color:         "#c9a96e",
              marginBottom:  "1rem",
              display:       "inline-flex",
              alignItems:    "center",
              gap:           "8px",
              padding:       "6px 16px",
              borderRadius:  "9999px",
              background:    "rgba(201,169,110,0.12)",
              border:        "1px solid rgba(201,169,110,0.25)",
            }}
          >
            ✦ Start Your Journey
          </span>

          <h2
            className="font-heading"
            style={{
              fontFamily:    "var(--font-heading), serif",
              fontSize:      "clamp(2.5rem, 5.2vw, 4.4rem)",
              fontWeight:    400,
              letterSpacing: "-0.02em",
              lineHeight:    1.12,
              color:         "#f0ede8",
              marginBottom:  "1.2rem",
              textShadow:    "0 15px 45px rgba(0,0,0,0.9)",
            }}
          >
            Ready to experience
            <br />
            <span style={{ color: "#e8d5b0", fontWeight: 400, fontStyle: "italic" }}>compassionate AI support?</span>
          </h2>

          <p
            className="font-body"
            style={{
              fontFamily:    "var(--font-body), sans-serif",
              fontSize:      "clamp(0.95rem, 1.4vw, 1.12rem)",
              fontWeight:    300,
              color:         "rgba(229,224,216,0.75)",
              maxWidth:      "540px",
              lineHeight:    1.65,
              marginBottom:  "2.2rem",
            }}
          >
            Begin your secure, private multimodal check-ins today. No pressure, completely at your own pace.
          </p>

          <Link
            href="/login"
            id="center-hero-get-started-btn"
            className="group relative inline-flex items-center gap-6 pl-12 pr-3.5 py-3.5 rounded-full text-white transition-all duration-500 transform hover:scale-[1.03] active:scale-[0.98] cursor-pointer pointer-events-auto overflow-hidden"
            style={{
              background: "linear-gradient(135deg, rgba(255, 255, 255, 0.12) 0%, rgba(255, 255, 255, 0.04) 100%)",
              backdropFilter: "blur(20px)",
              WebkitBackdropFilter: "blur(20px)",
              border: "1px solid rgba(255, 255, 255, 0.22)",
              boxShadow: "0 20px 40px -15px rgba(0,0,0,0.6), inset 0 1px 1px 0 rgba(255,255,255,0.4)",
              textDecoration: "none",
            }}
          >
            {/* Ambient hover light glow */}
            <div className="absolute inset-0 rounded-full opacity-0 group-hover:opacity-100 transition-opacity duration-500 pointer-events-none bg-gradient-to-r from-white/10 via-white/20 to-white/5" />

            <span
              className="text-base sm:text-lg font-medium tracking-wide text-white transition-colors duration-300"
              style={{ fontFamily: "var(--font-heading), serif" }}
            >
              Get Started with PsycheAI
            </span>

            {/* Icon Pill Badge */}
            <div className="relative z-10 w-11 h-11 rounded-full bg-white text-black flex items-center justify-center shadow-[0_4px_16px_rgba(0,0,0,0.25)] transition-all duration-300 group-hover:bg-white group-hover:scale-105 group-hover:shadow-[0_0_20px_rgba(255,255,255,0.4)]">
              <svg
                className="w-4 h-4 transition-transform duration-300 group-hover:translate-x-0.5"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2.5"
                strokeLinecap="round"
                strokeLinejoin="round"
              >
                <path d="M5 12h14" />
                <path d="M12 5l7 7-7 7" />
              </svg>
            </div>
          </Link>
        </div>
      </div>



      {/* ── VIDEO LAYER ── */}
      <div className="absolute inset-0 flex items-center justify-center">

        {/* ── FLOATING LARGE B&W VIDEOS ── */}
        {SELECTED_VIDEOS.map((video, idx) => {
          return (
            <div
              key={idx}
              ref={(el) => { floatingVideoRefs.current[idx] = el; }}
              className="absolute will-transform"
              style={{
                width:       "clamp(300px, 32vw, 520px)",
                aspectRatio: "16 / 9",
                borderRadius:"10px",
                overflow:    "hidden",
                boxShadow:   "0 15px 40px rgba(0,0,0,0.85), 0 0 0 1px rgba(255,255,255,0.04)",
                opacity:     0,
                zIndex:      video.z,
                backfaceVisibility: "hidden",
                transform:   "translateZ(0)",
              }}
            >
              <video
                src={video.src}
                autoPlay
                loop
                muted
                playsInline
                preload="metadata"
                style={{ width:"100%", height:"100%", objectFit:"cover", display:"block", pointerEvents:"none" }}
                suppressHydrationWarning
              />
            </div>
          );
        })}

        {/* ── CONTINUOUS COLOR COVER VIDEO ─────────────────────────
            coverWrapRef: emerges from center (scale 0.005 -> 1.00)
            and stays fullscreen as continuous backdrop
        ──────────────────────────────────────────────────────── */}
        <div
          ref={coverWrapRef}
          className="absolute z-[10] pointer-events-none will-transform"
          style={{
            width:        "100vw",
            height:       "100vh",
            overflow:     "hidden",
            borderRadius: "20px",
            boxShadow:    "0 24px 80px rgba(0,0,0,0.9), 0 0 0 1px rgba(255,255,255,0.04)",
            flexShrink:   0,
            left:         "50%",
            top:          "50%",
            transform:    "translate(-50%, -50%) scale(0.005)",
            opacity:      0,
            backfaceVisibility: "hidden",
          }}
        >
          <video
            ref={coverVideoElRef}
            src={COVER_VIDEO}
            autoPlay
            loop
            muted
            playsInline
            preload="auto"
            style={{
              width:      "100%",
              height:     "100%",
              objectFit:  "cover",
              display:    "block",
              willChange: "filter",
              transform:  "translateZ(0)",
            }}
            suppressHydrationWarning
          />

          {/* Cinematic dark overlay */}
          <div
            className="absolute inset-0 pointer-events-none z-10"
            style={{
              background:
                "radial-gradient(ellipse at center, rgba(0,0,0,0.2) 30%, rgba(8,8,8,0.75) 100%)",
            }}
          />
        </div>

      </div>
    </section>
  );
}
