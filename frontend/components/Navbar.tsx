"use client";

import Link from "next/link";
import { useState, useEffect } from "react";
import { accountApi, tokenManager } from "@/lib/api";

export default function Navbar() {
  const [scrolled, setScrolled] = useState(false);
  const [authChecked, setAuthChecked] = useState(false);
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [sessionExpired, setSessionExpired] = useState(false);

  useEffect(() => {
    const handleScroll = () => {
      setScrolled(window.scrollY > 20);
    };
    window.addEventListener("scroll", handleScroll);
    return () => window.removeEventListener("scroll", handleScroll);
  }, []);

  useEffect(() => {
    let active = true;
    let checking = false;
    const handleSessionExpired = () => {
      if (!active) return;
      setIsAuthenticated(false);
      setSessionExpired(true);
    };
    const checkSession = async () => {
      if (checking) return;
      checking = true;
      try {
        await accountApi.getProfile();
        if (active) setIsAuthenticated(true);
      } catch {
        if (active) setIsAuthenticated(false);
      } finally {
        checking = false;
        if (active) setAuthChecked(true);
      }
    };

    window.addEventListener("psycheai:session-expired", handleSessionExpired);
    void checkSession();
    const intervalId = window.setInterval(() => {
      if (tokenManager.hasAuthenticatedSession()) void checkSession();
    }, 60_000);
    const handleFocus = () => {
      if (tokenManager.hasAuthenticatedSession()) void checkSession();
    };
    window.addEventListener("focus", handleFocus);

    return () => {
      active = false;
      window.clearInterval(intervalId);
      window.removeEventListener("focus", handleFocus);
      window.removeEventListener("psycheai:session-expired", handleSessionExpired);
    };
  }, []);

  return (
    <>
    {sessionExpired && (
      <div role="alertdialog" aria-modal="true" aria-labelledby="session-expired-title" style={{ position: "fixed", inset: 0, zIndex: 100, display: "grid", placeItems: "center", padding: 24, background: "rgba(0,0,0,.72)" }}>
        <section style={{ width: "min(100%, 440px)", padding: 28, borderRadius: 16, background: "#151515", border: "1px solid rgba(201,169,110,.4)", color: "#f0ede8" }}>
          <h2 id="session-expired-title" style={{ margin: "0 0 12px", fontSize: 22 }}>Your login session has expired</h2>
          <p style={{ margin: "0 0 22px", color: "rgba(240,237,232,.75)" }}>Please sign in again to continue to your account.</p>
          <Link href="/login?sessionExpired=1" className="service-home-button" style={{ display: "inline-flex", textDecoration: "none" }}>Sign in again <span aria-hidden="true">→</span></Link>
        </section>
      </div>
    )}
    <header
      className="fixed top-0 left-0 right-0 z-50 flex items-center justify-between px-6 md:px-12 py-4 transition-all duration-500 pointer-events-auto"
      style={{
        background: scrolled
          ? "rgba(8, 8, 8, 0.75)"
          : "linear-gradient(to bottom, rgba(8, 8, 8, 0.6) 0%, transparent 100%)",
        backdropFilter: scrolled ? "blur(16px)" : "blur(4px)",
        WebkitBackdropFilter: scrolled ? "blur(16px)" : "blur(4px)",
        borderBottom: scrolled
          ? "1px solid rgba(255, 255, 255, 0.08)"
          : "1px solid transparent",
      }}
    >
      {/* Brand Logo */}
      <Link
        href="/"
        className="group flex items-center gap-3 transition-transform duration-300 hover:scale-[1.02]"
        style={{ textDecoration: "none" }}
      >
        <div
          className="w-9 h-9 rounded-xl flex items-center justify-center font-bold text-base text-[#080808] transition-all duration-300 shadow-[0_0_20px_rgba(201,169,110,0.3)] group-hover:shadow-[0_0_30px_rgba(201,169,110,0.6)]"
          style={{
            background: "linear-gradient(135deg, #c9a96e 0%, #ffffff 50%, #c9a96e 100%)",
          }}
        >
          <span style={{ fontFamily: "var(--font-brand), serif" }}>Ψ</span>
        </div>
        <span
          className="text-2xl font-normal tracking-wide text-[#f0ede8] transition-colors duration-300 group-hover:text-white"
          style={{ fontFamily: "var(--font-brand), cursive" }}
        >
          PsycheAI
        </span>
      </Link>

      {/* Nav Links */}
      <nav className="hidden md:flex items-center gap-8 px-6 py-2 rounded-full border border-white/10 bg-white/[0.03] backdrop-blur-md shadow-[0_4px_24px_rgba(0,0,0,0.4)]">
        <a
          href="#about"
          className="text-sm font-medium tracking-wide text-[#f0ede8]/70 hover:text-white transition-colors duration-200"
          style={{ fontFamily: "var(--font-body), sans-serif", textDecoration: "none" }}
        >
          About
        </a>
        <a
          href="#how-it-works"
          className="text-sm font-medium tracking-wide text-[#f0ede8]/70 hover:text-white transition-colors duration-200"
          style={{ fontFamily: "var(--font-body), sans-serif", textDecoration: "none" }}
        >
          How It Works
        </a>
        <a
          href="#features"
          className="text-sm font-medium tracking-wide text-[#f0ede8]/70 hover:text-white transition-colors duration-200"
          style={{ fontFamily: "var(--font-body), sans-serif", textDecoration: "none" }}
        >
          Features
        </a>
        {isAuthenticated && <Link
          href="/dashboard"
          className="text-sm font-medium tracking-wide text-[#c9a96e] hover:text-[#e8d5b0] transition-colors duration-200 flex items-center gap-1.5"
          style={{ fontFamily: "var(--font-body), sans-serif", textDecoration: "none" }}
        >
          <span className="w-1.5 h-1.5 rounded-full bg-[#7ef7c0] animate-pulse" />
          Dashboard
        </Link>}
      </nav>

      {/* Right CTA */}
      <div className="flex items-center gap-4">
        {authChecked && (isAuthenticated ? (
          <Link href="/account" className="hidden sm:inline-flex text-sm font-medium text-[#f0ede8]/75 transition-colors hover:text-white" style={{ fontFamily: "var(--font-body), sans-serif", textDecoration: "none" }}>Account</Link>
        ) : (
          <Link href="/login" className="hidden sm:inline-flex text-sm font-medium text-[#f0ede8]/75 transition-colors hover:text-white" style={{ fontFamily: "var(--font-body), sans-serif", textDecoration: "none" }}>Sign in</Link>
        ))}
        {authChecked && !isAuthenticated && <Link
          href="/signup"
          className="group relative inline-flex items-center gap-2.5 px-6 py-2.5 rounded-full text-sm font-medium text-white transition-all duration-300 transform hover:scale-[1.03] active:scale-[0.98] shadow-[0_4px_20px_rgba(0,0,0,0.4)] hover:shadow-[0_0_25px_rgba(201,169,110,0.35)]"
          style={{
            background: "linear-gradient(135deg, rgba(201,169,110,0.18) 0%, rgba(255,255,255,0.06) 100%)",
            border: "1px solid rgba(201, 169, 110, 0.4)",
            backdropFilter: "blur(12px)",
            textDecoration: "none",
            fontFamily: "var(--font-body), sans-serif",
          }}
        >
          <span>Get started</span>
          <svg
            className="w-3.5 h-3.5 transition-transform duration-300 group-hover:translate-x-0.5"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2.5"
          >
            <path d="M5 12h14M12 5l7 7-7 7" />
          </svg>
        </Link>}
      </div>
    </header>
    </>
  );
}
