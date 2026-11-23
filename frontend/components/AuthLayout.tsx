import Link from "next/link";
import type { ReactNode } from "react";

type AuthLayoutProps = {
  children: ReactNode;
  eyebrow: string;
  headline: ReactNode;
  description: string;
};

export default function AuthLayout({
  children,
  eyebrow,
  headline,
  description,
}: AuthLayoutProps) {
  return (
    <main className="auth-page">
      <video className="auth-background-video" autoPlay muted loop playsInline aria-hidden="true">
        <source src="/videos/beauty-fx30.mp4" type="video/mp4" />
      </video>
      <div className="auth-background-shade" aria-hidden="true" />

      <header className="auth-topbar">
        <Link href="/" className="auth-brand" aria-label="PsycheAI home">
          <span className="auth-brand-mark" aria-hidden="true">Ψ</span>
          <span>PsycheAI</span>
        </Link>
        <Link href="/" className="auth-home-link">Back to home <span aria-hidden="true">↗</span></Link>
      </header>

      <div className="auth-layout">
        <aside className="auth-intro">
          <span className="auth-eyebrow"><span aria-hidden="true">✦</span> {eyebrow}</span>
          <h2>{headline}</h2>
          <p>{description}</p>
          <div className="auth-promise">
            <span className="auth-promise-icon" aria-hidden="true">✳</span>
            <span>A thoughtful space for your wellbeing journey.</span>
          </div>
        </aside>
        {children}
      </div>
    </main>
  );
}
