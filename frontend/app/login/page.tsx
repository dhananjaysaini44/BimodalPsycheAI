"use client";

import Link from "next/link";
import { useState } from "react";
import { Suspense } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { authApi, tokenManager } from "@/lib/api";
import AuthLayout from "@/components/AuthLayout";

function LoginFormContent() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const [email, setEmail] = useState(searchParams.get("email") ?? "");
  const [password, setPassword] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState("");
  const isRegistrationComplete = searchParams.get("registered") === "1";
  const isSessionExpired = searchParams.get("sessionExpired") === "1";

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");
    setIsLoading(true);

    try {
      const response = await authApi.login(email, password);
      tokenManager.setAccessToken(response.token);
      router.push("/dashboard");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Login failed. Please check your credentials.");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <AuthLayout
      eyebrow="Welcome back"
      headline={<>A little space<br /><em>for you.</em></>}
      description="Sign in to return to your PsycheAI account and continue when you’re ready."
    >
        <section className="auth-card">
          {/* Logo/Brand */}
          <div className="auth-card-heading">
            <span className="auth-card-kicker">YOUR ACCOUNT</span>
            <h1 
              className="text-4xl mb-3"
              style={{ 
                fontFamily: "var(--font-heading)", 
                color: "var(--color-text)",
                fontWeight: 400,
                letterSpacing: "-0.02em"
              }}
            >
              Welcome Back
            </h1>
            <p style={{ color: "var(--color-muted)", fontFamily: "var(--font-body)" }}>
              Sign in securely to continue.
            </p>
          </div>

          {isRegistrationComplete && (
            <div className="auth-success" role="status">
              Your email is verified and account is ready. Sign in to continue.
            </div>
          )}

          {isSessionExpired && (
            <div className="auth-error" role="alert">
              Your login session has expired. Please sign in again to continue.
            </div>
          )}

          {/* Error Message */}
          {error && (
            <div className="mb-6 p-4 rounded-lg border text-sm" style={{ 
              background: "rgba(220, 38, 38, 0.15)", 
              borderColor: "rgba(220, 38, 38, 0.3)",
              color: "#fca5a5"
            }} role="alert">
              {error}
            </div>
          )}

          {/* Form */}
          <form onSubmit={handleSubmit} className="auth-form">
            <div>
              <label htmlFor="email" className="block text-sm mb-2" style={{ color: "var(--color-muted)", fontFamily: "var(--font-body)" }}>
                Email Address
              </label>
              <input
                id="email"
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                required
                autoComplete="email"
                autoFocus
                className="glass-input w-full px-4 py-3 rounded-lg"
                placeholder="you@example.com"
              />
            </div>

            <div>
              <label htmlFor="password" className="block text-sm mb-2" style={{ color: "var(--color-muted)", fontFamily: "var(--font-body)" }}>
                Password
              </label>
              <input
                id="password"
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                required
                autoComplete="current-password"
                className="glass-input w-full px-4 py-3 rounded-lg"
                placeholder="••••••••"
              />
            </div>

            <button
              type="submit"
              disabled={isLoading || !email.trim() || !password}
              className="auth-submit"
            >
              <span>{isLoading ? "Signing in..." : "Sign in"}</span>
              <span aria-hidden="true">→</span>
            </button>
          </form>

          {/* Sign Up Link */}
          <p className="auth-switch">
            Don&apos;t have an account?{" "}
            <Link 
              href="/signup" 
              className="font-medium hover:underline transition-colors"
              style={{ color: "var(--color-accent)" }}
            >
              Sign up
            </Link>
          </p>
          <p className="auth-privacy-note"><span aria-hidden="true">◇</span> Your wellbeing journey is personal. Your account stays yours.</p>
        </section>
    </AuthLayout>
  );
}

export default function LoginPage() {
  return (
    <Suspense
      fallback={
        <AuthLayout
          eyebrow="Welcome back"
          headline={<>A little space<br /><em>for you.</em></>}
          description="Sign in to return to your PsycheAI account and continue when you’re ready."
        >
          <section className="auth-card">
            <p style={{ color: "var(--color-muted)" }}>Preparing secure sign in…</p>
          </section>
        </AuthLayout>
      }
    >
      <LoginFormContent />
    </Suspense>
  );
}
