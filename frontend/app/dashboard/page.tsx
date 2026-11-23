"use client";

import Link from "next/link";
import { useCallback, useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { ApiError, apiClient, authApi, sessionApi, tokenManager } from "@/lib/api";
import type { SessionResponse } from "@/lib/api";

type HistoryResponse = {
  details: string | null;
  completed: boolean;
};

export default function DashboardPage() {
  const router = useRouter();
  const [isReady, setIsReady] = useState(false);
  const [isLoadingHistory, setIsLoadingHistory] = useState(true);
  const [isSigningOut, setIsSigningOut] = useState(false);
  const [isSavingMedical, setIsSavingMedical] = useState(false);
  const [isSavingFamily, setIsSavingFamily] = useState(false);
  const [isStartingSession, setIsStartingSession] = useState(false);
  const [medicalHistory, setMedicalHistory] = useState<HistoryResponse | null>(null);
  const [familyHistory, setFamilyHistory] = useState<HistoryResponse | null>(null);
  const [medicalDetails, setMedicalDetails] = useState("");
  const [familyDetails, setFamilyDetails] = useState("");
  const [activeSession, setActiveSession] = useState<SessionResponse | null>(null);
  const [isResumingSession, setIsResumingSession] = useState(false);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");

  const loadHistory = useCallback(async () => {
    setIsLoadingHistory(true);
    setError("");
    try {
      const [medical, family] = await Promise.all([
        apiClient.get<HistoryResponse>("/user/history/medical"),
        apiClient.get<HistoryResponse>("/user/history/family"),
      ]);
      setMedicalHistory(medical);
      setFamilyHistory(family);
      setMedicalDetails(medical.details ?? "");
      setFamilyDetails(family.details ?? "");
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Could not load your history forms.");
      if (cause instanceof ApiError && cause.status === 401) {
        router.replace("/login");
      }
    } finally {
      setIsLoadingHistory(false);
    }
  }, [router]);

  useEffect(() => {
    let isCurrent = true;

    async function restoreSession() {
      try {
        if (!tokenManager.getAccessToken()) {
          const response = await authApi.refreshToken();
          tokenManager.setAccessToken(response.token);
        }
        if (!isCurrent) return;
        setIsReady(true);
        await loadHistory();
        const draft = await apiClient.get<SessionResponse | undefined>("/session/draft");
        if (isCurrent && draft) setActiveSession(draft);
      } catch (cause) {
        if (!isCurrent) return;
        if (cause instanceof ApiError && [401, 403].includes(cause.status)) {
          router.replace("/login");
          return;
        }
        setError(cause instanceof Error ? cause.message : "Could not restore your session.");
        setIsLoadingHistory(false);
      }
    }

    void restoreSession();
    return () => {
      isCurrent = false;
    };
  }, [loadHistory, router]);

  async function saveHistory(kind: "medical" | "family") {
    const isMedical = kind === "medical";
    const details = isMedical ? medicalDetails : familyDetails;
    const setSaving = isMedical ? setIsSavingMedical : setIsSavingFamily;
    setSaving(true);
    setError("");
    setNotice("");
    try {
      const saved = await apiClient.put<HistoryResponse>(`/user/history/${kind}`, { details });
      if (isMedical) {
        setMedicalHistory(saved);
        setMedicalDetails(saved.details ?? "");
      } else {
        setFamilyHistory(saved);
        setFamilyDetails(saved.details ?? "");
      }
      setNotice(`${isMedical ? "Medical" : "Family"} history saved.`);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : `Could not save ${kind} history.`);
    } finally {
      setSaving(false);
    }
  }

  async function startSession() {
    setIsStartingSession(true);
    setError("");
    setNotice("");
    try {
      const session = await sessionApi.start();
      router.push(`/assessment/${session.id}`);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Could not start your session.");
    } finally {
      setIsStartingSession(false);
    }
  }

  async function resumeSession() {
    if (!activeSession) return;
    setIsResumingSession(true);
    setError("");
    try {
      const session = await sessionApi.resume(activeSession.id);
      router.push(`/assessment/${session.id}`);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Could not resume your session.");
    } finally {
      setIsResumingSession(false);
    }
  }

  async function handleSignOut() {
    setIsSigningOut(true);
    setError("");
    try {
      await authApi.logout();
      tokenManager.clearTokens();
      router.replace("/login");
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Could not sign out. Please try again.");
    } finally {
      setIsSigningOut(false);
    }
  }

  if (!isReady) {
    return (
      <main className="min-h-screen flex items-center justify-center px-6" style={{ backgroundColor: "var(--color-bg)" }}>
        <div className="glass-panel max-w-lg rounded-2xl p-8 text-center">
          {error ? (
            <>
              <p role="alert" className="mb-5 text-sm text-red-200">{error}</p>
              <Link href="/login" className="text-sm hover:underline" style={{ color: "var(--color-accent)" }}>
                Return to sign in
              </Link>
            </>
          ) : (
            <p className="text-sm" style={{ color: "var(--color-muted)" }}>Restoring your secure session…</p>
          )}
        </div>
      </main>
    );
  }

  const historiesComplete = Boolean(medicalHistory?.completed && familyHistory?.completed);

  return (
    <main className="service-page">
      <video className="auth-background-video" autoPlay muted loop playsInline aria-hidden="true">
        <source src="/videos/beauty-fx30.mp4" type="video/mp4" />
      </video>
      <div className="auth-background-shade" aria-hidden="true" />
      <header className="service-topbar">
        <Link href="/" className="auth-brand" aria-label="PsycheAI home">
          <span className="auth-brand-mark" aria-hidden="true">Ψ</span>
          <span>PsycheAI</span>
        </Link>
        <div className="service-topbar-actions">
          <Link href="/account" className="auth-home-link">Account</Link>
          <Link href="/" className="auth-home-link">Home</Link>
          <button
            type="button"
            onClick={handleSignOut}
            disabled={isSigningOut}
            className="service-signout"
          >
            {isSigningOut ? "Signing out…" : "Sign out"}
          </button>
        </div>
      </header>

      <section className="service-card history-card">
        <div className="service-status">
          <span className="service-status-dot" />
          You’re signed in
        </div>
        <p className="service-eyebrow">PsycheAI workspace</p>
        <h1>Before your first session</h1>
        <p className="service-description">
          To start an assessment session, please complete both your medical history and family history. This helps us prepare your session appropriately. Only you can access or update these details.
        </p>

        {error && <p role="alert" className="service-error">{error}</p>}
        {notice && <p role="status" className="history-notice">{notice}</p>}

        {isLoadingHistory ? (
          <p className="history-loading">Loading your history forms…</p>
        ) : (
          <>
            <div className="history-form">
              <label htmlFor="medical-history" className="history-label">
                Medical history
                <span className="history-completion">
                  {medicalHistory?.completed ? "Completed" : "Required"}
                </span>
              </label>
              <p className="history-help">
                Share relevant medical conditions, medications, allergies, or write “None known.”
              </p>
              <textarea
                id="medical-history"
                className="history-textarea"
                value={medicalDetails}
                onChange={(event) => setMedicalDetails(event.target.value)}
                maxLength={10000}
                rows={4}
                placeholder="Enter relevant medical history, or write “None known.”"
              />
              <button
                type="button"
                className="history-save-button"
                onClick={() => void saveHistory("medical")}
                disabled={isSavingMedical || !medicalDetails.trim()}
              >
                {isSavingMedical ? "Saving…" : "Save medical history"}
              </button>
            </div>

            <div className="history-form">
              <label htmlFor="family-history" className="history-label">
                Family history
                <span className="history-completion">
                  {familyHistory?.completed ? "Completed" : "Required"}
                </span>
              </label>
              <p className="history-help">
                Share relevant family health history, or write “None known.”
              </p>
              <textarea
                id="family-history"
                className="history-textarea"
                value={familyDetails}
                onChange={(event) => setFamilyDetails(event.target.value)}
                maxLength={10000}
                rows={4}
                placeholder="Enter relevant family history, or write “None known.”"
              />
              <button
                type="button"
                className="history-save-button"
                onClick={() => void saveHistory("family")}
                disabled={isSavingFamily || !familyDetails.trim()}
              >
                {isSavingFamily ? "Saving…" : "Save family history"}
              </button>
            </div>

            <div className="history-session">
              {!historiesComplete && (
                <p className="history-block-message" role="status">
                  A session can’t be created yet. Complete and save both history forms above to continue.
                </p>
              )}
              {activeSession ? (
                <div>
                  <p className="history-session-ready" role="status">
                    You have an unfinished session ({activeSession.status.toLowerCase().replace("_", " ")}).
                  </p>
                  <button
                    type="button"
                    className="service-home-button"
                    onClick={() => void resumeSession()}
                    disabled={isResumingSession}
                  >
                    {isResumingSession ? "Opening session…" : "Continue assessment"}
                    {!isResumingSession && <span aria-hidden="true"> →</span>}
                  </button>
                </div>
              ) : (
                <button
                  type="button"
                  className="service-home-button"
                  onClick={() => void startSession()}
                  disabled={!historiesComplete || isStartingSession || isLoadingHistory}
                >
                  {isStartingSession ? "Creating session…" : "Start assessment session"}
                  {!isStartingSession && <span aria-hidden="true"> →</span>}
                </button>
              )}
            </div>
          </>
        )}
      </section>
    </main>
  );
}
