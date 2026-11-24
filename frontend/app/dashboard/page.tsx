"use client";

import Link from "next/link";
import { useCallback, useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { ApiError, apiClient, authApi, sessionApi, tokenManager } from "@/lib/api";
import type { SessionResponse } from "@/lib/api";

type HistoryResponse = { details: string | null; completed: boolean };
type HistoryKind = "medical" | "family";
type ChatMessage = { role: "assistant" | "user"; text: string };

const historyPrompts: Record<HistoryKind, string> = {
  medical: "Before we begin, could you share any relevant medical conditions, medications, or allergies? You can also say “None known.”",
  family: "Thank you. Is there any family health history you’d like us to know about? You can also say “None known.”",
};

export default function DashboardPage() {
  const router = useRouter();
  const [isReady, setIsReady] = useState(false);
  const [isLoadingHistory, setIsLoadingHistory] = useState(true);
  const [isSigningOut, setIsSigningOut] = useState(false);
  const [isStartingSession, setIsStartingSession] = useState(false);
  const [isSavingHistory, setIsSavingHistory] = useState(false);
  const [isResumingSession, setIsResumingSession] = useState(false);
  const [deletingSessionId, setDeletingSessionId] = useState<number | null>(null);
  const [abandoningSessionId, setAbandoningSessionId] = useState<number | null>(null);
  const [medicalHistory, setMedicalHistory] = useState<HistoryResponse | null>(null);
  const [familyHistory, setFamilyHistory] = useState<HistoryResponse | null>(null);
  const [medicalDetails, setMedicalDetails] = useState("");
  const [familyDetails, setFamilyDetails] = useState("");
  const [activeSession, setActiveSession] = useState<SessionResponse | null>(null);
  const [sessions, setSessions] = useState<SessionResponse[]>([]);
  const [historyStep, setHistoryStep] = useState<HistoryKind | null>(null);
  const [historyAnswer, setHistoryAnswer] = useState("");
  const [historyMessages, setHistoryMessages] = useState<ChatMessage[]>([]);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");

  const loadHistory = useCallback(async () => {
    const [medical, family] = await Promise.all([
      apiClient.get<HistoryResponse>("/user/history/medical"),
      apiClient.get<HistoryResponse>("/user/history/family"),
    ]);
    setMedicalHistory(medical);
    setFamilyHistory(family);
    setMedicalDetails(medical.details ?? "");
    setFamilyDetails(family.details ?? "");
  }, []);

  const loadSessions = useCallback(async () => {
    const [history, draft] = await Promise.all([
      sessionApi.list(),
      apiClient.get<SessionResponse | undefined>("/session/draft"),
    ]);
    setSessions(history);
    setActiveSession(draft ?? null);
  }, []);

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
        await Promise.all([loadHistory(), loadSessions()]);
      } catch (cause) {
        if (!isCurrent) return;
        if (cause instanceof ApiError && [401, 403].includes(cause.status)) {
          router.replace(cause instanceof ApiError && cause.status === 401 && tokenManager.hasAuthenticatedSession() ? "/login?sessionExpired=1" : "/login");
          return;
        }
        setError(cause instanceof Error ? cause.message : "Could not restore your dashboard.");
      } finally {
        if (isCurrent) setIsLoadingHistory(false);
      }
    }

    void restoreSession();
    return () => { isCurrent = false; };
  }, [loadHistory, loadSessions, router]);

  const historiesComplete = Boolean(medicalHistory?.completed && familyHistory?.completed);

  function beginHistoryChat() {
    const firstMissing: HistoryKind | null = !medicalHistory?.completed
      ? "medical"
      : !familyHistory?.completed ? "family" : null;
    if (!firstMissing) {
      void startSession();
      return;
    }
    setError("");
    setNotice("");
    setHistoryStep(firstMissing);
    setHistoryAnswer(firstMissing === "medical" ? medicalDetails : familyDetails);
    setHistoryMessages([{ role: "assistant", text: historyPrompts[firstMissing] }]);
  }

  async function saveHistoryAnswer(event: { preventDefault(): void }) {
    event.preventDefault();
    if (!historyStep || !historyAnswer.trim()) return;
    setIsSavingHistory(true);
    setError("");
    setNotice("");
    try {
      const saved = await apiClient.put<HistoryResponse>(`/user/history/${historyStep}`, {
        details: historyAnswer.trim(),
      });
      if (historyStep === "medical") setMedicalHistory(saved);
      else setFamilyHistory(saved);

      const nextKind: HistoryKind | null = historyStep === "medical" && !familyHistory?.completed
        ? "family"
        : historyStep === "family" && !medicalHistory?.completed
          ? "medical"
          : null;
      const messages = [...historyMessages, { role: "user" as const, text: historyAnswer.trim() }];
      if (nextKind) {
        messages.push({ role: "assistant", text: historyPrompts[nextKind] });
        setHistoryStep(nextKind);
        setHistoryAnswer(nextKind === "medical" ? medicalDetails : familyDetails);
      } else {
        messages.push({ role: "assistant", text: "Thanks for sharing. Your history is saved and you’re ready to begin." });
        setHistoryStep(null);
        setHistoryAnswer("");
      }
      setHistoryMessages(messages);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : `Could not save your ${historyStep} history.`);
    } finally {
      setIsSavingHistory(false);
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
      tokenManager.clearSession();
      router.replace("/login");
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Could not sign out. Please try again.");
    } finally {
      setIsSigningOut(false);
    }
  }

  async function deleteSession(session: SessionResponse) {
    const confirmed = window.confirm("Permanently delete this session and its complete conversation, assessment results, and audio? This cannot be undone.");
    if (!confirmed) return;
    setDeletingSessionId(session.id);
    setError("");
    setNotice("");
    try {
      await sessionApi.delete(session.id);
      setSessions((current) => current.filter((item) => item.id !== session.id));
      setActiveSession((current) => current?.id === session.id ? null : current);
      setNotice("Session and its conversation were deleted.");
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Could not delete this session.");
    } finally {
      setDeletingSessionId(null);
    }
  }

  async function abandonSession(session: SessionResponse) {
    const confirmed = window.confirm("Abandon this assessment? You can still view its conversation, but you won’t be able to continue it.");
    if (!confirmed) return;
    setAbandoningSessionId(session.id);
    setError("");
    setNotice("");
    try {
      const abandoned = await sessionApi.abandon(session.id);
      setSessions((current) => current.map((item) => item.id === session.id ? abandoned : item));
      setActiveSession((current) => current?.id === session.id ? null : current);
      setNotice("Session abandoned. Its conversation is still available in your history.");
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Could not abandon this session.");
    } finally {
      setAbandoningSessionId(null);
    }
  }

  if (!isReady) {
    return (
      <main className="min-h-screen flex items-center justify-center px-6" style={{ backgroundColor: "var(--color-bg)" }}>
        <div className="glass-panel max-w-lg rounded-2xl p-8 text-center">
          {error ? <><p role="alert" className="mb-5 text-sm text-red-200">{error}</p><Link href="/login" className="text-sm hover:underline" style={{ color: "var(--color-accent)" }}>Return to sign in</Link></> : <p className="text-sm" style={{ color: "var(--color-muted)" }}>Restoring your secure session…</p>}
        </div>
      </main>
    );
  }

  return (
    <main className="service-page dashboard-page">
      <video className="auth-background-video" autoPlay muted loop playsInline aria-hidden="true"><source src="/videos/beauty-fx30.mp4" type="video/mp4" /></video>
      <div className="auth-background-shade" aria-hidden="true" />
      <header className="service-topbar">
        <Link href="/" className="auth-brand" aria-label="PsycheAI home"><span className="auth-brand-mark" aria-hidden="true">Ψ</span><span>PsycheAI</span></Link>
        <div className="service-topbar-actions">
          <Link href="/" className="app-back-button">← Home</Link>
          <Link href="/account" className="auth-home-link">Account</Link>
          <button type="button" onClick={handleSignOut} disabled={isSigningOut} className="service-signout">{isSigningOut ? "Signing out…" : "Sign out"}</button>
        </div>
      </header>

      <section className="service-card history-card dashboard-content">
        <div className="service-status"><span className="service-status-dot" />You’re signed in</div>
        <p className="service-eyebrow">PsycheAI workspace</p>
        <h1>Your sessions</h1>
        <p className="service-description">Start a private assessment, or return to any session to see the full conversation.</p>
        {error && <p role="alert" className="service-error">{error}</p>}
        {notice && <p role="status" className="history-notice">{notice}</p>}

        {activeSession && (
          <div className="history-session">
            <p className="history-session-ready">You have an unfinished session ({activeSession.status.toLowerCase().replace("_", " ")}).</p>
            <button type="button" className="service-home-button" onClick={() => void resumeSession()} disabled={isResumingSession}>
              {isResumingSession ? "Opening session…" : "Continue assessment"}<span aria-hidden="true"> →</span>
            </button>
          </div>
        )}

        {!activeSession && !historyStep && historyMessages.length === 0 && (
          <div className="history-session">
            {!historiesComplete && <p className="history-help">We’ll ask for any missing medical or family history after you choose to start. Your answers are private to your account.</p>}
            <button type="button" className="service-home-button" onClick={beginHistoryChat} disabled={isLoadingHistory || isStartingSession}>
              {isLoadingHistory ? "Loading your profile…" : isStartingSession ? "Creating session…" : "Start assessment"}<span aria-hidden="true"> →</span>
            </button>
          </div>
        )}

        {(historyStep || historyMessages.length > 0) && (
          <section className="history-chat" aria-label="Medical and family history conversation">
            <div className="history-chat-messages" aria-live="polite">
              {historyMessages.map((message, index) => (
                <div className={`chat-message ${message.role === "user" ? "chat-message-user" : "chat-message-assistant"}`} key={`${message.role}-${index}`}>
                  <span className="chat-message-label">{message.role === "user" ? "You" : "PsycheAI"}</span>
                  <p>{message.text}</p>
                </div>
              ))}
            </div>
            {historyStep ? (
              <form className="history-chat-form" onSubmit={(event) => void saveHistoryAnswer(event)}>
                <label htmlFor="history-answer" className="sr-only">Your {historyStep} history</label>
                <textarea id="history-answer" className="history-textarea" value={historyAnswer} onChange={(event) => setHistoryAnswer(event.target.value)} maxLength={10000} rows={3} required placeholder="Type your answer…" />
                <button type="submit" className="service-home-button" disabled={isSavingHistory || !historyAnswer.trim()}>{isSavingHistory ? "Saving…" : "Send"}<span aria-hidden="true"> ↑</span></button>
              </form>
            ) : (
              <button type="button" className="service-home-button" onClick={() => void startSession()} disabled={isStartingSession || !historiesComplete}>
                {isStartingSession ? "Creating session…" : "Start assessment"}<span aria-hidden="true"> →</span>
              </button>
            )}
          </section>
        )}

        <section className="session-list" aria-labelledby="session-list-heading">
          <h2 id="session-list-heading">Session history</h2>
          {isLoadingHistory ? <p className="history-loading">Loading your sessions…</p> : sessions.length === 0 ? (
            <p className="history-help">Your completed and in-progress sessions will appear here.</p>
          ) : (
            <div className="session-list-items">
              {sessions.map((item) => (
                <div className="session-list-item" key={item.id}>
                  <Link className="session-list-link" href={`/assessment/${item.id}`}>
                    <span><strong>Assessment session</strong><small>{new Date(item.createdAt).toLocaleString()} · {item.currentTurnNumber} {item.currentTurnNumber === 1 ? "response" : "responses"}</small></span>
                    <span className="session-list-status">{item.status.toLowerCase().replace("_", " ")} <span aria-hidden="true">→</span></span>
                  </Link>
                  <details className="session-actions-menu">
                    <summary aria-label={`Actions for session ${item.id}`} title="Session actions">⋯</summary>
                    <div className="session-actions-popover">
                      {["DRAFT", "IN_PROGRESS", "PAUSED"].includes(item.status) && (
                        <button type="button" className="session-abandon-action" disabled={abandoningSessionId === item.id || deletingSessionId === item.id} onClick={() => void abandonSession(item)}>
                          {abandoningSessionId === item.id ? "Abandoning…" : "Abandon session"}
                        </button>
                      )}
                      <button type="button" className="session-delete-action" disabled={deletingSessionId === item.id || abandoningSessionId === item.id} onClick={() => void deleteSession(item)}>
                        {deletingSessionId === item.id ? "Deleting…" : "Delete session"}
                      </button>
                    </div>
                  </details>
                </div>
              ))}
            </div>
          )}
        </section>
      </section>
    </main>
  );
}
