"use client";

import Link from "next/link";
import { useParams, useRouter } from "next/navigation";
import { useCallback, useEffect, useState } from "react";
import { ApiError, authApi, sessionApi, tokenManager } from "@/lib/api";
import type {
  AssessmentResponse,
  CrisisResource,
  NextQuestionResponse,
  SessionResponse,
  TurnResponse,
} from "@/lib/api";

export default function AssessmentPage() {
  const params = useParams<{ sessionId: string }>();
  const router = useRouter();
  const sessionId = Number(params.sessionId);
  const [session, setSession] = useState<SessionResponse | null>(null);
  const [question, setQuestion] = useState<NextQuestionResponse["question"]>(null);
  const [responseText, setResponseText] = useState("");
  const [audioFile, setAudioFile] = useState<File | null>(null);
  const [crisis, setCrisis] = useState<{ guidance: string; resources: CrisisResource[] } | null>(null);
  const [result, setResult] = useState<AssessmentResponse | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [isGenerating, setIsGenerating] = useState(false);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");

  const loadNextQuestion = useCallback(async () => {
    const next = await sessionApi.nextQuestion(sessionId);
    setQuestion(next.available ? next.question : null);
    if (next.conversationComplete) setNotice("You’ve answered every theme. You can finish and request your assessment.");
    return next;
  }, [sessionId]);

  const loadAssessment = useCallback(async () => {
    try {
      setResult(await sessionApi.getAssessment(sessionId));
    } catch (cause) {
      if (!(cause instanceof ApiError && cause.status === 404)) throw cause;
      setResult(null);
    }
  }, [sessionId]);

  useEffect(() => {
    let active = true;

    async function initialize() {
      if (!Number.isSafeInteger(sessionId) || sessionId <= 0) {
        setError("This assessment link is invalid.");
        setIsLoading(false);
        return;
      }

      try {
        if (!tokenManager.getAccessToken()) {
          const refreshed = await authApi.refreshToken();
          tokenManager.setAccessToken(refreshed.token);
        }
        const currentSession = await sessionApi.get(sessionId);
        if (!active) return;
        setSession(currentSession);

        if (currentSession.status === "IN_PROGRESS") {
          await loadNextQuestion();
        } else if (currentSession.status === "COMPLETED") {
          await loadAssessment();
        }
      } catch (cause) {
        if (!active) return;
        if (cause instanceof ApiError && cause.status === 401) {
          router.replace("/login");
          return;
        }
        setError(cause instanceof Error ? cause.message : "Could not load this assessment session.");
      } finally {
        if (active) setIsLoading(false);
      }
    }

    void initialize();
    return () => {
      active = false;
    };
  }, [loadAssessment, loadNextQuestion, router, sessionId]);

  async function submitText(event: { preventDefault(): void }) {
    event.preventDefault();
    if (!question || !responseText.trim()) return;
    setIsSubmitting(true);
    setError("");
    setNotice("");
    try {
      const submitted: TurnResponse = await sessionApi.submitText(sessionId, question.id, responseText.trim());
      setResponseText("");
      setSession((current) => current ? { ...current, currentTurnNumber: submitted.turnNumber } : current);
      if (submitted.crisisDetected) {
        setCrisis({
          guidance: submitted.crisisGuidance ?? "If you may be in immediate danger, contact local emergency services or a trusted person now.",
          resources: submitted.crisisResources ?? [],
        });
        setSession((current) => current ? { ...current, status: "PAUSED" } : current);
        setQuestion(null);
        return;
      }
      const next = await loadNextQuestion();
      if (!next.available && !next.conversationComplete) {
        setNotice("This session is paused. Continue when you’re ready.");
      }
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Your response could not be saved. Please try again.");
    } finally {
      setIsSubmitting(false);
    }
  }

  async function submitAudio(event: { preventDefault(): void }) {
    event.preventDefault();
    if (!question || !audioFile) return;
    setIsSubmitting(true);
    setError("");
    setNotice("");
    try {
      const uploaded = await sessionApi.submitAudio(sessionId, question.id, audioFile);
      setAudioFile(null);
      setSession((current) => current ? { ...current, currentTurnNumber: uploaded.turnNumber } : current);
      setNotice(`Audio saved securely. Analysis status: ${uploaded.analysisStatus}.`);
      await loadNextQuestion();
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Your audio could not be saved. Please try again.");
    } finally {
      setIsSubmitting(false);
    }
  }

  async function continuePausedSession() {
    setIsSubmitting(true);
    setError("");
    try {
      const resumed = await sessionApi.resume(sessionId);
      setSession(resumed);
      setCrisis(null);
      setNotice("");
      await loadNextQuestion();
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Could not continue this session.");
    } finally {
      setIsSubmitting(false);
    }
  }

  async function generateAssessment() {
    setIsGenerating(true);
    setError("");
    setNotice("");
    try {
      const completed = session?.status === "COMPLETED" ? session : await sessionApi.complete(sessionId);
      setSession(completed);
      setResult(await sessionApi.generateAssessment(sessionId));
    } catch (cause) {
      if (session?.status !== "COMPLETED") {
        try {
          const currentSession = await sessionApi.get(sessionId);
          setSession(currentSession);
        } catch {
          // Keep the original completion or analysis error visible.
        }
      }
      setError(cause instanceof Error ? cause.message : "Assessment analysis is not available yet.");
    } finally {
      setIsGenerating(false);
    }
  }

  if (isLoading) {
    return (
      <main className="service-page assessment-page">
        <section className="service-card assessment-card" aria-live="polite">Loading your secure session…</section>
      </main>
    );
  }

  return (
    <main className="service-page assessment-page">
      <video className="auth-background-video" autoPlay muted loop playsInline aria-hidden="true">
        <source src="/videos/beauty-fx30.mp4" type="video/mp4" />
      </video>
      <div className="auth-background-shade" aria-hidden="true" />
      <header className="service-topbar">
        <Link href="/" className="auth-brand" aria-label="PsycheAI home">
          <span className="auth-brand-mark" aria-hidden="true">Ψ</span><span>PsycheAI</span>
        </Link>
        <Link href="/dashboard" className="auth-home-link">Dashboard</Link>
      </header>

      <section className="service-card assessment-card">
        <div className="service-status"><span className="service-status-dot" />Private assessment session</div>
        <p className="service-eyebrow">PsycheAI workspace · Session {sessionId}</p>
        <h1>{result ? "Your assessment" : session?.status === "COMPLETED" ? "Session complete" : "A space to reflect"}</h1>
        <p className="service-description">
          Your responses are private to your account. This experience is not a diagnosis or a substitute for professional care.
        </p>

        {error && <p className="service-error" role="alert">{error}</p>}
        {notice && <p className="history-notice" role="status">{notice}</p>}

        {crisis && (
          <aside className="assessment-support" role="alert">
            <h2>Support is available</h2>
            <p>{crisis.guidance}</p>
            {crisis.resources.length > 0 && (
              <ul>
                {crisis.resources.map((resource, index) => (
                  <li key={`${resource.name}-${index}`}>
                    <strong>{resource.name}</strong>{resource.phone ? ` · ${resource.phone}` : ""}
                    {resource.url && <> · <a href={resource.url} target="_blank" rel="noreferrer">Visit resource</a></>}
                    {resource.description && <span> — {resource.description}</span>}
                  </li>
                ))}
              </ul>
            )}
            {session?.status === "PAUSED" && (
              <button className="service-home-button" type="button" onClick={() => void continuePausedSession()} disabled={isSubmitting}>
                {isSubmitting ? "Resuming…" : "Continue when you’re ready"}
              </button>
            )}
          </aside>
        )}

        {result && (
          <section className="assessment-result" aria-labelledby="result-heading">
            <h2 id="result-heading">Model-generated estimate</h2>
            <p className="assessment-severity">{result.severity}</p>
            <p>Score: {result.score} · Probability: {Math.round(result.probability * 100)}%</p>
            {result.explanation && <p>{result.explanation}</p>}
            <p className="assessment-disclaimer">This is an experimental model output, not a medical diagnosis.</p>
          </section>
        )}

        {!result && session?.status === "IN_PROGRESS" && question && !crisis && (
          <>
            <div className="assessment-question">
              <p className="assessment-kicker">{question.themeName} · {question.variantType.toLowerCase()} question</p>
              <h2>{question.questionText}</h2>
              <p className="history-help">Response {session.currentTurnNumber + 1}. Choose text or upload an audio response.</p>
            </div>
            <form className="assessment-form" onSubmit={(event) => void submitText(event)}>
              <label className="history-label" htmlFor="response-text">Write a response</label>
              <textarea
                id="response-text"
                className="history-textarea"
                value={responseText}
                onChange={(event) => setResponseText(event.target.value)}
                maxLength={10000}
                rows={5}
                required
                placeholder="Share only what you feel comfortable sharing."
              />
              <button type="submit" className="service-home-button" disabled={isSubmitting || !responseText.trim()}>
                {isSubmitting ? "Saving response…" : "Send text response"}
              </button>
            </form>
            <form className="assessment-form" onSubmit={(event) => void submitAudio(event)}>
              <label className="history-label" htmlFor="audio-response">Or upload an audio response</label>
              <p className="history-help">WAV, WebM, MP3, MP4, OGG, or AAC, up to 25 MB.</p>
              <input
                id="audio-response"
                className="assessment-file-input"
                type="file"
                accept="audio/wav,audio/webm,audio/mpeg,audio/mp4,audio/ogg,audio/aac,.wav,.webm,.mp3,.mp4,.ogg,.aac"
                onChange={(event) => setAudioFile(event.target.files?.[0] ?? null)}
              />
              <button type="submit" className="service-home-button" disabled={isSubmitting || !audioFile}>
                {isSubmitting ? "Uploading audio…" : "Upload audio response"}
              </button>
            </form>
          </>
        )}

        {!result && session?.status === "IN_PROGRESS" && !question && !crisis && (
          <div className="assessment-finish">
            <p>All question themes have been answered. Finish the session to request its assessment.</p>
            <button type="button" className="service-home-button" onClick={() => void generateAssessment()} disabled={isGenerating}>
              {isGenerating ? "Finishing session…" : "Finish and generate assessment"}
            </button>
          </div>
        )}

        {!result && session?.status === "COMPLETED" && (
          <div className="assessment-finish">
            <p>The session is complete. Assessment generation requires the text, voice, or fusion model endpoints to return valid results.</p>
            <button type="button" className="service-home-button" onClick={() => void generateAssessment()} disabled={isGenerating}>
              {isGenerating ? "Requesting assessment…" : "Retry assessment generation"}
            </button>
          </div>
        )}

        {!result && session?.status === "PAUSED" && !crisis && (
          <div className="assessment-finish">
            <p>This session is paused. You can continue when you’re ready.</p>
            <button type="button" className="service-home-button" onClick={() => void continuePausedSession()} disabled={isSubmitting}>
              {isSubmitting ? "Resuming…" : "Continue session"}
            </button>
          </div>
        )}

        {session?.status === "ABANDONED" && <p role="status">This session was abandoned. Return to your dashboard to start another one.</p>}
      </section>
    </main>
  );
}
