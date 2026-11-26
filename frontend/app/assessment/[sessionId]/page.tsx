"use client";

import Link from "next/link";
import { useParams, useRouter } from "next/navigation";
import { useCallback, useEffect, useRef, useState } from "react";
import { ApiError, authApi, sessionApi, tokenManager } from "@/lib/api";
import type {
  AssessmentResponse,
  CrisisResource,
  NextQuestionResponse,
  SessionResponse,
  TurnHistoryResponse,
  TurnResponse,
} from "@/lib/api";

export default function AssessmentPage() {
  const params = useParams<{ sessionId: string }>();
  const router = useRouter();
  const sessionId = Number(params.sessionId);
  const [session, setSession] = useState<SessionResponse | null>(null);
  const [question, setQuestion] = useState<NextQuestionResponse["question"]>(null);
  const [turns, setTurns] = useState<TurnHistoryResponse[]>([]);
  const [responseText, setResponseText] = useState("");
  const [audioFile, setAudioFile] = useState<File | null>(null);
  const [audioPreviewUrl, setAudioPreviewUrl] = useState<string | null>(null);
  const [isRecording, setIsRecording] = useState(false);
  const [isStartingRecording, setIsStartingRecording] = useState(false);
  const [recordingError, setRecordingError] = useState("");
  const audioPreviewUrlRef = useRef<string | null>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const recordingChunksRef = useRef<Blob[]>([]);
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

  const loadTurns = useCallback(async () => {
    setTurns(await sessionApi.getTurns(sessionId));
  }, [sessionId]);

  useEffect(() => () => {
    const recorder = mediaRecorderRef.current;
    if (recorder && recorder.state !== "inactive") {
      recorder.ondataavailable = null;
      recorder.onstop = null;
      recorder.stop();
    }
    mediaStreamRef.current?.getTracks().forEach((track) => track.stop());
    if (audioPreviewUrlRef.current) URL.revokeObjectURL(audioPreviewUrlRef.current);
  }, []);

  function chooseAudioFile(file: File | null) {
    if (audioPreviewUrlRef.current) URL.revokeObjectURL(audioPreviewUrlRef.current);
    audioPreviewUrlRef.current = file ? URL.createObjectURL(file) : null;
    setAudioPreviewUrl(audioPreviewUrlRef.current);
    setAudioFile(file);
  }

  async function startRecording() {
    setRecordingError("");
    setError("");
    if (!window.isSecureContext) {
      setRecordingError("Microphone recording requires a secure page. Open this site using HTTPS or localhost.");
      return;
    }
    if (!navigator.mediaDevices?.getUserMedia || typeof MediaRecorder === "undefined") {
      setRecordingError("Audio recording isn’t supported by this browser. Please choose an audio file instead.");
      return;
    }

    let stream: MediaStream | null = null;
    setIsStartingRecording(true);
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true },
      });
      if (!stream.getAudioTracks().some((track) => track.readyState === "live")) {
        throw new Error("No active microphone was found. Check your microphone and try again.");
      }
      const supportedTypes = ["audio/webm;codecs=opus", "audio/webm", "audio/ogg;codecs=opus", "audio/mp4"];
      const mimeType = supportedTypes.find((type) => MediaRecorder.isTypeSupported(type));
      const recorder = mimeType ? new MediaRecorder(stream, { mimeType }) : new MediaRecorder(stream);
      mediaStreamRef.current = stream;
      mediaRecorderRef.current = recorder;
      recordingChunksRef.current = [];
      recorder.ondataavailable = (event) => {
        if (event.data.size > 0) recordingChunksRef.current.push(event.data);
      };
      recorder.onerror = () => {
        setRecordingError("Recording failed. Please try again or choose an audio file.");
        setIsRecording(false);
        recorder.onstop = null;
        if (recorder.state !== "inactive") {
          try {
            recorder.stop();
          } catch {
            // The recorder can already be stopped when the error event fires.
          }
        }
        stream?.getTracks().forEach((track) => track.stop());
        mediaStreamRef.current = null;
        mediaRecorderRef.current = null;
      };
      recorder.onstop = () => {
        const type = recorder.mimeType || recordingChunksRef.current[0]?.type || "audio/webm";
        const blob = new Blob(recordingChunksRef.current, { type });
        const extension = type.includes("ogg") ? "ogg" : type.includes("mp4") ? "mp4" : "webm";
        if (blob.size > 0) chooseAudioFile(new File([blob], `recording-${Date.now()}.${extension}`, { type }));
        else setRecordingError("No audio was captured. Please record again.");
        stream?.getTracks().forEach((track) => track.stop());
        mediaStreamRef.current = null;
        mediaRecorderRef.current = null;
        setIsRecording(false);
      };
      recorder.start(1000);
      chooseAudioFile(null);
      setIsRecording(true);
    } catch (cause) {
      stream?.getTracks().forEach((track) => track.stop());
      mediaStreamRef.current = null;
      setRecordingError(cause instanceof Error && cause.name === "NotAllowedError"
        ? "Microphone access was denied. Allow microphone access in your browser or choose an audio file."
        : cause instanceof Error ? cause.message : "Could not start recording. Please choose an audio file instead.");
    } finally {
      setIsStartingRecording(false);
    }
  }

  function stopRecording() {
    const recorder = mediaRecorderRef.current;
    if (recorder && recorder.state === "recording") recorder.stop();
  }

  function submitResponse(event: { preventDefault(): void }) {
    return audioFile ? submitAudio(event) : submitText(event);
  }

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
        await loadTurns();

        if (currentSession.status === "IN_PROGRESS") {
          await loadNextQuestion();
        } else if (currentSession.status === "COMPLETED") {
          await loadAssessment();
        }
      } catch (cause) {
        if (!active) return;
        if (cause instanceof ApiError && cause.status === 401) {
          router.replace(cause instanceof ApiError && cause.status === 401 && tokenManager.hasAuthenticatedSession() ? "/login?sessionExpired=1" : "/login");
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
  }, [loadAssessment, loadNextQuestion, loadTurns, router, sessionId]);

  async function submitText(event: { preventDefault(): void }) {
    event.preventDefault();
    if (!question || !responseText.trim()) return;
    setIsSubmitting(true);
    setError("");
    setNotice("");
    try {
      const submittedText = responseText.trim();
      const submitted: TurnResponse = await sessionApi.submitText(sessionId, question.id, submittedText);
      setResponseText("");
      setTurns((current) => [...current, {
        id: submitted.id,
        turnNumber: submitted.turnNumber,
        questionText: submitted.questionText,
        responseType: "TEXT",
        responseText: submittedText,
        audio: null,
        sentimentScore: submitted.sentimentScore,
        crisisDetected: submitted.crisisDetected,
        crisisGuidance: submitted.crisisGuidance,
        crisisResources: submitted.crisisResources,
        createdAt: submitted.createdAt,
      }]);
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
      chooseAudioFile(null);
      setTurns((current) => [...current, {
        id: uploaded.turnId,
        turnNumber: uploaded.turnNumber,
        questionText: uploaded.questionText,
        responseType: "AUDIO",
        responseText: null,
        audio: {
          id: uploaded.audioId,
          filename: uploaded.filename,
          mimeType: uploaded.mimeType,
          sizeBytes: uploaded.sizeBytes,
        },
        sentimentScore: null,
        crisisDetected: false,
        crisisGuidance: null,
        crisisResources: [],
        createdAt: uploaded.createdAt,
      }]);
      setSession((current) => current ? { ...current, currentTurnNumber: uploaded.turnNumber } : current);
      setNotice(`Audio saved securely. Analysis status: ${uploaded.analysisStatus}.`);
      await loadNextQuestion();
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Your audio could not be saved. Please try again.");
    } finally {
      setIsSubmitting(false);
    }
  }

  async function abandonSession() {
    if (!session || !["DRAFT", "IN_PROGRESS", "PAUSED"].includes(session.status)) return;
    if (!window.confirm("Abandon this assessment? You won’t be able to continue it, but its conversation will remain in your session history.")) return;
    setIsSubmitting(true);
    setError("");
    setNotice("");
    try {
      const abandoned = await sessionApi.abandon(sessionId);
      setSession(abandoned);
      setQuestion(null);
      setCrisis(null);
      setNotice("Session abandoned. Its conversation is still available from your dashboard.");
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Could not abandon this session.");
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
        <div className="service-topbar-actions">
          {session && ["DRAFT", "IN_PROGRESS", "PAUSED"].includes(session.status) && (
            <button type="button" className="session-abandon-button" onClick={() => void abandonSession()} disabled={isSubmitting || isGenerating}>Abandon session</button>
          )}
          <Link href="/dashboard" className="app-back-button">← Back to sessions</Link>
        </div>
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

        {turns.length > 0 && (
          <section className="assessment-transcript" aria-label="Session conversation">
            <h2 className="transcript-heading">Conversation</h2>
            {turns.map((turn) => (
              <div className="transcript-turn" key={turn.id}>
                <div className="chat-message chat-message-assistant">
                  <span className="chat-message-label">PsycheAI · Question {turn.turnNumber}</span>
                  <p>{turn.questionText}</p>
                </div>
                <div className="chat-message chat-message-user">
                  <span className="chat-message-label">You</span>
                  <p>{turn.responseType === "TEXT" ? turn.responseText : `Audio response${turn.audio?.filename ? ` · ${turn.audio.filename}` : ""}`}</p>
                </div>
              </div>
            ))}
          </section>
        )}

        {!result && session?.status === "IN_PROGRESS" && question && !crisis && (
          <>
            <div className="assessment-question chat-message chat-message-assistant">
              <span className="chat-message-label">PsycheAI · {question.themeName} · {question.variantType.toLowerCase()}</span>
              <p>{question.questionText}</p>
              <p className="history-help">Response {session.currentTurnNumber + 1}. Share only what you feel comfortable sharing.</p>
            </div>
            <form className="assessment-composer" onSubmit={(event) => void submitResponse(event)}>
              {audioFile && (
                <div className="audio-preview">
                  <span>{audioFile.name} · {(audioFile.size / (1024 * 1024)).toFixed(2)} MB</span>
                  {audioPreviewUrl && <audio controls src={audioPreviewUrl} />}
                  <button type="button" className="audio-discard-button" onClick={() => chooseAudioFile(null)} disabled={isSubmitting || isRecording}>Remove audio</button>
                </div>
              )}
              <textarea
                id="response-text"
                className="composer-textarea"
                value={responseText}
                onChange={(event) => setResponseText(event.target.value)}
                maxLength={10000}
                rows={2}
                disabled={isSubmitting || isRecording}
                placeholder={isRecording ? "Recording audio…" : "Message PsycheAI…"}
              />
              <div className="composer-toolbar">
                <div className="composer-tools">
                  <label className="composer-icon-button" htmlFor="audio-response" title="Attach an audio file" aria-label="Attach an audio file">
                    <span aria-hidden="true">＋</span>
                  </label>
                  {!isRecording ? (
                    <button type="button" className="composer-icon-button" onClick={() => void startRecording()} disabled={isSubmitting || isStartingRecording} title="Record audio" aria-label="Record audio">
                      <svg viewBox="0 0 24 24" fill="none" aria-hidden="true"><rect x="9" y="3" width="6" height="12" rx="3" /><path d="M5.5 11.5a6.5 6.5 0 0 0 13 0M12 18v3m-4 0h8" /></svg>
                    </button>
                  ) : (
                    <button type="button" className="audio-stop-button" onClick={stopRecording} title="Stop recording">
                      <span className="audio-recording-pulse" aria-hidden="true" /> Stop
                    </button>
                  )}
                  {isStartingRecording && <span className="audio-recording-label" role="status">Starting microphone…</span>}
                  {isRecording && <span className="audio-recording-label" role="status">Recording…</span>}
                </div>
                <button type="submit" className="composer-send-button" disabled={isSubmitting || isRecording || (!audioFile && !responseText.trim())} aria-label={isSubmitting ? "Sending response" : "Send response"}>
                  {isSubmitting ? "…" : "↑"}
                </button>
              </div>
              <p className="composer-hint">Type a message, attach an audio file, or record a voice response.</p>
              {recordingError && <p className="service-error" role="alert">{recordingError}</p>}
              <input
                id="audio-response"
                className="composer-file-input"
                type="file"
                disabled={isSubmitting || isRecording || isStartingRecording}
                accept="audio/wav,audio/webm,audio/mpeg,audio/mp4,audio/ogg,audio/aac,.wav,.webm,.mp3,.mp4,.ogg,.aac"
                onChange={(event) => {
                  chooseAudioFile(event.target.files?.[0] ?? null);
                  setRecordingError("");
                }}
              />
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
