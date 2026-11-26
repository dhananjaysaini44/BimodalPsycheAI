package com.bimodalpsyche.backend.session.service;

import com.bimodalpsyche.backend.crisis.model.CrisisEvent;
import com.bimodalpsyche.backend.crisis.repository.CrisisResourceRepository;
import com.bimodalpsyche.backend.question.model.QuestionVariant;
import com.bimodalpsyche.backend.session.exception.InvalidSessionStateException;
import com.bimodalpsyche.backend.session.exception.MlAnalysisUnavailableException;
import com.bimodalpsyche.backend.session.exception.SessionNotFoundException;
import com.bimodalpsyche.backend.session.model.AssessmentSession;
import com.bimodalpsyche.backend.session.model.AudioFile;
import com.bimodalpsyche.backend.session.model.ResponseType;
import com.bimodalpsyche.backend.session.model.SessionStatus;
import com.bimodalpsyche.backend.session.model.SessionTurn;
import com.bimodalpsyche.backend.session.model.TurnAnalysis;
import com.bimodalpsyche.backend.session.repository.AssessmentSessionRepository;
import com.bimodalpsyche.backend.session.repository.AudioFileRepository;
import com.bimodalpsyche.backend.session.repository.CrisisEventRepository;
import com.bimodalpsyche.backend.session.repository.SessionTurnRepository;
import com.bimodalpsyche.backend.session.repository.TurnAnalysisRepository;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.time.LocalDateTime;

@Service
@Slf4j
public class SessionTurnService {
    private final AssessmentSessionRepository sessionRepository;
    private final SessionTurnRepository sessionTurnRepository;
    private final AudioFileRepository audioFileRepository;
    private final TurnAnalysisRepository turnAnalysisRepository;
    private final MlAnalysisClient mlAnalysisClient;
    private final TextEncryptionService textEncryptionService;
    private final SessionQuestionService sessionQuestionService;
    private final CrisisEventRepository crisisEventRepository;
    private final CrisisResourceRepository crisisResourceRepository;

    public SessionTurnService(
            AssessmentSessionRepository sessionRepository,
            SessionTurnRepository sessionTurnRepository,
            AudioFileRepository audioFileRepository,
            TurnAnalysisRepository turnAnalysisRepository,
            MlAnalysisClient mlAnalysisClient,
            TextEncryptionService textEncryptionService,
            SessionQuestionService sessionQuestionService,
            CrisisEventRepository crisisEventRepository,
            CrisisResourceRepository crisisResourceRepository) {
        this.sessionRepository = sessionRepository;
        this.sessionTurnRepository = sessionTurnRepository;
        this.audioFileRepository = audioFileRepository;
        this.turnAnalysisRepository = turnAnalysisRepository;
        this.mlAnalysisClient = mlAnalysisClient;
        this.textEncryptionService = textEncryptionService;
        this.sessionQuestionService = sessionQuestionService;
        this.crisisEventRepository = crisisEventRepository;
        this.crisisResourceRepository = crisisResourceRepository;
    }

    @Transactional
    public TurnResponse submitTextTurn(Long sessionId, String email, Long questionVariantId, String responseText) {
        AssessmentSession session = sessionRepository.findByIdAndUser_EmailForUpdate(sessionId, email)
                .orElseThrow(() -> new SessionNotFoundException("Session not found"));
        if (session.getStatus() != SessionStatus.IN_PROGRESS) {
            throw new InvalidSessionStateException("A response can only be added to an in-progress session.");
        }

        QuestionVariant questionVariant = sessionQuestionService.requireExpectedQuestion(session, questionVariantId);

        MlAnalysisClient.TurnAnalysisResult analysis;
        try {
            analysis = mlAnalysisClient.analyze(responseText);
        } catch (RuntimeException ex) {
            if (ex instanceof MlAnalysisClient.MlServiceHttpException mlException) {
                log.warn("ML analysis failed at {} with HTTP {}", mlException.endpoint(), mlException.statusCode());
            } else {
                log.warn("ML analysis failed for a session turn ({})", ex.getClass().getSimpleName());
            }
            throw new MlAnalysisUnavailableException(
                    "Turn analysis is temporarily unavailable. Your response was not saved; please try again.", ex);
        }

        LocalDateTime now = LocalDateTime.now();
        SessionTurn turn = new SessionTurn();
        turn.setSession(session);
        turn.setTurnNumber(session.getCurrentTurnNumber() + 1);
        turn.setQuestionVariant(questionVariant);
        turn.setQuestionTextSnapshot(questionVariant.getQuestionText());
        turn.setResponseType(ResponseType.TEXT);
        turn.setResponseTextEncrypted(textEncryptionService.encrypt(responseText));
        turn.setSentimentScore(analysis.sentimentScore());
        turn.setCrisisDetected(analysis.crisisDetected());
        turn = sessionTurnRepository.saveAndFlush(turn);

        java.util.List<CrisisResourceResponse> crisisResources = crisisResources(analysis.crisisDetected());
        if (analysis.crisisDetected()) {
            CrisisEvent event = new CrisisEvent();
            event.setSession(session);
            event.setTurn(turn);
            event.setDetectedAt(now);
            event.setDetectionType("ML_TEXT_SIGNAL");
            event.setPatternCode("KEYWORD_SIGNAL");
            event.setResourceShown(!crisisResources.isEmpty());
            crisisEventRepository.save(event);
            session.setStatus(SessionStatus.PAUSED);
        }

        TurnAnalysis turnAnalysis = new TurnAnalysis();
        turnAnalysis.setTurn(turn);
        turnAnalysis.setSentimentScore(analysis.sentimentScore());
        turnAnalysis.setCrisisDetected(analysis.crisisDetected());
        turnAnalysis.setAnalyzedAt(now);
        turnAnalysisRepository.save(turnAnalysis);

        session.setCurrentTurnNumber(turn.getTurnNumber());
        session.setCurrentQuestionVariant(questionVariant);
        session.setLastActivityAt(now);
        sessionRepository.save(session);

        return new TurnResponse(
                turn.getId(),
                turn.getTurnNumber(),
                turn.getQuestionTextSnapshot(),
                analysis.sentimentScore(),
                analysis.crisisDetected(),
                crisisGuidance(analysis.crisisDetected()),
                crisisResources,
                now);
    }

    @Transactional(readOnly = true)
    public java.util.List<TurnHistoryResponse> getTurns(Long sessionId, String email) {
        sessionRepository.findByIdAndUser_Email(sessionId, email)
                .orElseThrow(() -> new SessionNotFoundException("Session not found"));
        return sessionTurnRepository.findAllBySession_IdOrderByTurnNumberAsc(sessionId)
                .stream()
                .map(turn -> {
                    AudioFile audio = turn.getResponseType() == ResponseType.AUDIO
                            ? audioFileRepository.findByTurn_Id(turn.getId()).orElseThrow(
                                    () -> new IllegalStateException("Audio metadata is missing for a saved audio turn"))
                            : null;
                    return new TurnHistoryResponse(
                            turn.getId(),
                            turn.getTurnNumber(),
                            turn.getQuestionTextSnapshot(),
                            turn.getResponseType(),
                            turn.getResponseTextEncrypted() == null ? null
                                    : textEncryptionService.decrypt(turn.getResponseTextEncrypted()),
                            audio == null ? null : new AudioMetadata(
                                    audio.getId(), audio.getOriginalFilename(), audio.getMimeType(), audio.getFileSize()),
                            turn.getSentimentScore(),
                            turn.isCrisisDetected(),
                            crisisGuidance(turn.isCrisisDetected()),
                            crisisResources(turn.isCrisisDetected()),
                            turn.getCreatedAt());
                })
                .toList();
    }

    private String crisisGuidance(boolean crisisDetected) {
        if (!crisisDetected) {
            return null;
        }
        return "Some words in your response triggered an automated support signal. This is not a diagnosis or an emergency assessment. The conversation is paused. If you may be in immediate danger, contact your local emergency service or a trusted person now; consider reaching out to a qualified mental-health professional.";
    }

    private java.util.List<CrisisResourceResponse> crisisResources(boolean crisisDetected) {
        if (!crisisDetected) {
            return java.util.List.of();
        }
        return crisisResourceRepository.findAllByActiveTrueOrderByPriorityAsc().stream()
                .map(resource -> new CrisisResourceResponse(
                        resource.getName(),
                        resource.getDescription(),
                        resource.getPhone(),
                        resource.getUrl(),
                        resource.getCountry(),
                        resource.getResourceType()))
                .toList();
    }

    public record TurnResponse(
            Long id,
            int turnNumber,
            String questionText,
            float sentimentScore,
            boolean crisisDetected,
            String crisisGuidance,
            java.util.List<CrisisResourceResponse> crisisResources,
            LocalDateTime createdAt) {
    }

    public record TurnHistoryResponse(
            Long id,
            int turnNumber,
            String questionText,
            ResponseType responseType,
            String responseText,
            AudioMetadata audio,
            Float sentimentScore,
            boolean crisisDetected,
            String crisisGuidance,
            java.util.List<CrisisResourceResponse> crisisResources,
            LocalDateTime createdAt) {
    }

    public record AudioMetadata(Long id, String filename, String mimeType, Long sizeBytes) {
    }

    public record CrisisResourceResponse(
            String name,
            String description,
            String phone,
            String url,
            String country,
            String resourceType) {
    }
}
