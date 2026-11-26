package com.bimodalpsyche.backend.session.service;

import com.bimodalpsyche.backend.question.model.QuestionVariant;
import com.bimodalpsyche.backend.question.model.QuestionBankVersion;
import com.bimodalpsyche.backend.question.model.VariantType;
import com.bimodalpsyche.backend.question.repository.QuestionBankVersionRepository;
import com.bimodalpsyche.backend.question.repository.QuestionVariantRepository;
import com.bimodalpsyche.backend.session.exception.InvalidSessionStateException;
import com.bimodalpsyche.backend.session.exception.SessionNotFoundException;
import com.bimodalpsyche.backend.session.model.AssessmentSession;
import com.bimodalpsyche.backend.session.model.SessionStatus;
import com.bimodalpsyche.backend.session.model.SessionTurn;
import com.bimodalpsyche.backend.session.repository.AssessmentSessionRepository;
import com.bimodalpsyche.backend.session.repository.SessionTurnRepository;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.HashSet;
import java.util.List;
import java.util.Set;

@Service
public class SessionQuestionService {
    private static final float DEEP_QUESTION_SENTIMENT_THRESHOLD = -0.25f;

    private final AssessmentSessionRepository sessionRepository;
    private final QuestionBankVersionRepository bankVersionRepository;
    private final QuestionVariantRepository variantRepository;
    private final SessionTurnRepository turnRepository;

    public SessionQuestionService(
            AssessmentSessionRepository sessionRepository,
            QuestionBankVersionRepository bankVersionRepository,
            QuestionVariantRepository variantRepository,
            SessionTurnRepository turnRepository) {
        this.sessionRepository = sessionRepository;
        this.bankVersionRepository = bankVersionRepository;
        this.variantRepository = variantRepository;
        this.turnRepository = turnRepository;
    }

    @Transactional(readOnly = true)
    public NextQuestionResponse getNextQuestion(Long sessionId, String email) {
        AssessmentSession session = getOwnedSession(sessionId, email);
        if (session.getStatus() != SessionStatus.IN_PROGRESS && session.getStatus() != SessionStatus.PAUSED) {
            throw new InvalidSessionStateException("Questions are only available for unfinished sessions.");
        }
        if (session.getStatus() == SessionStatus.PAUSED) {
            return new NextQuestionResponse(false, false, null);
        }
        return nextQuestion(session, turnRepository.findAllBySession_IdOrderByTurnNumberAsc(sessionId));
    }

    @Transactional(readOnly = true)
    public QuestionVariant requireExpectedQuestion(AssessmentSession session, Long questionVariantId) {
        List<SessionTurn> turns = turnRepository.findAllBySession_IdOrderByTurnNumberAsc(session.getId());
        NextQuestionResponse expected = nextQuestion(session, turns);
        if (expected.question() == null || !expected.question().id().equals(questionVariantId)) {
            throw new InvalidSessionStateException("Submit a response to the next question returned for this session.");
        }
        return variantRepository.findByIdAndTheme_BankVersion_Id(questionVariantId, questionBankVersion(session).getId())
                .orElseThrow(() -> new InvalidSessionStateException("The selected question is no longer active."));
    }

    @Transactional(readOnly = true)
    public boolean hasCompletedQuestionBank(Long sessionId) {
        AssessmentSession session = sessionRepository.findById(sessionId)
                .orElseThrow(() -> new SessionNotFoundException("Session not found"));
        return hasCompletedQuestionBank(session);
    }

    @Transactional(readOnly = true)
    public boolean hasCompletedQuestionBank(AssessmentSession session) {
        QuestionBankVersion version = questionBankVersion(session);
        List<QuestionVariant> variants = variantRepository
                .findAllByTheme_BankVersion_IdOrderByTheme_IdAscIdAsc(version.getId());
        Set<Long> requiredThemeIds = new HashSet<>();
        variants.forEach(variant -> requiredThemeIds.add(variant.getTheme().getId()));
        if (requiredThemeIds.isEmpty()) {
            throw new IllegalStateException("The active question bank has no question themes");
        }
        Set<Long> answeredThemeIds = new HashSet<>();
        turnRepository.findAllBySession_IdOrderByTurnNumberAsc(session.getId())
                .forEach(turn -> {
                    Long answeredThemeId = turn.getQuestionVariant().getTheme().getId();
                    if (requiredThemeIds.contains(answeredThemeId)) {
                        answeredThemeIds.add(answeredThemeId);
                    }
                });
        return answeredThemeIds.containsAll(requiredThemeIds);
    }

    private NextQuestionResponse nextQuestion(AssessmentSession session, List<SessionTurn> turns) {
        QuestionBankVersion version = questionBankVersion(session);
        List<QuestionVariant> variants = variantRepository
                .findAllByTheme_BankVersion_IdOrderByTheme_IdAscIdAsc(version.getId());
        Set<Long> answeredThemeIds = new HashSet<>();
        turns.forEach(turn -> answeredThemeIds.add(turn.getQuestionVariant().getTheme().getId()));
        VariantType preferredType = turns.isEmpty() || turns.getLast().getSentimentScore() == null
                || turns.getLast().getSentimentScore() > DEEP_QUESTION_SENTIMENT_THRESHOLD
                ? VariantType.LIGHT
                : VariantType.DEEP;

        QuestionVariant selected = variants.stream()
                .filter(variant -> !answeredThemeIds.contains(variant.getTheme().getId()))
                .findFirst()
                .map(firstForTheme -> variants.stream()
                        .filter(variant -> variant.getTheme().getId().equals(firstForTheme.getTheme().getId()))
                        .filter(variant -> variant.getVariantType() == preferredType)
                        .findFirst()
                        .orElse(firstForTheme))
                .orElse(null);
        if (selected == null) {
            return new NextQuestionResponse(false, true, null);
        }
        return new NextQuestionResponse(true, false, new QuestionResponse(
                selected.getId(),
                selected.getTheme().getThemeCode(),
                selected.getTheme().getDisplayName(),
                selected.getVariantType(),
                selected.getQuestionText()));
    }

    private QuestionBankVersion questionBankVersion(AssessmentSession session) {
        if (session.getQuestionBankVersion() != null) {
            return session.getQuestionBankVersion();
        }
        return bankVersionRepository.findFirstByActiveTrueOrderByIdDesc()
                .orElseThrow(() -> new IllegalStateException("No active question bank is configured"));
    }

    private AssessmentSession getOwnedSession(Long sessionId, String email) {
        return sessionRepository.findByIdAndUser_Email(sessionId, email)
                .orElseThrow(() -> new SessionNotFoundException("Session not found"));
    }

    public record NextQuestionResponse(boolean available, boolean conversationComplete, QuestionResponse question) {
    }

    public record QuestionResponse(
            Long id,
            String themeCode,
            String themeName,
            VariantType variantType,
            String questionText) {
    }
}
