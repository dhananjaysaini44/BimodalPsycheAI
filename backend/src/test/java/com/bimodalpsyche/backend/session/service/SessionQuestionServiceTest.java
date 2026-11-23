package com.bimodalpsyche.backend.session.service;

import com.bimodalpsyche.backend.question.model.QuestionBankVersion;
import com.bimodalpsyche.backend.question.model.QuestionTheme;
import com.bimodalpsyche.backend.question.model.QuestionVariant;
import com.bimodalpsyche.backend.question.model.VariantType;
import com.bimodalpsyche.backend.question.repository.QuestionBankVersionRepository;
import com.bimodalpsyche.backend.question.repository.QuestionVariantRepository;
import com.bimodalpsyche.backend.session.exception.InvalidSessionStateException;
import com.bimodalpsyche.backend.session.model.AssessmentSession;
import com.bimodalpsyche.backend.session.model.SessionStatus;
import com.bimodalpsyche.backend.session.model.SessionTurn;
import com.bimodalpsyche.backend.session.repository.AssessmentSessionRepository;
import com.bimodalpsyche.backend.session.repository.SessionTurnRepository;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import java.util.List;
import java.util.Optional;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.lenient;
import static org.mockito.Mockito.when;

@ExtendWith(MockitoExtension.class)
class SessionQuestionServiceTest {
    private static final Long SESSION_ID = 12L;
    private static final String EMAIL = "person@example.com";

    @Mock
    private AssessmentSessionRepository sessionRepository;
    @Mock
    private QuestionBankVersionRepository bankVersionRepository;
    @Mock
    private QuestionVariantRepository variantRepository;
    @Mock
    private SessionTurnRepository turnRepository;

    private SessionQuestionService service;
    private QuestionBankVersion bank;
    private List<QuestionVariant> variants;

    @BeforeEach
    void setUp() {
        service = new SessionQuestionService(
                sessionRepository, bankVersionRepository, variantRepository, turnRepository);
        bank = new QuestionBankVersion();
        bank.setId(1L);
        bank.setActive(true);
        QuestionTheme mood = theme(11L, "mood");
        QuestionTheme sleep = theme(12L, "sleep");
        variants = List.of(
                variant(101L, mood, VariantType.LIGHT, "Mood light"),
                variant(102L, mood, VariantType.DEEP, "Mood deep"),
                variant(201L, sleep, VariantType.LIGHT, "Sleep light"),
                variant(202L, sleep, VariantType.DEEP, "Sleep deep"));
        lenient().when(bankVersionRepository.findFirstByActiveTrueOrderByIdDesc()).thenReturn(Optional.of(bank));
        lenient().when(variantRepository.findAllByTheme_BankVersion_IdOrderByTheme_IdAscIdAsc(1L)).thenReturn(variants);
    }

    @Test
    void firstPromptUsesLightVersionAndIsStableBeforeAnAnswer() {
        AssessmentSession session = activeSession();
        when(sessionRepository.findByIdAndUser_Email(SESSION_ID, EMAIL)).thenReturn(Optional.of(session));
        when(turnRepository.findAllBySession_IdOrderByTurnNumberAsc(SESSION_ID)).thenReturn(List.of());

        SessionQuestionService.NextQuestionResponse response = service.getNextQuestion(SESSION_ID, EMAIL);

        assertTrue(response.available());
        assertFalse(response.conversationComplete());
        assertEquals(101L, response.question().id());
        assertEquals(VariantType.LIGHT, response.question().variantType());
    }

    @Test
    void selectsDeepVariantForNextThemeAfterNegativeSentiment() {
        AssessmentSession session = activeSession();
        when(sessionRepository.findByIdAndUser_Email(SESSION_ID, EMAIL)).thenReturn(Optional.of(session));
        when(turnRepository.findAllBySession_IdOrderByTurnNumberAsc(SESSION_ID))
                .thenReturn(List.of(turn(1, variants.getFirst(), -0.5f)));

        SessionQuestionService.NextQuestionResponse response = service.getNextQuestion(SESSION_ID, EMAIL);

        assertEquals("sleep", response.question().themeCode());
        assertEquals(202L, response.question().id());
        assertEquals(VariantType.DEEP, response.question().variantType());
    }

    @Test
    void reportsConversationCompleteAfterEveryThemeHasAnAnswer() {
        AssessmentSession session = activeSession();
        when(sessionRepository.findByIdAndUser_Email(SESSION_ID, EMAIL)).thenReturn(Optional.of(session));
        when(turnRepository.findAllBySession_IdOrderByTurnNumberAsc(SESSION_ID))
                .thenReturn(List.of(turn(1, variants.getFirst(), -0.5f), turn(2, variants.get(2), 0.1f)));

        assertTrue(service.hasCompletedQuestionBank(session));
        SessionQuestionService.NextQuestionResponse response = service.getNextQuestion(SESSION_ID, EMAIL);
        assertTrue(response.conversationComplete());
        assertFalse(response.available());
        assertNull(response.question());
    }

    @Test
    void pausesQuestionDeliveryWhileSessionIsPaused() {
        AssessmentSession session = activeSession();
        session.setStatus(SessionStatus.PAUSED);
        when(sessionRepository.findByIdAndUser_Email(SESSION_ID, EMAIL)).thenReturn(Optional.of(session));

        SessionQuestionService.NextQuestionResponse response = service.getNextQuestion(SESSION_ID, EMAIL);

        assertFalse(response.available());
        assertFalse(response.conversationComplete());
        assertNull(response.question());
    }

    @Test
    void rejectsQuestionIdsOtherThanTheServerSelectedNextQuestion() {
        AssessmentSession session = activeSession();
        when(turnRepository.findAllBySession_IdOrderByTurnNumberAsc(SESSION_ID)).thenReturn(List.of());

        assertThrows(InvalidSessionStateException.class, () -> service.requireExpectedQuestion(session, 201L));
    }

    private AssessmentSession activeSession() {
        AssessmentSession session = new AssessmentSession();
        session.setId(SESSION_ID);
        session.setStatus(SessionStatus.IN_PROGRESS);
        session.setQuestionBankVersion(bank);
        return session;
    }

    private QuestionTheme theme(Long id, String code) {
        QuestionTheme theme = new QuestionTheme();
        theme.setId(id);
        theme.setBankVersion(bank);
        theme.setThemeCode(code);
        theme.setDisplayName(code);
        return theme;
    }

    private QuestionVariant variant(Long id, QuestionTheme theme, VariantType type, String text) {
        QuestionVariant variant = new QuestionVariant();
        variant.setId(id);
        variant.setTheme(theme);
        variant.setVariantType(type);
        variant.setQuestionText(text);
        return variant;
    }

    private SessionTurn turn(int turnNumber, QuestionVariant question, float sentiment) {
        SessionTurn turn = new SessionTurn();
        turn.setTurnNumber(turnNumber);
        turn.setQuestionVariant(question);
        turn.setSentimentScore(sentiment);
        return turn;
    }
}
