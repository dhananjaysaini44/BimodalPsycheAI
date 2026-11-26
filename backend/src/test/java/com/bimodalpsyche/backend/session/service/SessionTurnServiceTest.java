package com.bimodalpsyche.backend.session.service;

import com.bimodalpsyche.backend.question.model.QuestionVariant;
import com.bimodalpsyche.backend.session.exception.InvalidSessionStateException;
import com.bimodalpsyche.backend.session.exception.MlAnalysisUnavailableException;
import com.bimodalpsyche.backend.session.exception.SessionNotFoundException;
import com.bimodalpsyche.backend.session.model.AssessmentSession;
import com.bimodalpsyche.backend.session.model.SessionStatus;
import com.bimodalpsyche.backend.session.model.SessionTurn;
import com.bimodalpsyche.backend.session.repository.AssessmentSessionRepository;
import com.bimodalpsyche.backend.session.repository.AudioFileRepository;
import com.bimodalpsyche.backend.session.repository.CrisisEventRepository;
import com.bimodalpsyche.backend.crisis.repository.CrisisResourceRepository;
import com.bimodalpsyche.backend.session.repository.SessionTurnRepository;
import com.bimodalpsyche.backend.session.repository.TurnAnalysisRepository;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import java.util.Optional;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

@ExtendWith(MockitoExtension.class)
class SessionTurnServiceTest {
    private static final String EMAIL = "person@example.com";
    private static final Long SESSION_ID = 25L;
    private static final Long QUESTION_ID = 7L;

    @Mock
    private AssessmentSessionRepository sessionRepository;
    @Mock
    private SessionTurnRepository sessionTurnRepository;
    @Mock
    private AudioFileRepository audioFileRepository;
    @Mock
    private TurnAnalysisRepository turnAnalysisRepository;
    @Mock
    private MlAnalysisClient mlAnalysisClient;
    @Mock
    private TextEncryptionService textEncryptionService;
    @Mock
    private SessionQuestionService sessionQuestionService;
    @Mock
    private CrisisEventRepository crisisEventRepository;
    @Mock
    private CrisisResourceRepository crisisResourceRepository;

    private SessionTurnService service;

    @BeforeEach
    void setUp() {
        service = new SessionTurnService(
                sessionRepository,
                sessionTurnRepository,
                audioFileRepository,
                turnAnalysisRepository,
                mlAnalysisClient,
                textEncryptionService,
                sessionQuestionService,
                crisisEventRepository,
                crisisResourceRepository);
    }

    @Test
    void analyzesAndPersistsOwnedActiveSessionResponse() {
        AssessmentSession session = activeSession();
        QuestionVariant question = new QuestionVariant();
        question.setId(QUESTION_ID);
        question.setQuestionText("How have you been feeling?");
        when(sessionRepository.findByIdAndUser_EmailForUpdate(SESSION_ID, EMAIL)).thenReturn(Optional.of(session));
        when(sessionQuestionService.requireExpectedQuestion(session, QUESTION_ID)).thenReturn(question);
        when(mlAnalysisClient.analyze("Some answer"))
                .thenReturn(new MlAnalysisClient.TurnAnalysisResult(-0.4f, false));
        when(textEncryptionService.encrypt("Some answer")).thenReturn("v1:encrypted");
        when(sessionTurnRepository.saveAndFlush(any(SessionTurn.class))).thenAnswer(invocation -> {
            SessionTurn turn = invocation.getArgument(0);
            turn.setId(101L);
            return turn;
        });

        SessionTurnService.TurnResponse response = service.submitTextTurn(
                SESSION_ID, EMAIL, QUESTION_ID, "Some answer");

        assertEquals(101L, response.id());
        assertEquals(1, response.turnNumber());
        assertEquals("How have you been feeling?", response.questionText());
        assertEquals(-0.4f, response.sentimentScore());
        assertFalse(response.crisisDetected());
        verify(sessionTurnRepository).saveAndFlush(org.mockito.ArgumentMatchers.argThat(turn ->
                "v1:encrypted".equals(turn.getResponseTextEncrypted())
                        && "How have you been feeling?".equals(turn.getQuestionTextSnapshot())
                        && turn.getTurnNumber() == 1));
        verify(turnAnalysisRepository).save(any());
        verify(sessionRepository).save(session);
        assertEquals(1, session.getCurrentTurnNumber());
    }

    @Test
    void doesNotCallMlForSessionNotOwnedByUser() {
        when(sessionRepository.findByIdAndUser_EmailForUpdate(SESSION_ID, EMAIL)).thenReturn(Optional.empty());

        assertThrows(SessionNotFoundException.class, () -> service.submitTextTurn(
                SESSION_ID, EMAIL, QUESTION_ID, "answer"));

        verify(mlAnalysisClient, never()).analyze(any());
        verify(sessionTurnRepository, never()).saveAndFlush(any());
    }

    @Test
    void refusesResponsesForCompletedSession() {
        AssessmentSession session = activeSession();
        session.setStatus(SessionStatus.COMPLETED);
        when(sessionRepository.findByIdAndUser_EmailForUpdate(SESSION_ID, EMAIL)).thenReturn(Optional.of(session));

        assertThrows(InvalidSessionStateException.class, () -> service.submitTextTurn(
                SESSION_ID, EMAIL, QUESTION_ID, "answer"));

        verify(mlAnalysisClient, never()).analyze(any());
    }

    @Test
    void doesNotPersistResponseWhenMlIsUnavailable() {
        QuestionVariant question = new QuestionVariant();
        when(sessionRepository.findByIdAndUser_EmailForUpdate(SESSION_ID, EMAIL))
                .thenReturn(Optional.of(activeSession()));
        when(sessionQuestionService.requireExpectedQuestion(any(), org.mockito.ArgumentMatchers.eq(QUESTION_ID)))
                .thenReturn(question);
        when(mlAnalysisClient.analyze("answer")).thenThrow(new IllegalStateException("service unavailable"));

        assertThrows(MlAnalysisUnavailableException.class, () -> service.submitTextTurn(
                SESSION_ID, EMAIL, QUESTION_ID, "answer"));

        verify(sessionTurnRepository, never()).saveAndFlush(any());
        verify(turnAnalysisRepository, never()).save(any());
    }

    @Test
    void pausesSessionAndReturnsGuidanceWhenCrisisSignalIsRaised() {
        AssessmentSession session = activeSession();
        QuestionVariant question = new QuestionVariant();
        when(sessionRepository.findByIdAndUser_EmailForUpdate(SESSION_ID, EMAIL)).thenReturn(Optional.of(session));
        when(sessionQuestionService.requireExpectedQuestion(session, QUESTION_ID)).thenReturn(question);
        when(mlAnalysisClient.analyze("help"))
                .thenReturn(new MlAnalysisClient.TurnAnalysisResult(-0.9f, true));
        when(textEncryptionService.encrypt("help")).thenReturn("v1:encrypted");
        when(sessionTurnRepository.saveAndFlush(any(SessionTurn.class))).thenAnswer(invocation -> {
            SessionTurn turn = invocation.getArgument(0);
            turn.setId(102L);
            return turn;
        });

        SessionTurnService.TurnResponse response = service.submitTextTurn(SESSION_ID, EMAIL, QUESTION_ID, "help");

        assertTrue(response.crisisDetected());
        assertTrue(response.crisisGuidance().contains("not a diagnosis"));
        assertEquals(SessionStatus.PAUSED, session.getStatus());
        verify(crisisEventRepository).save(any());
        verify(sessionRepository).save(session);
    }

    private AssessmentSession activeSession() {
        AssessmentSession session = new AssessmentSession();
        session.setId(SESSION_ID);
        session.setStatus(SessionStatus.IN_PROGRESS);
        session.setCurrentTurnNumber(0);
        return session;
    }
}
