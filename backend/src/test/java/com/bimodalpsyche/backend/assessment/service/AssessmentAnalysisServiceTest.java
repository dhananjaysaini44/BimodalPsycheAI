package com.bimodalpsyche.backend.assessment.service;

import com.bimodalpsyche.backend.assessment.repository.AssessmentAnalysisResultRepository;
import com.bimodalpsyche.backend.assessment.repository.AssessmentRepository;
import com.bimodalpsyche.backend.session.exception.MlAnalysisUnavailableException;
import com.bimodalpsyche.backend.session.model.AssessmentSession;
import com.bimodalpsyche.backend.session.model.ResponseType;
import com.bimodalpsyche.backend.session.model.SessionStatus;
import com.bimodalpsyche.backend.session.model.SessionTurn;
import com.bimodalpsyche.backend.session.repository.AssessmentSessionRepository;
import com.bimodalpsyche.backend.session.repository.AudioFileRepository;
import com.bimodalpsyche.backend.session.repository.SessionTurnRepository;
import com.bimodalpsyche.backend.session.service.AudioStorageService;
import com.bimodalpsyche.backend.session.service.MlAnalysisClient;
import com.bimodalpsyche.backend.session.service.TextEncryptionService;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Optional;

import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class AssessmentAnalysisServiceTest {
    private final AssessmentSessionRepository sessionRepository = mock(AssessmentSessionRepository.class);
    private final SessionTurnRepository turnRepository = mock(SessionTurnRepository.class);
    private final AudioFileRepository audioFileRepository = mock(AudioFileRepository.class);
    private final AssessmentRepository assessmentRepository = mock(AssessmentRepository.class);
    private final AssessmentAnalysisResultRepository analysisResultRepository =
            mock(AssessmentAnalysisResultRepository.class);
    private final MlAnalysisClient mlAnalysisClient = mock(MlAnalysisClient.class);
    private final AudioStorageService audioStorageService = mock(AudioStorageService.class);
    private final TextEncryptionService encryptionService = mock(TextEncryptionService.class);
    private final AssessmentAnalysisService service = new AssessmentAnalysisService(
            sessionRepository, turnRepository, audioFileRepository, assessmentRepository,
            analysisResultRepository, mlAnalysisClient, audioStorageService, encryptionService);

    @Test
    void doesNotPersistPartialResultWhenModelAnalysisFails() {
        AssessmentSession session = new AssessmentSession();
        session.setId(21L);
        session.setStatus(SessionStatus.COMPLETED);
        SessionTurn turn = new SessionTurn();
        turn.setQuestionTextSnapshot("How have you been feeling?");
        turn.setResponseType(ResponseType.TEXT);
        turn.setResponseTextEncrypted("encrypted-answer");

        when(sessionRepository.findByIdAndUser_EmailForUpdate(21L, "owner@example.test"))
                .thenReturn(Optional.of(session));
        when(assessmentRepository.findBySession_Id(21L)).thenReturn(Optional.empty());
        when(turnRepository.findAllBySession_IdOrderByTurnNumberAsc(21L)).thenReturn(List.of(turn));
        when(audioFileRepository.findAllByTurn_Session_IdOrderByTurn_TurnNumberAsc(21L)).thenReturn(List.of());
        when(encryptionService.decrypt("encrypted-answer")).thenReturn("A dummy response.");
        when(mlAnalysisClient.analyzeText(any())).thenThrow(new IllegalStateException("ML placeholder"));

        assertThrows(MlAnalysisUnavailableException.class, () -> service.analyze(21L, "owner@example.test"));

        verify(assessmentRepository, never()).save(any());
        verify(analysisResultRepository, never()).save(any());
    }
}
