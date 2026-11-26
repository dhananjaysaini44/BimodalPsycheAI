package com.bimodalpsyche.backend.session.service;

import com.bimodalpsyche.backend.assessment.repository.AssessmentAnalysisResultRepository;
import com.bimodalpsyche.backend.assessment.repository.AssessmentRepository;
import com.bimodalpsyche.backend.assessment.repository.AssessmentSignalRepository;
import com.bimodalpsyche.backend.session.exception.SessionNotFoundException;
import com.bimodalpsyche.backend.session.model.AssessmentSession;
import com.bimodalpsyche.backend.session.repository.AssessmentSessionRepository;
import com.bimodalpsyche.backend.session.repository.AudioFileRepository;
import com.bimodalpsyche.backend.session.repository.CrisisEventRepository;
import com.bimodalpsyche.backend.session.repository.SessionTurnRepository;
import com.bimodalpsyche.backend.session.repository.TurnAnalysisRepository;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import java.util.List;
import java.util.Optional;

import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

@ExtendWith(MockitoExtension.class)
class SessionDeletionServiceTest {
    @Mock private AssessmentSessionRepository sessionRepository;
    @Mock private SessionTurnRepository turnRepository;
    @Mock private AudioFileRepository audioFileRepository;
    @Mock private TurnAnalysisRepository turnAnalysisRepository;
    @Mock private CrisisEventRepository crisisEventRepository;
    @Mock private AssessmentRepository assessmentRepository;
    @Mock private AssessmentAnalysisResultRepository analysisResultRepository;
    @Mock private AssessmentSignalRepository assessmentSignalRepository;
    @Mock private AudioStorageService audioStorageService;

    private SessionDeletionService service;

    @BeforeEach
    void setUp() {
        service = new SessionDeletionService(sessionRepository, turnRepository, audioFileRepository,
                turnAnalysisRepository, crisisEventRepository, assessmentRepository,
                analysisResultRepository, assessmentSignalRepository, audioStorageService);
    }

    @Test
    void deletesOwnedSessionAndRelatedDataAndAudio() {
        AssessmentSession session = new AssessmentSession();
        when(sessionRepository.findByIdAndUser_EmailForUpdate(42L, "person@example.com"))
                .thenReturn(Optional.of(session));
        when(turnRepository.findAllBySession_IdOrderByTurnNumberAsc(42L)).thenReturn(List.of());
        when(audioFileRepository.findAllByTurn_Session_IdOrderByTurn_TurnNumberAsc(42L)).thenReturn(List.of());
        when(assessmentRepository.findAllBySession_IdIn(List.of(42L))).thenReturn(List.of());

        service.delete(42L, "person@example.com");

        verify(crisisEventRepository).deleteAllBySession_Id(42L);
        verify(audioFileRepository).deleteAll(List.of());
        verify(sessionRepository).delete(session);
        verify(audioStorageService).deleteSessionAudio(42L);
    }

    @Test
    void doesNotDeleteSessionOwnedByAnotherUser() {
        when(sessionRepository.findByIdAndUser_EmailForUpdate(42L, "person@example.com"))
                .thenReturn(Optional.empty());

        assertThrows(SessionNotFoundException.class, () -> service.delete(42L, "person@example.com"));

        verify(sessionRepository, never()).delete(org.mockito.ArgumentMatchers.any(AssessmentSession.class));
        verify(audioStorageService, never()).deleteSessionAudio(42L);
    }
}
