package com.bimodalpsyche.backend.session.service;

import com.bimodalpsyche.backend.assessment.model.Assessment;
import com.bimodalpsyche.backend.assessment.repository.AssessmentAnalysisResultRepository;
import com.bimodalpsyche.backend.assessment.repository.AssessmentRepository;
import com.bimodalpsyche.backend.assessment.repository.AssessmentSignalRepository;
import com.bimodalpsyche.backend.session.exception.SessionNotFoundException;
import com.bimodalpsyche.backend.session.model.AssessmentSession;
import com.bimodalpsyche.backend.session.model.AudioFile;
import com.bimodalpsyche.backend.session.model.SessionTurn;
import com.bimodalpsyche.backend.session.repository.AssessmentSessionRepository;
import com.bimodalpsyche.backend.session.repository.AudioFileRepository;
import com.bimodalpsyche.backend.session.repository.CrisisEventRepository;
import com.bimodalpsyche.backend.session.repository.SessionTurnRepository;
import com.bimodalpsyche.backend.session.repository.TurnAnalysisRepository;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.transaction.support.TransactionSynchronization;
import org.springframework.transaction.support.TransactionSynchronizationManager;

import java.util.List;

@Service
@Slf4j
public class SessionDeletionService {
    private final AssessmentSessionRepository sessionRepository;
    private final SessionTurnRepository turnRepository;
    private final AudioFileRepository audioFileRepository;
    private final TurnAnalysisRepository turnAnalysisRepository;
    private final CrisisEventRepository crisisEventRepository;
    private final AssessmentRepository assessmentRepository;
    private final AssessmentAnalysisResultRepository analysisResultRepository;
    private final AssessmentSignalRepository assessmentSignalRepository;
    private final AudioStorageService audioStorageService;

    public SessionDeletionService(
            AssessmentSessionRepository sessionRepository,
            SessionTurnRepository turnRepository,
            AudioFileRepository audioFileRepository,
            TurnAnalysisRepository turnAnalysisRepository,
            CrisisEventRepository crisisEventRepository,
            AssessmentRepository assessmentRepository,
            AssessmentAnalysisResultRepository analysisResultRepository,
            AssessmentSignalRepository assessmentSignalRepository,
            AudioStorageService audioStorageService) {
        this.sessionRepository = sessionRepository;
        this.turnRepository = turnRepository;
        this.audioFileRepository = audioFileRepository;
        this.turnAnalysisRepository = turnAnalysisRepository;
        this.crisisEventRepository = crisisEventRepository;
        this.assessmentRepository = assessmentRepository;
        this.analysisResultRepository = analysisResultRepository;
        this.assessmentSignalRepository = assessmentSignalRepository;
        this.audioStorageService = audioStorageService;
    }

    @Transactional
    public void delete(Long sessionId, String email) {
        AssessmentSession session = sessionRepository.findByIdAndUser_EmailForUpdate(sessionId, email)
                .orElseThrow(() -> new SessionNotFoundException("Session not found"));

        List<SessionTurn> turns = turnRepository.findAllBySession_IdOrderByTurnNumberAsc(sessionId);
        List<AudioFile> audioFiles = audioFileRepository.findAllByTurn_Session_IdOrderByTurn_TurnNumberAsc(sessionId);
        List<Assessment> assessments = assessmentRepository.findAllBySession_IdIn(List.of(sessionId));
        List<Long> turnIds = turns.stream().map(SessionTurn::getId).toList();
        List<Long> assessmentIds = assessments.stream().map(Assessment::getId).toList();

        crisisEventRepository.deleteAllBySession_Id(sessionId);
        if (!assessmentIds.isEmpty()) {
            assessmentSignalRepository.deleteAllByAssessment_IdIn(assessmentIds);
            analysisResultRepository.deleteAllByAssessment_IdIn(assessmentIds);
            assessmentRepository.deleteAll(assessments);
        }
        audioFileRepository.deleteAll(audioFiles);
        if (!turnIds.isEmpty()) {
            turnAnalysisRepository.deleteAllByTurn_IdIn(turnIds);
            turnRepository.deleteAll(turns);
        }
        sessionRepository.delete(session);

        if (TransactionSynchronizationManager.isSynchronizationActive()) {
            TransactionSynchronizationManager.registerSynchronization(new TransactionSynchronization() {
                @Override
                public void afterCommit() {
                    deleteStoredAudio(sessionId);
                }
            });
        } else {
            deleteStoredAudio(sessionId);
        }
    }

    private void deleteStoredAudio(Long sessionId) {
        try {
            audioStorageService.deleteSessionAudio(sessionId);
        } catch (RuntimeException exception) {
            log.error("Could not remove stored audio after deleting session {}", sessionId, exception);
        }
    }
}
