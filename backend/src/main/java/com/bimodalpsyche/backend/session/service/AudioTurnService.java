package com.bimodalpsyche.backend.session.service;

import com.bimodalpsyche.backend.question.model.QuestionVariant;
import com.bimodalpsyche.backend.session.exception.InvalidSessionStateException;
import com.bimodalpsyche.backend.session.exception.SessionNotFoundException;
import com.bimodalpsyche.backend.session.model.AssessmentSession;
import com.bimodalpsyche.backend.session.model.AudioFile;
import com.bimodalpsyche.backend.session.model.ResponseType;
import com.bimodalpsyche.backend.session.model.SessionStatus;
import com.bimodalpsyche.backend.session.model.SessionTurn;
import com.bimodalpsyche.backend.session.repository.AssessmentSessionRepository;
import com.bimodalpsyche.backend.session.repository.AudioFileRepository;
import com.bimodalpsyche.backend.session.repository.SessionTurnRepository;
import com.bimodalpsyche.backend.user.model.User;
import org.springframework.stereotype.Service;
import org.springframework.transaction.support.TransactionSynchronization;
import org.springframework.transaction.support.TransactionSynchronizationManager;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.web.multipart.MultipartFile;

import java.time.LocalDateTime;

@Service
public class AudioTurnService {
    private final AssessmentSessionRepository sessionRepository;
    private final SessionTurnRepository turnRepository;
    private final AudioFileRepository audioFileRepository;
    private final SessionQuestionService sessionQuestionService;
    private final AudioStorageService audioStorageService;

    public AudioTurnService(
            AssessmentSessionRepository sessionRepository,
            SessionTurnRepository turnRepository,
            AudioFileRepository audioFileRepository,
            SessionQuestionService sessionQuestionService,
            AudioStorageService audioStorageService) {
        this.sessionRepository = sessionRepository;
        this.turnRepository = turnRepository;
        this.audioFileRepository = audioFileRepository;
        this.sessionQuestionService = sessionQuestionService;
        this.audioStorageService = audioStorageService;
    }

    @Transactional
    public AudioTurnResponse submitAudioTurn(
            Long sessionId, String email, Long questionVariantId, MultipartFile upload) {
        AssessmentSession session = sessionRepository.findByIdAndUser_EmailForUpdate(sessionId, email)
                .orElseThrow(() -> new SessionNotFoundException("Session not found"));
        if (session.getStatus() != SessionStatus.IN_PROGRESS) {
            throw new InvalidSessionStateException("Audio can only be added to an in-progress session.");
        }

        QuestionVariant question = sessionQuestionService.requireExpectedQuestion(session, questionVariantId);
        AudioStorageService.StoredAudio stored = null;
        try {
            LocalDateTime now = LocalDateTime.now();
            SessionTurn turn = new SessionTurn();
            turn.setSession(session);
            turn.setTurnNumber(session.getCurrentTurnNumber() + 1);
            turn.setQuestionVariant(question);
            turn.setQuestionTextSnapshot(question.getQuestionText());
            turn.setResponseType(ResponseType.AUDIO);
            turn.setSentimentScore(null);
            turn.setCrisisDetected(false);
            turn = turnRepository.saveAndFlush(turn);

            stored = audioStorageService.store(upload, sessionId, turn.getId());
            AudioStorageService.StoredAudio storedAudio = stored;
            if (TransactionSynchronizationManager.isSynchronizationActive()) {
                TransactionSynchronizationManager.registerSynchronization(new TransactionSynchronization() {
                    @Override
                    public void afterCompletion(int status) {
                        if (status != STATUS_COMMITTED) {
                            audioStorageService.delete(storedAudio.storagePath());
                        }
                    }
                });
            }

            AudioFile audio = new AudioFile();
            audio.setTurn(turn);
            audio.setStoragePath(stored.storagePath());
            audio.setOriginalFilename(stored.originalFilename());
            audio.setMimeType(stored.mimeType());
            audio.setFileSize(stored.sizeBytes());
            audio.setEncryptionVersion("AES-256-GCM-v1");
            audio.setEncryptedAt(now);
            audioFileRepository.save(audio);

            session.setCurrentTurnNumber(turn.getTurnNumber());
            session.setCurrentQuestionVariant(question);
            session.setLastActivityAt(now);
            sessionRepository.save(session);
            return new AudioTurnResponse(turn.getId(), turn.getTurnNumber(), question.getQuestionText(),
                    audio.getId(), audio.getOriginalFilename(), audio.getMimeType(), audio.getFileSize(),
                    "PENDING_MODEL_ANALYSIS", now);
        } catch (RuntimeException ex) {
            if (stored != null) {
                audioStorageService.delete(stored.storagePath());
            }
            throw ex;
        }
    }

    public record AudioTurnResponse(
            Long turnId,
            int turnNumber,
            String questionText,
            Long audioId,
            String filename,
            String mimeType,
            Long sizeBytes,
            String analysisStatus,
            LocalDateTime createdAt) {
    }
}
