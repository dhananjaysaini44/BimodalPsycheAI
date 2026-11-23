package com.bimodalpsyche.backend.assessment.service;

import com.bimodalpsyche.backend.assessment.model.AnalysisModality;
import com.bimodalpsyche.backend.assessment.model.Assessment;
import com.bimodalpsyche.backend.assessment.model.AssessmentAnalysisResult;
import com.bimodalpsyche.backend.assessment.model.Severity;
import com.bimodalpsyche.backend.assessment.repository.AssessmentAnalysisResultRepository;
import com.bimodalpsyche.backend.assessment.repository.AssessmentRepository;
import com.bimodalpsyche.backend.question.model.QuestionBankVersion;
import com.bimodalpsyche.backend.session.exception.InvalidSessionStateException;
import com.bimodalpsyche.backend.session.exception.MlAnalysisUnavailableException;
import com.bimodalpsyche.backend.session.exception.SessionNotFoundException;
import com.bimodalpsyche.backend.session.model.AssessmentSession;
import com.bimodalpsyche.backend.session.model.AudioFile;
import com.bimodalpsyche.backend.session.model.ResponseType;
import com.bimodalpsyche.backend.session.model.SessionTurn;
import com.bimodalpsyche.backend.session.repository.AssessmentSessionRepository;
import com.bimodalpsyche.backend.session.repository.AudioFileRepository;
import com.bimodalpsyche.backend.session.repository.SessionTurnRepository;
import com.bimodalpsyche.backend.session.service.AudioStorageService;
import com.bimodalpsyche.backend.session.service.MlAnalysisClient;
import com.bimodalpsyche.backend.session.service.TextEncryptionService;
import com.bimodalpsyche.backend.user.model.User;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.List;

@Service
public class AssessmentAnalysisService {
    private final AssessmentSessionRepository sessionRepository;
    private final SessionTurnRepository turnRepository;
    private final AudioFileRepository audioFileRepository;
    private final AssessmentRepository assessmentRepository;
    private final AssessmentAnalysisResultRepository analysisResultRepository;
    private final MlAnalysisClient mlAnalysisClient;
    private final AudioStorageService audioStorageService;
    private final TextEncryptionService encryptionService;

    public AssessmentAnalysisService(
            AssessmentSessionRepository sessionRepository,
            SessionTurnRepository turnRepository,
            AudioFileRepository audioFileRepository,
            AssessmentRepository assessmentRepository,
            AssessmentAnalysisResultRepository analysisResultRepository,
            MlAnalysisClient mlAnalysisClient,
            AudioStorageService audioStorageService,
            TextEncryptionService encryptionService) {
        this.sessionRepository = sessionRepository;
        this.turnRepository = turnRepository;
        this.audioFileRepository = audioFileRepository;
        this.assessmentRepository = assessmentRepository;
        this.analysisResultRepository = analysisResultRepository;
        this.mlAnalysisClient = mlAnalysisClient;
        this.audioStorageService = audioStorageService;
        this.encryptionService = encryptionService;
    }

    @Transactional
    public AssessmentResponse analyze(Long sessionId, String email) {
        AssessmentSession session = sessionRepository.findByIdAndUser_EmailForUpdate(sessionId, email)
                .orElseThrow(() -> new SessionNotFoundException("Session not found"));
        if (session.getStatus() != com.bimodalpsyche.backend.session.model.SessionStatus.COMPLETED) {
            throw new InvalidSessionStateException("Complete the session before generating assessment results.");
        }
        return assessmentRepository.findBySession_Id(sessionId)
                .map(this::response)
                .orElseGet(() -> analyzeAndSave(session));
    }

    @Transactional(readOnly = true)
    public AssessmentResponse get(Long sessionId, String email) {
        sessionRepository.findByIdAndUser_Email(sessionId, email)
                .orElseThrow(() -> new SessionNotFoundException("Session not found"));
        return assessmentRepository.findBySession_Id(sessionId)
                .map(this::response)
                .orElseThrow(() -> new SessionNotFoundException("Assessment result not found"));
    }

    private AssessmentResponse analyzeAndSave(AssessmentSession session) {
        List<SessionTurn> turns = turnRepository.findAllBySession_IdOrderByTurnNumberAsc(session.getId());
        List<SessionTurn> textTurns = turns.stream()
                .filter(turn -> turn.getResponseType() == ResponseType.TEXT)
                .toList();
        List<AudioFile> audioFiles = audioFileRepository
                .findAllByTurn_Session_IdOrderByTurn_TurnNumberAsc(session.getId());
        if (textTurns.isEmpty() && audioFiles.isEmpty()) {
            throw new InvalidSessionStateException("The completed session has no analyzable text or audio responses.");
        }

        MlAnalysisClient.ModelAnalysisResult textResult = null;
        MlAnalysisClient.ModelAnalysisResult voiceResult = null;
        MlAnalysisClient.ModelAnalysisResult finalResult;
        try {
            if (!textTurns.isEmpty()) {
                String text = textTurns.stream()
                        .map(turn -> turn.getQuestionTextSnapshot() + "\n" +
                                encryptionService.decrypt(turn.getResponseTextEncrypted()))
                        .reduce((first, second) -> first + "\n\n" + second)
                        .orElseThrow();
                textResult = mlAnalysisClient.analyzeText(text);
            }
            if (!audioFiles.isEmpty()) {
                List<MlAnalysisClient.ModelAnalysisResult> perTurn = new ArrayList<>();
                for (AudioFile file : audioFiles) {
                    byte[] audio = audioStorageService.read(file.getStoragePath());
                    perTurn.add(mlAnalysisClient.analyzeVoice(
                            audio, file.getMimeType(), file.getOriginalFilename()));
                }
                voiceResult = aggregateVoiceResults(perTurn);
            }
            if (textResult != null && voiceResult != null) {
                finalResult = mlAnalysisClient.fuse(textResult, voiceResult);
            } else {
                finalResult = textResult != null ? textResult : voiceResult;
            }
        } catch (RuntimeException ex) {
            throw new MlAnalysisUnavailableException(
                    "Assessment analysis is unavailable. No assessment result was saved; try again when the ML service is ready.", ex);
        }

        User user = session.getUser();
        Assessment assessment = new Assessment();
        assessment.setSession(session);
        assessment.setUser(user);
        assessment.setScore(finalResult.score());
        assessment.setProbability(finalResult.probability());
        assessment.setSeverity(finalResult.severityValue());
        assessment.setExplanationEncrypted(finalResult.explanation() == null || finalResult.explanation().isBlank()
                ? null : encryptionService.encrypt(finalResult.explanation()));
        assessment.setCompletedAt(LocalDateTime.now());
        assessment = assessmentRepository.saveAndFlush(assessment);

        if (textResult != null) {
            saveAnalysisResult(assessment, AnalysisModality.TEXT, textResult);
        }
        if (voiceResult != null) {
            saveAnalysisResult(assessment, AnalysisModality.VOICE, voiceResult);
        }
        if (textResult != null && voiceResult != null) {
            saveAnalysisResult(assessment, AnalysisModality.FUSION, finalResult);
        }
        analysisResultRepository.flush();
        return response(assessment);
    }

    private MlAnalysisClient.ModelAnalysisResult aggregateVoiceResults(
            List<MlAnalysisClient.ModelAnalysisResult> results) {
        float score = (float) results.stream().mapToDouble(MlAnalysisClient.ModelAnalysisResult::score).average()
                .orElseThrow();
        float probability = (float) results.stream()
                .mapToDouble(MlAnalysisClient.ModelAnalysisResult::probability).average().orElseThrow();
        Severity severity = results.stream().map(MlAnalysisClient.ModelAnalysisResult::severityValue)
                .max(java.util.Comparator.comparingInt(AssessmentAnalysisService::severityRank)).orElseThrow();
        return new MlAnalysisClient.ModelAnalysisResult(score, probability, severity.name(), null);
    }

    private void saveAnalysisResult(
            Assessment assessment, AnalysisModality modality, MlAnalysisClient.ModelAnalysisResult result) {
        AssessmentAnalysisResult row = new AssessmentAnalysisResult();
        row.setAssessment(assessment);
        row.setModality(modality);
        row.setScore(result.score());
        row.setProbability(result.probability());
        analysisResultRepository.save(row);
    }

    private AssessmentResponse response(Assessment assessment) {
        List<AnalysisResponse> analyses = analysisResultRepository
                .findAllByAssessment_IdOrderByIdAsc(assessment.getId()).stream()
                .map(row -> new AnalysisResponse(row.getModality(), row.getScore(), row.getProbability()))
                .toList();
        String explanation = assessment.getExplanationEncrypted() == null ? null
                : encryptionService.decrypt(assessment.getExplanationEncrypted());
        return new AssessmentResponse(assessment.getId(), assessment.getSession().getId(), assessment.getScore(),
                assessment.getProbability(), assessment.getSeverity(), explanation, analyses,
                assessment.getCompletedAt());
    }

    private static int severityRank(Severity severity) {
        return switch (severity) {
            case LOW -> 0;
            case MODERATE -> 1;
            case HIGH -> 2;
        };
    }

    public record AssessmentResponse(
            Long id,
            Long sessionId,
            Float score,
            Float probability,
            Severity severity,
            String explanation,
            List<AnalysisResponse> analyses,
            LocalDateTime completedAt) {
    }

    public record AnalysisResponse(AnalysisModality modality, Float score, Float probability) {
    }
}
