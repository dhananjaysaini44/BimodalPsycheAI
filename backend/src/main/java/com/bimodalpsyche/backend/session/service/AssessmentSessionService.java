package com.bimodalpsyche.backend.session.service;

import com.bimodalpsyche.backend.auth.exception.LegalConsentRequiredException;
import com.bimodalpsyche.backend.session.exception.ActiveSessionExistsException;
import com.bimodalpsyche.backend.session.exception.InvalidSessionStateException;
import com.bimodalpsyche.backend.session.exception.SessionNotFoundException;
import com.bimodalpsyche.backend.session.model.AssessmentSession;
import com.bimodalpsyche.backend.session.model.SessionStatus;
import com.bimodalpsyche.backend.session.repository.AssessmentSessionRepository;
import com.bimodalpsyche.backend.user.exception.HistoryRequiredException;
import com.bimodalpsyche.backend.user.model.User;
import com.bimodalpsyche.backend.user.repository.UserRepository;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.security.core.userdetails.UsernameNotFoundException;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.time.LocalDateTime;
import java.util.List;
import java.util.Objects;
import java.util.Optional;

@Service
public class AssessmentSessionService {
    private static final List<SessionStatus> UNFINISHED_STATUSES =
            List.of(SessionStatus.DRAFT, SessionStatus.IN_PROGRESS, SessionStatus.PAUSED);

    private final AssessmentSessionRepository sessionRepository;
    private final UserRepository userRepository;
    private final String termsVersion;
    private final String privacyPolicyVersion;

    public AssessmentSessionService(
            AssessmentSessionRepository sessionRepository,
            UserRepository userRepository,
            @Value("${app.legal.terms-version}") String termsVersion,
            @Value("${app.legal.privacy-policy-version}") String privacyPolicyVersion) {
        this.sessionRepository = sessionRepository;
        this.userRepository = userRepository;
        this.termsVersion = termsVersion;
        this.privacyPolicyVersion = privacyPolicyVersion;
    }

    @Transactional
    public SessionResponse start(String email) {
        User user = lockUser(email);
        requireCurrentConsent(user);
        requireCompletedHistories(user);

        if (findUnfinished(user.getId()).isPresent()) {
            throw new ActiveSessionExistsException(
                    "An unfinished session already exists. Resume or abandon it before starting another.");
        }

        LocalDateTime now = LocalDateTime.now();
        AssessmentSession session = new AssessmentSession();
        session.setUser(user);
        session.setStatus(SessionStatus.IN_PROGRESS);
        session.setCurrentTurnNumber(0);
        session.setStartedAt(now);
        session.setLastActivityAt(now);

        return SessionResponse.from(sessionRepository.saveAndFlush(session));
    }

    @Transactional(readOnly = true)
    public Optional<SessionResponse> findUnfinished(String email) {
        User user = userRepository.findByEmail(email)
                .orElseThrow(() -> new UsernameNotFoundException("User not found"));
        return findUnfinished(user.getId()).map(SessionResponse::from);
    }

    @Transactional(readOnly = true)
    public SessionResponse get(Long sessionId, String email) {
        User user = userRepository.findByEmail(email)
                .orElseThrow(() -> new UsernameNotFoundException("User not found"));
        return getOwnedSession(sessionId, user.getId()).map(SessionResponse::from)
                .orElseThrow(() -> new SessionNotFoundException("Session not found"));
    }

    @Transactional
    public SessionResponse resume(Long sessionId, String email) {
        User user = lockUser(email);
        requireCurrentConsent(user);

        AssessmentSession session = getOwnedSession(sessionId, user.getId())
                .orElseThrow(() -> new SessionNotFoundException("Session not found"));

        if (session.getStatus() == SessionStatus.IN_PROGRESS) {
            session.setLastActivityAt(LocalDateTime.now());
            return SessionResponse.from(sessionRepository.saveAndFlush(session));
        }
        if (session.getStatus() != SessionStatus.DRAFT && session.getStatus() != SessionStatus.PAUSED) {
            throw new InvalidSessionStateException("Only an unfinished session can be resumed.");
        }

        LocalDateTime now = LocalDateTime.now();
        session.setStatus(SessionStatus.IN_PROGRESS);
        if (session.getStartedAt() == null) session.setStartedAt(now);
        session.setLastActivityAt(now);
        return SessionResponse.from(sessionRepository.saveAndFlush(session));
    }

    @Transactional
    public SessionResponse complete(Long sessionId, String email) {
        User user = lockUser(email);
        AssessmentSession session = getOwnedSession(sessionId, user.getId())
                .orElseThrow(() -> new SessionNotFoundException("Session not found"));

        if (session.getStatus() != SessionStatus.IN_PROGRESS && session.getStatus() != SessionStatus.PAUSED) {
            throw new InvalidSessionStateException("Only an in-progress session can be completed.");
        }

        LocalDateTime now = LocalDateTime.now();
        session.setStatus(SessionStatus.COMPLETED);
        session.setCompletedAt(now);
        session.setLastActivityAt(now);
        return SessionResponse.from(sessionRepository.saveAndFlush(session));
    }

    @Transactional
    public SessionResponse abandon(Long sessionId, String email) {
        User user = lockUser(email);
        AssessmentSession session = getOwnedSession(sessionId, user.getId())
                .orElseThrow(() -> new SessionNotFoundException("Session not found"));

        if (!UNFINISHED_STATUSES.contains(session.getStatus())) {
            throw new InvalidSessionStateException("Only an unfinished session can be abandoned.");
        }

        session.setStatus(SessionStatus.ABANDONED);
        session.setLastActivityAt(LocalDateTime.now());
        return SessionResponse.from(sessionRepository.saveAndFlush(session));
    }

    private User lockUser(String email) {
        return userRepository.findByEmailForUpdate(email)
                .orElseThrow(() -> new UsernameNotFoundException("User not found"));
    }

    private void requireCurrentConsent(User user) {
        boolean hasCurrentTerms = user.getTermsAcceptedAt() != null
                && Objects.equals(user.getTermsVersion(), termsVersion);
        boolean hasCurrentPrivacyPolicy = user.getPrivacyPolicyAcceptedAt() != null
                && Objects.equals(user.getPrivacyPolicyVersion(), privacyPolicyVersion);

        if (!hasCurrentTerms || !hasCurrentPrivacyPolicy) {
            throw new LegalConsentRequiredException(
                    "Accept the current Terms of Service and Privacy Policy before starting or resuming a session.");
        }
    }

    private void requireCompletedHistories(User user) {
        if (!user.isMedicalProfileCompleted() || !user.isFamilyHistoryCompleted()) {
            throw new HistoryRequiredException(
                    "Complete both medical history and family history before starting an assessment session.");
        }
    }

    private Optional<AssessmentSession> findUnfinished(Long userId) {
        return sessionRepository.findFirstByUser_IdAndStatusInOrderByCreatedAtDesc(
                userId, UNFINISHED_STATUSES);
    }

    private Optional<AssessmentSession> getOwnedSession(Long sessionId, Long userId) {
        return sessionRepository.findByIdAndUser_Id(sessionId, userId);
    }

    public record SessionResponse(
            Long id,
            SessionStatus status,
            int currentTurnNumber,
            LocalDateTime startedAt,
            LocalDateTime lastActivityAt,
            LocalDateTime completedAt,
            LocalDateTime createdAt) {
        private static SessionResponse from(AssessmentSession session) {
            return new SessionResponse(
                    session.getId(),
                    session.getStatus(),
                    session.getCurrentTurnNumber(),
                    session.getStartedAt(),
                    session.getLastActivityAt(),
                    session.getCompletedAt(),
                    session.getCreatedAt());
        }
    }
}
