package com.bimodalpsyche.backend.session.service;

import com.bimodalpsyche.backend.session.model.AssessmentSession;
import com.bimodalpsyche.backend.session.model.SessionStatus;
import com.bimodalpsyche.backend.session.repository.AssessmentSessionRepository;
import com.bimodalpsyche.backend.user.exception.HistoryRequiredException;
import com.bimodalpsyche.backend.user.model.User;
import com.bimodalpsyche.backend.user.repository.UserRepository;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import java.time.LocalDateTime;
import java.util.Optional;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

@ExtendWith(MockitoExtension.class)
class AssessmentSessionServiceTest {
    private static final String EMAIL = "person@example.com";

    @Mock
    private AssessmentSessionRepository sessionRepository;

    @Mock
    private UserRepository userRepository;

    private AssessmentSessionService service;

    @BeforeEach
    void setUp() {
        service = new AssessmentSessionService(sessionRepository, userRepository, "v1", "v1");
    }

    @Test
    void refusesToStartWhenEitherHistoryIsIncomplete() {
        User user = eligibleUser();
        user.setFamilyHistoryCompleted(false);
        when(userRepository.findByEmailForUpdate(EMAIL)).thenReturn(Optional.of(user));

        assertThrows(HistoryRequiredException.class, () -> service.start(EMAIL));

        verify(sessionRepository, never()).saveAndFlush(any());
    }

    @Test
    void startsSessionAfterBothHistoriesAreComplete() {
        User user = eligibleUser();
        AssessmentSession session = new AssessmentSession();
        session.setStatus(SessionStatus.IN_PROGRESS);
        when(userRepository.findByEmailForUpdate(EMAIL)).thenReturn(Optional.of(user));
        when(sessionRepository.findFirstByUser_IdAndStatusInOrderByCreatedAtDesc(
                eq(user.getId()), any())).thenReturn(Optional.empty());
        when(sessionRepository.saveAndFlush(any(AssessmentSession.class))).thenReturn(session);

        AssessmentSessionService.SessionResponse response = service.start(EMAIL);

        assertEquals(SessionStatus.IN_PROGRESS, response.status());
        verify(sessionRepository).saveAndFlush(any(AssessmentSession.class));
    }

    private User eligibleUser() {
        User user = new User();
        user.setId(17L);
        user.setEmail(EMAIL);
        user.setTermsVersion("v1");
        user.setTermsAcceptedAt(LocalDateTime.now());
        user.setPrivacyPolicyVersion("v1");
        user.setPrivacyPolicyAcceptedAt(LocalDateTime.now());
        user.setMedicalProfileCompleted(true);
        user.setFamilyHistoryCompleted(true);
        return user;
    }
}
