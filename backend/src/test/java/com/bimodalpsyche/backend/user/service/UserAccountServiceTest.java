package com.bimodalpsyche.backend.user.service;

import com.bimodalpsyche.backend.assessment.repository.AssessmentAnalysisResultRepository;
import com.bimodalpsyche.backend.assessment.repository.AssessmentRepository;
import com.bimodalpsyche.backend.auth.service.OtpService;
import com.bimodalpsyche.backend.auth.repository.OtpVerificationRepository;
import com.bimodalpsyche.backend.auth.repository.RefreshTokenRepository;
import com.bimodalpsyche.backend.session.repository.AssessmentSessionRepository;
import com.bimodalpsyche.backend.session.repository.AudioFileRepository;
import com.bimodalpsyche.backend.session.repository.CrisisEventRepository;
import com.bimodalpsyche.backend.session.repository.SessionTurnRepository;
import com.bimodalpsyche.backend.session.repository.TurnAnalysisRepository;
import com.bimodalpsyche.backend.session.service.AudioStorageService;
import com.bimodalpsyche.backend.user.dto.UpdateProfileRequest;
import com.bimodalpsyche.backend.user.model.EmploymentType;
import com.bimodalpsyche.backend.user.model.Gender;
import com.bimodalpsyche.backend.user.model.MaritalStatus;
import com.bimodalpsyche.backend.user.model.User;
import com.bimodalpsyche.backend.user.repository.FamilyHistoryRepository;
import com.bimodalpsyche.backend.user.repository.MedicalHistoryRepository;
import com.bimodalpsyche.backend.user.repository.UserRepository;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.springframework.security.authentication.BadCredentialsException;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.test.util.ReflectionTestUtils;

import java.time.LocalDate;
import java.time.LocalDateTime;
import java.util.List;
import java.util.Optional;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

@ExtendWith(MockitoExtension.class)
class UserAccountServiceTest {
    private static final String EMAIL = "person@example.com";

    @Mock
    private UserRepository userRepository;
    @Mock
    private PasswordEncoder passwordEncoder;
    @Mock
    private OtpService otpService;
    @Mock
    private AssessmentSessionRepository sessionRepository;
    @Mock
    private SessionTurnRepository turnRepository;
    @Mock
    private AudioFileRepository audioFileRepository;
    @Mock
    private TurnAnalysisRepository turnAnalysisRepository;
    @Mock
    private CrisisEventRepository crisisEventRepository;
    @Mock
    private AssessmentRepository assessmentRepository;
    @Mock
    private AssessmentAnalysisResultRepository assessmentAnalysisResultRepository;
    @Mock
    private MedicalHistoryRepository medicalHistoryRepository;
    @Mock
    private FamilyHistoryRepository familyHistoryRepository;
    @Mock
    private RefreshTokenRepository refreshTokenRepository;
    @Mock
    private OtpVerificationRepository otpVerificationRepository;
    @Mock
    private AudioStorageService audioStorageService;

    private UserAccountService service;

    @BeforeEach
    void setUp() {
        service = new UserAccountService(
                userRepository, passwordEncoder, otpService, sessionRepository, turnRepository,
                audioFileRepository, turnAnalysisRepository, crisisEventRepository,
                assessmentRepository, assessmentAnalysisResultRepository,
                medicalHistoryRepository, familyHistoryRepository, refreshTokenRepository,
                otpVerificationRepository, audioStorageService);
    }

    @Test
    void profileUpdateLeavesEmailAndServerManagedConsentUnchanged() throws Exception {
        User user = new User();
        user.setId(17L);
        user.setEmail(EMAIL);
        user.setPassword("encoded-password");
        user.setTermsVersion("terms-v1");
        user.setTermsAcceptedAt(LocalDateTime.parse("2026-01-01T10:00:00"));
        user.setEmailVerified(true);
        when(userRepository.findByEmailForUpdate(EMAIL)).thenReturn(Optional.of(user));
        when(userRepository.save(user)).thenReturn(user);
        UpdateProfileRequest request = new UpdateProfileRequest();
        ReflectionTestUtils.setField(request, "firstName", "New");
        ReflectionTestUtils.setField(request, "lastName", "Name");
        ReflectionTestUtils.setField(request, "dateOfBirth", LocalDate.of(1990, 1, 1));
        ReflectionTestUtils.setField(request, "gender", Gender.OTHERS);
        ReflectionTestUtils.setField(request, "maritalStatus", MaritalStatus.SINGLE);
        ReflectionTestUtils.setField(request, "jobTitle", "Engineer");
        ReflectionTestUtils.setField(request, "employmentType", EmploymentType.EMPLOYED);
        ReflectionTestUtils.setField(request, "phoneNumber", "+12345678901");
        ReflectionTestUtils.setField(request, "workingHoursPerWeek", 40);

        UserAccountService.ProfileResponse result = service.updateProfile(EMAIL, request);

        assertEquals(EMAIL, result.email());
        assertEquals("New", result.firstName());
        assertEquals(LocalDate.of(1990, 1, 1), result.dateOfBirth());
        assertEquals("terms-v1", user.getTermsVersion());
        assertEquals(LocalDateTime.parse("2026-01-01T10:00:00"), user.getTermsAcceptedAt());
        assertTrue(user.isEmailVerified());
    }

    @Test
    void accountDeletionRequiresCorrectPasswordBeforeRemovingData() {
        User user = new User();
        user.setId(17L);
        user.setEmail(EMAIL);
        user.setPassword("encoded-password");
        when(userRepository.findByEmailForUpdate(EMAIL)).thenReturn(Optional.of(user));
        when(passwordEncoder.matches("incorrect", user.getPassword())).thenReturn(false);

        assertThrows(BadCredentialsException.class,
                () -> service.deleteAccount(EMAIL, "incorrect", "123456"));

        verify(userRepository, never()).delete(user);
        verify(refreshTokenRepository, never()).deleteByUser(user);
    }

    @Test
    void accountDeletionRemovesRefreshSessionAndOtpWithTheUser() {
        User user = new User();
        user.setId(17L);
        user.setEmail(EMAIL);
        user.setPassword("encoded-password");
        when(userRepository.findByEmailForUpdate(EMAIL)).thenReturn(Optional.of(user));
        when(passwordEncoder.matches("correct", user.getPassword())).thenReturn(true);
        when(sessionRepository.findAllByUser_Id(user.getId())).thenReturn(List.of());
        when(medicalHistoryRepository.findByUser(user)).thenReturn(Optional.empty());
        when(familyHistoryRepository.findByUser(user)).thenReturn(Optional.empty());

        service.deleteAccount(EMAIL, "correct", "123456");

        verify(otpService).verifyAndConsumeAccountActionOtp(EMAIL, "123456");
        verify(refreshTokenRepository).deleteByUser(user);
        verify(otpVerificationRepository).deleteByEmail(EMAIL);
        verify(userRepository).delete(user);
    }
}
