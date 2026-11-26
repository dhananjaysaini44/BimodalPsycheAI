package com.bimodalpsyche.backend.user.service;

import com.bimodalpsyche.backend.assessment.model.Assessment;
import com.bimodalpsyche.backend.assessment.repository.AssessmentAnalysisResultRepository;
import com.bimodalpsyche.backend.assessment.repository.AssessmentRepository;
import com.bimodalpsyche.backend.auth.repository.OtpVerificationRepository;
import com.bimodalpsyche.backend.auth.repository.RefreshTokenRepository;
import com.bimodalpsyche.backend.auth.service.OtpService;
import com.bimodalpsyche.backend.session.model.AssessmentSession;
import com.bimodalpsyche.backend.session.model.AudioFile;
import com.bimodalpsyche.backend.session.model.SessionTurn;
import com.bimodalpsyche.backend.session.repository.AssessmentSessionRepository;
import com.bimodalpsyche.backend.session.repository.AudioFileRepository;
import com.bimodalpsyche.backend.session.repository.CrisisEventRepository;
import com.bimodalpsyche.backend.session.repository.SessionTurnRepository;
import com.bimodalpsyche.backend.session.repository.TurnAnalysisRepository;
import com.bimodalpsyche.backend.session.service.AudioStorageService;
import com.bimodalpsyche.backend.user.dto.ChangePasswordRequest;
import com.bimodalpsyche.backend.user.dto.UpdateProfileRequest;
import com.bimodalpsyche.backend.user.model.EmploymentType;
import com.bimodalpsyche.backend.user.model.Gender;
import com.bimodalpsyche.backend.user.model.MaritalStatus;
import com.bimodalpsyche.backend.user.model.User;
import com.bimodalpsyche.backend.user.repository.FamilyHistoryRepository;
import com.bimodalpsyche.backend.user.repository.MedicalHistoryRepository;
import com.bimodalpsyche.backend.user.repository.UserRepository;
import lombok.extern.slf4j.Slf4j;
import org.springframework.security.authentication.BadCredentialsException;
import org.springframework.security.core.userdetails.UsernameNotFoundException;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.transaction.support.TransactionSynchronization;
import org.springframework.transaction.support.TransactionSynchronizationManager;

import java.time.LocalDate;
import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.List;

@Service
@Slf4j
public class UserAccountService {
    private final UserRepository userRepository;
    private final PasswordEncoder passwordEncoder;
    private final OtpService otpService;
    private final AssessmentSessionRepository sessionRepository;
    private final SessionTurnRepository turnRepository;
    private final AudioFileRepository audioFileRepository;
    private final TurnAnalysisRepository turnAnalysisRepository;
    private final CrisisEventRepository crisisEventRepository;
    private final AssessmentRepository assessmentRepository;
    private final AssessmentAnalysisResultRepository assessmentAnalysisResultRepository;
    private final MedicalHistoryRepository medicalHistoryRepository;
    private final FamilyHistoryRepository familyHistoryRepository;
    private final RefreshTokenRepository refreshTokenRepository;
    private final OtpVerificationRepository otpVerificationRepository;
    private final AudioStorageService audioStorageService;

    public UserAccountService(
            UserRepository userRepository,
            PasswordEncoder passwordEncoder,
            OtpService otpService,
            AssessmentSessionRepository sessionRepository,
            SessionTurnRepository turnRepository,
            AudioFileRepository audioFileRepository,
            TurnAnalysisRepository turnAnalysisRepository,
            CrisisEventRepository crisisEventRepository,
            AssessmentRepository assessmentRepository,
            AssessmentAnalysisResultRepository assessmentAnalysisResultRepository,
            MedicalHistoryRepository medicalHistoryRepository,
            FamilyHistoryRepository familyHistoryRepository,
            RefreshTokenRepository refreshTokenRepository,
            OtpVerificationRepository otpVerificationRepository,
            AudioStorageService audioStorageService) {
        this.userRepository = userRepository;
        this.passwordEncoder = passwordEncoder;
        this.otpService = otpService;
        this.sessionRepository = sessionRepository;
        this.turnRepository = turnRepository;
        this.audioFileRepository = audioFileRepository;
        this.turnAnalysisRepository = turnAnalysisRepository;
        this.crisisEventRepository = crisisEventRepository;
        this.assessmentRepository = assessmentRepository;
        this.assessmentAnalysisResultRepository = assessmentAnalysisResultRepository;
        this.medicalHistoryRepository = medicalHistoryRepository;
        this.familyHistoryRepository = familyHistoryRepository;
        this.refreshTokenRepository = refreshTokenRepository;
        this.otpVerificationRepository = otpVerificationRepository;
        this.audioStorageService = audioStorageService;
    }

    @Transactional(readOnly = true)
    public ProfileResponse getProfile(String email) {
        return ProfileResponse.from(findUser(email));
    }

    @Transactional
    public ProfileResponse updateProfile(String email, UpdateProfileRequest request) {
        User user = findUserForUpdate(email);
        user.setFirstName(request.getFirstName().trim());
        user.setLastName(trimToNull(request.getLastName()));
        user.setGender(request.getGender());
        user.setMaritalStatus(request.getMaritalStatus());
        user.setJobTitle(trimToNull(request.getJobTitle()));
        user.setEmploymentType(request.getEmploymentType());
        user.setPhoneNumber(trimToNull(request.getPhoneNumber()));
        user.setWorkingHoursPerWeek(request.getWorkingHoursPerWeek());
        user.setDateOfBirth(request.getDateOfBirth());
        return ProfileResponse.from(userRepository.save(user));
    }

    @Transactional
    public void changePassword(String email, ChangePasswordRequest request) {
        User user = findUserForUpdate(email);
        verifyPassword(user, request.getCurrentPassword());
        otpService.verifyAndConsumeAccountActionOtp(email, request.getOtp());
        otpVerificationRepository.deleteByEmail(email);
        user.setPassword(passwordEncoder.encode(request.getNewPassword()));
        userRepository.save(user);
        refreshTokenRepository.deleteByUser(user);
    }

    @Transactional
    public void deleteAccount(String email, String currentPassword, String otp) {
        User user = findUserForUpdate(email);
        verifyPassword(user, currentPassword);
        otpService.verifyAndConsumeAccountActionOtp(email, otp);
        otpVerificationRepository.deleteByEmail(email);

        List<AssessmentSession> sessions = sessionRepository.findAllByUser_Id(user.getId());
        List<Long> sessionIds = sessions.stream().map(AssessmentSession::getId).toList();
        List<SessionTurn> turns = new ArrayList<>();
        List<AudioFile> audioFiles = new ArrayList<>();

        for (Long sessionId : sessionIds) {
            turns.addAll(turnRepository.findAllBySession_IdOrderByTurnNumberAsc(sessionId));
            audioFiles.addAll(audioFileRepository.findAllByTurn_Session_IdOrderByTurn_TurnNumberAsc(sessionId));
            crisisEventRepository.deleteAllBySession_Id(sessionId);
        }

        List<Long> turnIds = turns.stream().map(SessionTurn::getId).toList();
        List<Assessment> assessments = sessionIds.isEmpty()
                ? List.of()
                : assessmentRepository.findAllBySession_IdIn(sessionIds);
        List<Long> assessmentIds = assessments.stream().map(Assessment::getId).toList();
        if (!assessmentIds.isEmpty()) {
            assessmentAnalysisResultRepository.deleteAllByAssessment_IdIn(assessmentIds);
            assessmentRepository.deleteAll(assessments);
        }

        audioFileRepository.deleteAll(audioFiles);
        if (!turnIds.isEmpty()) {
            turnAnalysisRepository.deleteAllByTurn_IdIn(turnIds);
            turnRepository.deleteAll(turns);
        }
        sessionRepository.deleteAll(sessions);

        medicalHistoryRepository.findByUser(user).ifPresent(medicalHistoryRepository::delete);
        familyHistoryRepository.findByUser(user).ifPresent(familyHistoryRepository::delete);
        refreshTokenRepository.deleteByUser(user);
        userRepository.delete(user);

        if (TransactionSynchronizationManager.isSynchronizationActive()) {
            TransactionSynchronizationManager.registerSynchronization(new TransactionSynchronization() {
                @Override
                public void afterCommit() {
                    deleteStoredAudio(sessionIds);
                }
            });
        } else {
            deleteStoredAudio(sessionIds);
        }
    }

    @Transactional
    public String sendAccountActionOtp(String email) {
        findUserForUpdate(email);
        return otpService.sendAccountActionOtp(email);
    }

    private void deleteStoredAudio(List<Long> sessionIds) {
        for (Long sessionId : sessionIds) {
            try {
                audioStorageService.deleteSessionAudio(sessionId);
            } catch (RuntimeException exception) {
                log.error("Could not remove stored audio during account deletion for session {}", sessionId, exception);
            }
        }
    }

    private User findUser(String email) {
        return userRepository.findByEmail(email)
                .orElseThrow(() -> new UsernameNotFoundException("Authenticated user no longer exists."));
    }

    private User findUserForUpdate(String email) {
        return userRepository.findByEmailForUpdate(email)
                .orElseThrow(() -> new UsernameNotFoundException("Authenticated user no longer exists."));
    }

    private void verifyPassword(User user, String currentPassword) {
        if (!passwordEncoder.matches(currentPassword, user.getPassword())) {
            throw new BadCredentialsException("Current password is incorrect.");
        }
    }

    private static String trimToNull(String value) {
        if (value == null) {
            return null;
        }
        String trimmed = value.trim();
        return trimmed.isEmpty() ? null : trimmed;
    }

    public record ProfileResponse(
            Long id,
            String email,
            String firstName,
            String lastName,
            LocalDate dateOfBirth,
            Gender gender,
            MaritalStatus maritalStatus,
            String jobTitle,
            EmploymentType employmentType,
            String phoneNumber,
            Integer workingHoursPerWeek,
            LocalDateTime createdAt,
            LocalDateTime updatedAt) {
        private static ProfileResponse from(User user) {
            return new ProfileResponse(
                    user.getId(), user.getEmail(), user.getFirstName(), user.getLastName(),
                    user.getDateOfBirth(), user.getGender(), user.getMaritalStatus(),
                    user.getJobTitle(), user.getEmploymentType(), user.getPhoneNumber(),
                    user.getWorkingHoursPerWeek(), user.getCreatedAt(), user.getUpdatedAt());
        }
    }
}
