package com.bimodalpsyche.backend.auth.service;

import com.bimodalpsyche.backend.auth.exception.InvalidOtpException;
import com.bimodalpsyche.backend.auth.model.OtpVerification;
import com.bimodalpsyche.backend.auth.repository.OtpVerificationRepository;
import com.bimodalpsyche.backend.auth.security.TokenHashUtil;
import com.bimodalpsyche.backend.user.model.User;
import com.bimodalpsyche.backend.user.repository.UserRepository;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.ArgumentCaptor;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.springframework.mail.SimpleMailMessage;
import org.springframework.mail.javamail.JavaMailSender;

import java.time.LocalDateTime;
import java.util.Optional;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

@ExtendWith(MockitoExtension.class)
class OtpServiceAccountActionTest {
    private static final String EMAIL = "person@example.com";

    @Mock
    private UserRepository userRepository;
    @Mock
    private OtpVerificationRepository otpVerificationRepository;
    @Mock
    private JavaMailSender mailSender;

    private OtpService service;

    @BeforeEach
    void setUp() {
        service = new OtpService(userRepository, otpVerificationRepository, mailSender);
    }

    @Test
    void sendsAndHashesAnAccountActionCode() {
        when(userRepository.findByEmail(EMAIL)).thenReturn(Optional.of(new User()));
        when(otpVerificationRepository.findByEmailForUpdate(EMAIL)).thenReturn(Optional.empty());

        assertEquals("Account verification code sent", service.sendAccountActionOtp(EMAIL));

        ArgumentCaptor<OtpVerification> savedOtp = ArgumentCaptor.forClass(OtpVerification.class);
        ArgumentCaptor<SimpleMailMessage> sentEmail = ArgumentCaptor.forClass(SimpleMailMessage.class);
        verify(otpVerificationRepository).save(savedOtp.capture());
        verify(mailSender).send(sentEmail.capture());
        String code = sentEmail.getValue().getText().split("\\s+")[5];
        assertEquals("ACCOUNT_ACTION", savedOtp.getValue().getPurpose());
        assertEquals(TokenHashUtil.hash(code), savedOtp.getValue().getOtpHash());
        assertNotEquals(code, savedOtp.getValue().getOtpHash());
    }

    @Test
    void consumesAValidAccountActionCodeOnce() {
        OtpVerification verification = validAccountOtp("123456");
        when(otpVerificationRepository.findByEmailForUpdate(EMAIL)).thenReturn(Optional.of(verification));

        service.verifyAndConsumeAccountActionOtp(EMAIL, "123456");

        assertNull(verification.getOtpHash());
        assertNull(verification.getPurpose());
        verify(otpVerificationRepository).save(verification);
    }

    @Test
    void burnsTheCodeAfterFiveIncorrectAttempts() {
        OtpVerification verification = validAccountOtp("123456");
        when(otpVerificationRepository.findByEmailForUpdate(EMAIL)).thenReturn(Optional.of(verification));

        for (int attempt = 0; attempt < 5; attempt++) {
            assertThrows(InvalidOtpException.class,
                    () -> service.verifyAndConsumeAccountActionOtp(EMAIL, "654321"));
        }

        assertEquals(5, verification.getAttemptCount());
        assertNull(verification.getOtpHash());
        assertNull(verification.getPurpose());
        verify(otpVerificationRepository, org.mockito.Mockito.times(5)).save(verification);
    }

    private static OtpVerification validAccountOtp(String code) {
        OtpVerification verification = new OtpVerification();
        verification.setEmail(EMAIL);
        verification.setPurpose("ACCOUNT_ACTION");
        verification.setOtpHash(TokenHashUtil.hash(code));
        verification.setExpiryTime(LocalDateTime.now().plusMinutes(5));
        verification.setAttemptCount(0);
        return verification;
    }
}
