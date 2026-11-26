package com.bimodalpsyche.backend.user.service;

import com.bimodalpsyche.backend.auth.dto.RegisterRequest;
import com.bimodalpsyche.backend.auth.exception.EmailNotVerifiedException;
import com.bimodalpsyche.backend.auth.model.OtpVerification;
import com.bimodalpsyche.backend.auth.repository.OtpVerificationRepository;
import com.bimodalpsyche.backend.user.exception.UserAlreadyExistsException;
import com.bimodalpsyche.backend.user.model.Role;
import com.bimodalpsyche.backend.user.model.User;
import com.bimodalpsyche.backend.user.repository.UserRepository;
import jakarta.transaction.Transactional;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.security.core.userdetails.UserDetails;
import org.springframework.security.core.userdetails.UserDetailsService;
import org.springframework.security.core.userdetails.UsernameNotFoundException;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.stereotype.Service;

import java.time.LocalDateTime;

@Service
public class UserService implements UserDetailsService {

    private final PasswordEncoder passwordEncoder;
    private final UserRepository userRepository;
    private final OtpVerificationRepository otpVerificationRepository;
    private final String termsVersion;
    private final String privacyPolicyVersion;

    public UserService(
            PasswordEncoder passwordEncoder,
            UserRepository userRepository,
            OtpVerificationRepository otpVerificationRepository,
            @Value("${app.legal.terms-version}") String termsVersion,
            @Value("${app.legal.privacy-policy-version}") String privacyPolicyVersion) {
        this.otpVerificationRepository = otpVerificationRepository;
        this.passwordEncoder = passwordEncoder;
        this.userRepository = userRepository;
        this.termsVersion = termsVersion;
        this.privacyPolicyVersion = privacyPolicyVersion;
    }

    @Override
    public UserDetails loadUserByUsername(String username) throws UsernameNotFoundException {
        return userRepository.findByEmail(username).orElseThrow(() -> new UsernameNotFoundException("User not found"));
    }

    @Transactional
    public void register(RegisterRequest registerRequest) {
        if(userRepository.findByEmail(registerRequest.getEmail()).isPresent())
            throw new UserAlreadyExistsException("User already exists");

        OtpVerification verification=otpVerificationRepository.findByEmail(registerRequest.getEmail())
                .orElseThrow(()->new EmailNotVerifiedException("Email not verified"));

        // Verification is only honoured for a short window after the OTP was confirmed
        if(!verification.isVerified() || verification.getExpiryTime().isBefore(LocalDateTime.now()))
            throw new EmailNotVerifiedException("Email not verified");

        User user=new User();

        user.setEmail(registerRequest.getEmail());
        user.setPassword(passwordEncoder.encode(registerRequest.getPassword()));
        user.setFirstName(registerRequest.getFirstName());
        user.setLastName(registerRequest.getLastName());
        user.setDateOfBirth(registerRequest.getDateOfBirth());
        user.setGender(registerRequest.getGender());
        user.setMaritalStatus(registerRequest.getMaritalStatus());
        user.setJobTitle(registerRequest.getJobTitle());
        user.setEmploymentType(registerRequest.getEmploymentType());
        user.setPhoneNumber(registerRequest.getPhoneNumber());
        user.setWorkingHoursPerWeek(registerRequest.getWorkingHoursPerWeek());

        LocalDateTime consentAcceptedAt = LocalDateTime.now();
        user.setConsentAcceptedAt(consentAcceptedAt);
        user.setTermsVersion(termsVersion);
        user.setTermsAcceptedAt(consentAcceptedAt);
        user.setPrivacyPolicyVersion(privacyPolicyVersion);
        user.setPrivacyPolicyAcceptedAt(consentAcceptedAt);

        user.setEmailVerified(true);
        user.setRole(Role.ROLE_USER);

        user.setMedicalProfileCompleted(false);
        user.setFamilyHistoryCompleted(false);

        userRepository.save(user);
        // One verification = one registration
        otpVerificationRepository.delete(verification);
    }

}
