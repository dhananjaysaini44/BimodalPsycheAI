package com.bimodalpsyche.backend.auth.service;

import com.bimodalpsyche.backend.auth.dto.OtpVerificationRequest;
import com.bimodalpsyche.backend.auth.dto.SendOtpRequest;
import com.bimodalpsyche.backend.auth.exception.InvalidOtpException;
import com.bimodalpsyche.backend.auth.exception.TooManyRequestsException;
import com.bimodalpsyche.backend.auth.model.OtpVerification;
import com.bimodalpsyche.backend.auth.repository.OtpVerificationRepository;
import com.bimodalpsyche.backend.auth.security.TokenHashUtil;
import com.bimodalpsyche.backend.user.exception.UserAlreadyExistsException;
import com.bimodalpsyche.backend.user.repository.UserRepository;
import jakarta.transaction.Transactional;
import org.springframework.mail.SimpleMailMessage;
import org.springframework.mail.javamail.JavaMailSender;
import org.springframework.stereotype.Service;

import java.security.SecureRandom;
import java.time.LocalDateTime;

@Service
public class OtpService {

    private static final int OTP_VALIDITY_MINUTES = 5;
    private static final int RESEND_COOLDOWN_SECONDS = 60;
    private static final int MAX_VERIFY_ATTEMPTS = 5;
    // How long a verified email may be used to complete registration
    public static final int REGISTRATION_WINDOW_MINUTES = 15;

    private final UserRepository userRepository;
    private final OtpVerificationRepository otpVerificationRepository;
    private final JavaMailSender mailSender;
    private final SecureRandom secureRandom=new SecureRandom();

    public OtpService(UserRepository userRepository, OtpVerificationRepository otpVerificationRepository, JavaMailSender mailSender){
        this.userRepository=userRepository;
        this.otpVerificationRepository=otpVerificationRepository;
        this.mailSender=mailSender;
    }

    @Transactional
    public String sendOtp(SendOtpRequest request){
        String email= request.getEmail();

        if(userRepository.findByEmail(email).isPresent()) throw new UserAlreadyExistsException("User already exists.");

        OtpVerification verification=otpVerificationRepository.findByEmail(email).orElse(new OtpVerification());

        // expiryTime - validity = when the last OTP was sent; blocks mail flooding and OTP re-rolling
        if(verification.getExpiryTime()!=null && !verification.isVerified()
                && verification.getExpiryTime().minusMinutes(OTP_VALIDITY_MINUTES)
                .plusSeconds(RESEND_COOLDOWN_SECONDS).isAfter(LocalDateTime.now())){
            throw new TooManyRequestsException("Please wait before requesting another OTP");
        }

        String otp=String.format("%06d",secureRandom.nextInt(1000000));
        verification.setEmail(email);
        verification.setExpiryTime(LocalDateTime.now().plusMinutes(OTP_VALIDITY_MINUTES));
        verification.setOtpHash(TokenHashUtil.hash(otp));
        verification.setVerified(false);
        verification.setAttemptCount(0);

        SimpleMailMessage message=new SimpleMailMessage();
        message.setTo(email);
        message.setSubject("OTP Verification");
        message.setText("Your OTP is: " + otp + "\n\n" + "This OTP is valid for " + OTP_VALIDITY_MINUTES + " minutes only.");
        otpVerificationRepository.save(verification);
        mailSender.send(message);
        return "Otp Sent Successfully";
    }

    @Transactional(dontRollbackOn = InvalidOtpException.class)
    public void verifyOtp(OtpVerificationRequest request){
        String email=request.getEmail();
        String otp=request.getOtp();

        OtpVerification verification=otpVerificationRepository.findByEmail(email)
                .orElseThrow(()->new InvalidOtpException("Invalid OTP"));

        if(verification.getOtpHash()==null) throw new InvalidOtpException("Invalid OTP");

        if(verification.getExpiryTime().isBefore(LocalDateTime.now())){
            throw new InvalidOtpException("OTP has expired");
        }

        if (!TokenHashUtil.matches(otp, verification.getOtpHash())) {
            verification.setAttemptCount(verification.getAttemptCount()+1);
            // Too many wrong guesses burns the OTP so it can't be brute-forced
            if(verification.getAttemptCount()>=MAX_VERIFY_ATTEMPTS){
                verification.setOtpHash(null);
                otpVerificationRepository.save(verification);
                throw new InvalidOtpException("Too many failed attempts. Please request a new OTP");
            }
            otpVerificationRepository.save(verification);
            throw new InvalidOtpException("Invalid OTP");
        }

        verification.setVerified(true);
        verification.setOtpHash(null);
        verification.setExpiryTime(LocalDateTime.now().plusMinutes(REGISTRATION_WINDOW_MINUTES));
        otpVerificationRepository.save(verification);
    }
}
