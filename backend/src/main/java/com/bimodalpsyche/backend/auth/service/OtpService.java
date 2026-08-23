package com.bimodalpsyche.backend.auth.service;

import com.bimodalpsyche.backend.auth.dto.OtpVerificationRequest;
import com.bimodalpsyche.backend.auth.dto.SendOtpRequest;
import com.bimodalpsyche.backend.auth.exception.InvalidOtpException;
import com.bimodalpsyche.backend.auth.model.OtpVerification;
import com.bimodalpsyche.backend.auth.repository.OtpVerificationRepository;
import com.bimodalpsyche.backend.user.exception.UserAlreadyExistsException;
import com.bimodalpsyche.backend.user.model.User;
import com.bimodalpsyche.backend.user.repository.UserRepository;
import jakarta.transaction.Transactional;
import org.springframework.mail.SimpleMailMessage;
import org.springframework.mail.javamail.JavaMailSender;
import org.springframework.stereotype.Service;

import java.security.SecureRandom;
import java.time.LocalDateTime;

@Service
public class OtpService {

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

        User user=userRepository.findByEmail(email).orElse(null);

        if(user!=null && user.isEmailVerified()) throw new UserAlreadyExistsException("User already exists.");

        String otp=String.format("%06d",secureRandom.nextInt(1000000));
        OtpVerification verification=otpVerificationRepository.findByEmail(email).orElse(new OtpVerification());
        verification.setEmail(email);
        verification.setExpiryTime(LocalDateTime.now().plusMinutes(5));
        verification.setOtp(otp);
        verification.setVerified(false);

        SimpleMailMessage message=new SimpleMailMessage();
        message.setTo(email);
        message.setSubject("OTP Verification");
        message.setText("Your OTP is: " + otp + "\n\n" + "This OTP is valid for 5 minutes only.");
        otpVerificationRepository.save(verification);
        mailSender.send(message);
        return "Otp Sent Successfully";
    }

    public void verifyOtp(OtpVerificationRequest request){
        String email=request.getEmail();
        String otp=request.getOtp();

        OtpVerification verification=otpVerificationRepository.findByEmail(email)
                .orElseThrow(()->new InvalidOtpException("Invalid OTP"));

        if(verification.getExpiryTime().isBefore(LocalDateTime.now())){
            throw new InvalidOtpException("OTP has expired");
        }

        if (verification.getOtp()==null || !verification.getOtp().equals(otp))
            throw new InvalidOtpException("Invalid OTP");

        verification.setVerified(true);
        verification.setOtp(null);
        otpVerificationRepository.save(verification);
    }
}
