package com.bimodalpsyche.backend.user.service;

import com.bimodalpsyche.backend.auth.dto.RegisterRequest;
import com.bimodalpsyche.backend.auth.exception.EmailNotVerifiedException;
import com.bimodalpsyche.backend.auth.model.OtpVerification;
import com.bimodalpsyche.backend.auth.repository.OtpVerificationRepository;
import com.bimodalpsyche.backend.user.exception.UserAlreadyExistsException;
import com.bimodalpsyche.backend.user.model.Role;
import com.bimodalpsyche.backend.user.model.User;
import com.bimodalpsyche.backend.user.repository.UserRepository;
import jakarta.validation.Valid;
import org.springframework.security.core.userdetails.UserDetails;
import org.springframework.security.core.userdetails.UserDetailsService;
import org.springframework.security.core.userdetails.UsernameNotFoundException;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.stereotype.Service;

@Service
public class UserService implements UserDetailsService {

    private PasswordEncoder passwordEncoder;
    private UserRepository userRepository;
    private OtpVerificationRepository otpVerificationRepository;

    public UserService(PasswordEncoder passwordEncoder, UserRepository userRepository, OtpVerificationRepository otpVerificationRepository) {
        this.otpVerificationRepository = otpVerificationRepository;
        this.passwordEncoder = passwordEncoder;
        this.userRepository = userRepository;
    }

    @Override
    public UserDetails loadUserByUsername(String username) throws UsernameNotFoundException {
        return userRepository.findByEmail(username).orElseThrow(() -> new UsernameNotFoundException("User not found with email: " + username));
    }

    public void register(@Valid RegisterRequest registerRequest) {
        User user=userRepository.findByEmail(registerRequest.getEmail()).orElse(null);
        if(user!=null) throw new UserAlreadyExistsException("User already exists");

        OtpVerification verification=otpVerificationRepository.findByEmail(registerRequest.getEmail()).orElse(null);
        if(verification==null) throw new EmailNotVerifiedException("Email not verified");

        if(!verification.isVerified()) throw new EmailNotVerifiedException("Email not verified");

        user=new User();

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

        user.setEmailVerified(true);
        user.setRole(Role.ROLE_USER);

        user.setMedicalProfileCompleted(false);
        user.setFamilyHistoryCompleted(false);

        userRepository.save(user);
    }

}