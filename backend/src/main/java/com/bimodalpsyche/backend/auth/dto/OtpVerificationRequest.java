package com.bimodalpsyche.backend.auth.dto;

import jakarta.validation.constraints.Email;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.Pattern;
import jakarta.validation.constraints.Size;
import lombok.Getter;

import java.util.Locale;

@Getter
public class OtpVerificationRequest {

    @NotBlank(message = "Otp cannot be blank")
    @Pattern(regexp = "\\d{6}", message = "Otp must be 6 digits")
    private String otp;

    @NotBlank(message = "Email cannot be blank")
    @Email(message = "Email must be valid")
    @Size(max = 255, message = "Email is too long")
    private String email;

    public String getEmail() {
        return email == null ? null : email.trim().toLowerCase(Locale.ROOT);
    }
}
