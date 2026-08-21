package com.bimodalpsyche.backend.auth.dto;

import jakarta.validation.constraints.NotBlank;
import lombok.Getter;

@Getter
public class OtpVerificationRequest {

    @NotBlank(message = "Otp cannot be blank")
    private String otp;

    @NotBlank(message = "Email cannot be blank")
    private String email;
}
