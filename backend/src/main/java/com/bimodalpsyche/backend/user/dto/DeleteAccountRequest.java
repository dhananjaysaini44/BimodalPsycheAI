package com.bimodalpsyche.backend.user.dto;

import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.Pattern;
import jakarta.validation.constraints.Size;
import lombok.Getter;

@Getter
public class DeleteAccountRequest {
    @NotBlank(message = "Current password is required to delete your account")
    @Size(max = 72, message = "Password must be at most 72 characters")
    private String currentPassword;

    @NotBlank(message = "Email verification code is required")
    @Pattern(regexp = "\\d{6}", message = "Email verification code must be 6 digits")
    private String otp;
}
