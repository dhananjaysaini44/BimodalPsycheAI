package com.bimodalpsyche.backend.user.dto;

import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.Pattern;
import jakarta.validation.constraints.Size;
import lombok.Getter;

@Getter
public class ChangePasswordRequest {
    @NotBlank(message = "Current password is required")
    @Size(max = 72, message = "Current password must be at most 72 characters")
    private String currentPassword;

    @NotBlank(message = "New password is required")
    @Size(min = 8, max = 72, message = "New password must be between 8 and 72 characters")
    private String newPassword;

    @NotBlank(message = "Email verification code is required")
    @Pattern(regexp = "\\d{6}", message = "Email verification code must be 6 digits")
    private String otp;
}
