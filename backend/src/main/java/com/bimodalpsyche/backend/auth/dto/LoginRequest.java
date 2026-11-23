package com.bimodalpsyche.backend.auth.dto;

import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.Size;
import lombok.Getter;

import java.util.Locale;

@Getter
public class LoginRequest {
    @NotBlank(message = "Email cannot be blank")
    @Size(max = 255, message = "Email is too long")
    private String email;

    @NotBlank(message = "Password cannot be blank")
    @Size(max = 72, message = "Password is too long")
    private String password;

    public String getEmail() {
        return email == null ? null : email.trim().toLowerCase(Locale.ROOT);
    }
}
