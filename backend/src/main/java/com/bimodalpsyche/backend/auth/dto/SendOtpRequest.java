package com.bimodalpsyche.backend.auth.dto;

import jakarta.validation.constraints.NotBlank;
import lombok.Getter;

@Getter
public class SendOtpRequest {

    @NotBlank(message = "Email cannot be blank")
    private String email;

}
