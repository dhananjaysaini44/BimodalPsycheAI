package com.bimodalpsyche.backend.user.dto;

import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.Size;
import lombok.Getter;

@Getter
public class HistoryDetailsRequest {
    @NotBlank(message = "Please provide details or state that there is no known history")
    @Size(max = 10000, message = "History details must be at most 10000 characters")
    private String details;
}
