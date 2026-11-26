package com.bimodalpsyche.backend.session.dto;

import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import jakarta.validation.constraints.Size;

public record SubmitTextTurnRequest(
        @NotNull Long questionVariantId,
        @NotBlank @Size(max = 10000) String responseText) {
}
