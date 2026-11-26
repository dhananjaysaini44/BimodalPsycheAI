package com.bimodalpsyche.backend.auth.dto;

import jakarta.validation.Validation;
import jakarta.validation.Validator;
import org.junit.jupiter.api.Test;

import java.lang.reflect.Field;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class RegisterRequestConsentTest {
    private final Validator validator = Validation.buildDefaultValidatorFactory().getValidator();

    @Test
    void missingConsentIsRejected() {
        RegisterRequest request = new RegisterRequest();

        assertFalse(validator.validateProperty(request, "termsAccepted").isEmpty());
        assertFalse(validator.validateProperty(request, "privacyPolicyAccepted").isEmpty());
    }

    @Test
    void registrationWithBothConsentsPassesConsentValidation() throws Exception {
        RegisterRequest request = new RegisterRequest();
        setAccepted(request, "termsAccepted");
        setAccepted(request, "privacyPolicyAccepted");

        assertTrue(validator.validateProperty(request, "termsAccepted").isEmpty());
        assertTrue(validator.validateProperty(request, "privacyPolicyAccepted").isEmpty());
    }

    private void setAccepted(RegisterRequest request, String fieldName) throws Exception {
        Field field = RegisterRequest.class.getDeclaredField(fieldName);
        field.setAccessible(true);
        field.setBoolean(request, true);
    }
}
