package com.bimodalpsyche.backend.auth.exception;

public class LegalConsentRequiredException extends RuntimeException {
    public LegalConsentRequiredException(String message) {
        super(message);
    }
}
