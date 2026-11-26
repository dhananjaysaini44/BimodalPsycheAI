package com.bimodalpsyche.backend.session.exception;

public class MlAnalysisUnavailableException extends RuntimeException {
    public MlAnalysisUnavailableException(String message, Throwable cause) {
        super(message, cause);
    }

    public MlAnalysisUnavailableException(String message) {
        super(message);
    }
}
