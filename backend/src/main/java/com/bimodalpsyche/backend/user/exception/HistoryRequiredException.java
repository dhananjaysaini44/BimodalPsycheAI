package com.bimodalpsyche.backend.user.exception;

public class HistoryRequiredException extends RuntimeException {
    public HistoryRequiredException(String message) {
        super(message);
    }
}
