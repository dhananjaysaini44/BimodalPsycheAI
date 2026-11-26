package com.bimodalpsyche.backend.session.controller;

import com.bimodalpsyche.backend.session.dto.SubmitTextTurnRequest;
import com.bimodalpsyche.backend.session.service.SessionTurnService;
import com.bimodalpsyche.backend.session.service.SessionTurnService.TurnResponse;
import jakarta.validation.Valid;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.security.core.annotation.AuthenticationPrincipal;
import org.springframework.security.core.userdetails.UserDetails;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/session/{sessionId}/turns")
public class SessionTurnController {
    private final SessionTurnService sessionTurnService;

    public SessionTurnController(SessionTurnService sessionTurnService) {
        this.sessionTurnService = sessionTurnService;
    }

    @PostMapping
    public ResponseEntity<TurnResponse> submitTextTurn(
            @PathVariable Long sessionId,
            @AuthenticationPrincipal UserDetails user,
            @Valid @RequestBody SubmitTextTurnRequest request) {
        TurnResponse response = sessionTurnService.submitTextTurn(
                sessionId, user.getUsername(), request.questionVariantId(), request.responseText());
        return ResponseEntity.status(HttpStatus.CREATED).body(response);
    }

    @GetMapping
    public java.util.List<SessionTurnService.TurnHistoryResponse> getTurns(
            @PathVariable Long sessionId,
            @AuthenticationPrincipal UserDetails user) {
        return sessionTurnService.getTurns(sessionId, user.getUsername());
    }
}
