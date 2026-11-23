package com.bimodalpsyche.backend.session.controller;

import com.bimodalpsyche.backend.session.service.AssessmentSessionService;
import com.bimodalpsyche.backend.session.service.AssessmentSessionService.SessionResponse;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.security.core.annotation.AuthenticationPrincipal;
import org.springframework.security.core.userdetails.UserDetails;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/session")
public class AssessmentSessionController {
    private final AssessmentSessionService sessionService;

    public AssessmentSessionController(AssessmentSessionService sessionService) {
        this.sessionService = sessionService;
    }

    @PostMapping("/start")
    public ResponseEntity<SessionResponse> start(@AuthenticationPrincipal UserDetails user) {
        return ResponseEntity.status(HttpStatus.CREATED).body(sessionService.start(user.getUsername()));
    }

    @GetMapping("/draft")
    public ResponseEntity<SessionResponse> findUnfinished(@AuthenticationPrincipal UserDetails user) {
        return sessionService.findUnfinished(user.getUsername())
                .map(ResponseEntity::ok)
                .orElseGet(() -> ResponseEntity.noContent().build());
    }

    @GetMapping("/{sessionId}")
    public SessionResponse get(
            @PathVariable Long sessionId,
            @AuthenticationPrincipal UserDetails user) {
        return sessionService.get(sessionId, user.getUsername());
    }

    @PostMapping("/resume/{sessionId}")
    public SessionResponse resume(
            @PathVariable Long sessionId,
            @AuthenticationPrincipal UserDetails user) {
        return sessionService.resume(sessionId, user.getUsername());
    }

    @PostMapping("/complete/{sessionId}")
    public SessionResponse complete(
            @PathVariable Long sessionId,
            @AuthenticationPrincipal UserDetails user) {
        return sessionService.complete(sessionId, user.getUsername());
    }

    @PostMapping("/abandon/{sessionId}")
    public SessionResponse abandon(
            @PathVariable Long sessionId,
            @AuthenticationPrincipal UserDetails user) {
        return sessionService.abandon(sessionId, user.getUsername());
    }
}
