package com.bimodalpsyche.backend.session.controller;

import com.bimodalpsyche.backend.session.service.SessionQuestionService;
import com.bimodalpsyche.backend.session.service.SessionQuestionService.NextQuestionResponse;
import org.springframework.security.core.annotation.AuthenticationPrincipal;
import org.springframework.security.core.userdetails.UserDetails;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/session/{sessionId}/question")
public class SessionQuestionController {
    private final SessionQuestionService sessionQuestionService;

    public SessionQuestionController(SessionQuestionService sessionQuestionService) {
        this.sessionQuestionService = sessionQuestionService;
    }

    @GetMapping("/next")
    public NextQuestionResponse getNextQuestion(
            @PathVariable Long sessionId,
            @AuthenticationPrincipal UserDetails user) {
        return sessionQuestionService.getNextQuestion(sessionId, user.getUsername());
    }
}
