package com.bimodalpsyche.backend.assessment.controller;

import com.bimodalpsyche.backend.assessment.service.AssessmentAnalysisService;
import com.bimodalpsyche.backend.assessment.service.AssessmentAnalysisService.AssessmentResponse;
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
@RequestMapping("/session/{sessionId}/assessment")
public class AssessmentAnalysisController {
    private final AssessmentAnalysisService assessmentAnalysisService;

    public AssessmentAnalysisController(AssessmentAnalysisService assessmentAnalysisService) {
        this.assessmentAnalysisService = assessmentAnalysisService;
    }

    @PostMapping
    public ResponseEntity<AssessmentResponse> analyze(
            @PathVariable Long sessionId, @AuthenticationPrincipal UserDetails user) {
        return ResponseEntity.status(HttpStatus.CREATED)
                .body(assessmentAnalysisService.analyze(sessionId, user.getUsername()));
    }

    @GetMapping
    public AssessmentResponse get(
            @PathVariable Long sessionId, @AuthenticationPrincipal UserDetails user) {
        return assessmentAnalysisService.get(sessionId, user.getUsername());
    }
}
