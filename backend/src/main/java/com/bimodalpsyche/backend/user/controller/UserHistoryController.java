package com.bimodalpsyche.backend.user.controller;

import com.bimodalpsyche.backend.user.dto.HistoryDetailsRequest;
import com.bimodalpsyche.backend.user.service.UserHistoryService;
import com.bimodalpsyche.backend.user.service.UserHistoryService.HistoryResponse;
import jakarta.validation.Valid;
import org.springframework.security.core.annotation.AuthenticationPrincipal;
import org.springframework.security.core.userdetails.UserDetails;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PutMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/user/history")
public class UserHistoryController {
    private final UserHistoryService userHistoryService;

    public UserHistoryController(UserHistoryService userHistoryService) {
        this.userHistoryService = userHistoryService;
    }

    @GetMapping("/medical")
    public HistoryResponse getMedical(@AuthenticationPrincipal UserDetails user) {
        return userHistoryService.getMedicalHistory(user.getUsername());
    }

    @PutMapping("/medical")
    public HistoryResponse saveMedical(
            @AuthenticationPrincipal UserDetails user,
            @Valid @RequestBody HistoryDetailsRequest request) {
        return userHistoryService.saveMedicalHistory(user.getUsername(), request);
    }

    @GetMapping("/family")
    public HistoryResponse getFamily(@AuthenticationPrincipal UserDetails user) {
        return userHistoryService.getFamilyHistory(user.getUsername());
    }

    @PutMapping("/family")
    public HistoryResponse saveFamily(
            @AuthenticationPrincipal UserDetails user,
            @Valid @RequestBody HistoryDetailsRequest request) {
        return userHistoryService.saveFamilyHistory(user.getUsername(), request);
    }
}
