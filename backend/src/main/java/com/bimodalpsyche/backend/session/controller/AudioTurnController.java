package com.bimodalpsyche.backend.session.controller;

import com.bimodalpsyche.backend.session.service.AudioTurnService;
import com.bimodalpsyche.backend.session.service.AudioTurnService.AudioTurnResponse;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.security.core.annotation.AuthenticationPrincipal;
import org.springframework.security.core.userdetails.UserDetails;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;

@RestController
public class AudioTurnController {
    private final AudioTurnService audioTurnService;

    public AudioTurnController(AudioTurnService audioTurnService) {
        this.audioTurnService = audioTurnService;
    }

    @PostMapping(value = "/session/{sessionId}/turns/audio", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
    public ResponseEntity<AudioTurnResponse> submitAudioTurn(
            @PathVariable Long sessionId,
            @AuthenticationPrincipal UserDetails user,
            @RequestParam Long questionVariantId,
            @RequestParam("file") MultipartFile file) {
        return ResponseEntity.status(HttpStatus.CREATED).body(
                audioTurnService.submitAudioTurn(sessionId, user.getUsername(), questionVariantId, file));
    }
}
