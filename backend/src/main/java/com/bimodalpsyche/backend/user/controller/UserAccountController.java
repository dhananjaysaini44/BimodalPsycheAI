package com.bimodalpsyche.backend.user.controller;

import com.bimodalpsyche.backend.user.dto.ChangePasswordRequest;
import com.bimodalpsyche.backend.user.dto.DeleteAccountRequest;
import com.bimodalpsyche.backend.user.dto.UpdateProfileRequest;
import com.bimodalpsyche.backend.user.service.UserAccountService;
import com.bimodalpsyche.backend.user.service.UserAccountService.ProfileResponse;
import jakarta.validation.Valid;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.HttpHeaders;
import org.springframework.http.ResponseCookie;
import org.springframework.http.ResponseEntity;
import org.springframework.security.core.annotation.AuthenticationPrincipal;
import org.springframework.security.core.userdetails.UserDetails;
import org.springframework.web.bind.annotation.DeleteMapping;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PutMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.time.Duration;

@RestController
@RequestMapping("/user")
public class UserAccountController {
    private final UserAccountService userAccountService;
    private final boolean cookieSecure;

    public UserAccountController(
            UserAccountService userAccountService,
            @Value("${app.cookie.secure}") boolean cookieSecure) {
        this.userAccountService = userAccountService;
        this.cookieSecure = cookieSecure;
    }

    @GetMapping("/profile")
    public ProfileResponse getProfile(@AuthenticationPrincipal UserDetails user) {
        return userAccountService.getProfile(user.getUsername());
    }

    @PutMapping("/profile")
    public ProfileResponse updateProfile(
            @AuthenticationPrincipal UserDetails user,
            @Valid @RequestBody UpdateProfileRequest request) {
        return userAccountService.updateProfile(user.getUsername(), request);
    }

    @PutMapping("/password")
    public ResponseEntity<Void> changePassword(
            @AuthenticationPrincipal UserDetails user,
            @Valid @RequestBody ChangePasswordRequest request) {
        userAccountService.changePassword(user.getUsername(), request);
        return ResponseEntity.noContent()
                .header(HttpHeaders.SET_COOKIE, expiredRefreshCookie().toString())
                .build();
    }

    @PostMapping("/security/send-otp")
    public ResponseEntity<String> sendSecurityOtp(@AuthenticationPrincipal UserDetails user) {
        return ResponseEntity.ok(userAccountService.sendAccountActionOtp(user.getUsername()));
    }

    @DeleteMapping("/account")
    public ResponseEntity<Void> deleteAccount(
            @AuthenticationPrincipal UserDetails user,
            @Valid @RequestBody DeleteAccountRequest confirmation) {
        userAccountService.deleteAccount(
                user.getUsername(), confirmation.getCurrentPassword(), confirmation.getOtp());
        return ResponseEntity.noContent()
                .header(HttpHeaders.SET_COOKIE, expiredRefreshCookie().toString())
                .build();
    }

    private ResponseCookie expiredRefreshCookie() {
        return ResponseCookie.from("refreshToken", "")
                .httpOnly(true)
                .secure(cookieSecure)
                .path("/auth")
                .maxAge(Duration.ZERO)
                .sameSite("Strict")
                .build();
    }
}
