package com.bimodalpsyche.backend.auth.controller;

import com.bimodalpsyche.backend.auth.dto.LoginRequest;
import com.bimodalpsyche.backend.auth.dto.OtpVerificationRequest;
import com.bimodalpsyche.backend.auth.dto.RegisterRequest;
import com.bimodalpsyche.backend.auth.dto.SendOtpRequest;
import com.bimodalpsyche.backend.auth.service.AuthService;
import com.bimodalpsyche.backend.auth.service.OtpService;
import com.bimodalpsyche.backend.user.service.UserService;
import jakarta.validation.Valid;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.HttpHeaders;
import org.springframework.http.ResponseCookie;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.time.Duration;

@RestController
@RequestMapping("/auth")
public class AuthController {

    private final UserService userService;
    private final AuthService authService;
    private final OtpService otpService;
    private final boolean cookieSecure;

    public AuthController(UserService userService, AuthService authService, OtpService otpService,
                          @Value("${app.cookie.secure}") boolean cookieSecure) {
        this.userService = userService;
        this.authService = authService;
        this.otpService = otpService;
        this.cookieSecure = cookieSecure;
    }

    private ResponseCookie refreshCookie(String value, Duration maxAge) {
        return ResponseCookie.from("refreshToken", value)
                .httpOnly(true)
                .secure(cookieSecure)
                .path("/auth")
                .maxAge(maxAge)
                .sameSite("Strict")
                .build();
    }

    @PostMapping("/register")
    public ResponseEntity<String> registerUser(@Valid @RequestBody RegisterRequest registerRequest) {
        userService.register(registerRequest);
        return ResponseEntity.ok("User registered successfully");
    }

    @PostMapping("/login")
    public ResponseEntity<LoginResponse> login(
            @Valid @RequestBody LoginRequest loginRequest) {

        AuthService.LoginTokens tokens = authService.login(loginRequest);

        ResponseCookie refreshCookie = refreshCookie(tokens.refreshToken(), Duration.ofDays(7));

        return ResponseEntity.ok()
                .header(HttpHeaders.SET_COOKIE, refreshCookie.toString())
                .body(new LoginResponse("Login Successful", tokens.accessToken()));
    }

    @PostMapping("/verify-otp")
    public ResponseEntity<String> verifyOtp(@Valid @RequestBody OtpVerificationRequest otpVerificationRequest) {
        otpService.verifyOtp(otpVerificationRequest);
        return ResponseEntity.ok("OTP verified successfully");
    }

    @PostMapping("/send-otp")
    public ResponseEntity<String> sendOtp(@Valid @RequestBody SendOtpRequest request){
        return ResponseEntity.ok(otpService.sendOtp(request));
    }

    @PostMapping("/refresh")
    public ResponseEntity<LoginResponse> refresh(
            @CookieValue("refreshToken") String refreshToken) {

        AuthService.LoginTokens tokens = authService.refreshTokens(refreshToken);

        ResponseCookie refreshCookie = refreshCookie(tokens.refreshToken(), Duration.ofDays(7));

        return ResponseEntity.ok()
                .header(HttpHeaders.SET_COOKIE, refreshCookie.toString())
                .body(new LoginResponse(
                        "Token refreshed successfully",
                        tokens.accessToken()
                ));
    }

    @PostMapping("/logout")
    public ResponseEntity<String> logout(
            @CookieValue(value = "refreshToken", required = false) String refreshToken) {

        authService.logout(refreshToken);

        ResponseCookie cookie = refreshCookie("", Duration.ZERO);

        return ResponseEntity.ok()
                .header(HttpHeaders.SET_COOKIE, cookie.toString())
                .body("Logged out successfully");
    }

    public record LoginResponse(String message,String token){}
}
