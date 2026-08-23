package com.bimodalpsyche.backend.auth.service;

import com.bimodalpsyche.backend.auth.dto.LoginRequest;
import com.bimodalpsyche.backend.auth.model.RefreshToken;
import com.bimodalpsyche.backend.auth.repository.RefreshTokenRepository;
import com.bimodalpsyche.backend.auth.security.JWTService;
import com.bimodalpsyche.backend.auth.security.TokenHashUtil;
import com.bimodalpsyche.backend.user.model.User;
import com.bimodalpsyche.backend.user.repository.UserRepository;
import jakarta.validation.Valid;
import org.springframework.security.authentication.AuthenticationManager;
import org.springframework.security.authentication.UsernamePasswordAuthenticationToken;
import org.springframework.security.core.Authentication;
import org.springframework.security.core.userdetails.UsernameNotFoundException;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.stereotype.Service;

import java.time.LocalDateTime;

@Service
public class AuthService {
    private final AuthenticationManager authenticationManager;
    private final JWTService jwtService;
    private final RefreshTokenRepository refreshTokenRepository;
    private final UserRepository userRepository;

    public AuthService(
            AuthenticationManager authenticationManager,
            JWTService jwtService,
            RefreshTokenRepository refreshTokenRepository,
            UserRepository userRepository) {

        this.authenticationManager = authenticationManager;
        this.jwtService = jwtService;
        this.refreshTokenRepository = refreshTokenRepository;
        this.userRepository = userRepository;
    }

    public LoginTokens login(@Valid LoginRequest loginRequest){
        Authentication authentication=authenticationManager.authenticate(
                new UsernamePasswordAuthenticationToken(loginRequest.getEmail(),loginRequest.getPassword()));

        String username=authentication.getName();

        String accessToken= jwtService.generateAccessToken(username);
        String refreshToken=jwtService.generateRefreshToken(username);

        User user =userRepository.findByEmail(username).orElseThrow(()->new UsernameNotFoundException("User not found with email "+username));
        refreshTokenRepository.deleteByUser(user);

        RefreshToken token = new RefreshToken();
        token.setUser(user);
        token.setTokenHash(TokenHashUtil.hash(refreshToken));
        token.setExpiresAt(LocalDateTime.now().plusDays(7));
        token.setRevoked(false);

        refreshTokenRepository.save(token);

        return new LoginTokens(accessToken, refreshToken);
    }

    public LoginTokens refreshTokens(String refreshToken) {

        if (!jwtService.isRefreshToken(refreshToken)) {
            throw new RuntimeException("Invalid refresh token");
        }

        String username = jwtService.extractUsername(refreshToken);

        User user = userRepository.findByEmail(username)
                .orElseThrow(() -> new UsernameNotFoundException("User not found"));

        RefreshToken storedToken = refreshTokenRepository.findByUser(user)
                .orElseThrow(() -> new RuntimeException("Refresh token not found"));

        if (storedToken.isRevoked()) {
            throw new RuntimeException("Refresh token has been revoked");
        }

        if (storedToken.getExpiresAt().isBefore(LocalDateTime.now())) {
            throw new RuntimeException("Refresh token expired");
        }

        if (!TokenHashUtil.hash(refreshToken).equals(storedToken.getTokenHash())) {
            throw new RuntimeException("Invalid refresh token");
        }

        String newAccessToken = jwtService.generateAccessToken(username);
        String newRefreshToken = jwtService.generateRefreshToken(username);

        storedToken.setTokenHash(TokenHashUtil.hash(newRefreshToken));
        storedToken.setExpiresAt(LocalDateTime.now().plusDays(7));
        storedToken.setRevoked(false);

        refreshTokenRepository.save(storedToken);

        return new LoginTokens(newAccessToken, newRefreshToken);
    }

    public void logout(String refreshToken) {

        if (refreshToken == null || refreshToken.isBlank()) {
            return;
        }

        String username = jwtService.extractUsername(refreshToken);

        User user = userRepository.findByEmail(username)
                .orElseThrow(() -> new UsernameNotFoundException("User not found"));

        RefreshToken storedToken = refreshTokenRepository.findByUser(user)
                .orElseThrow(() -> new RuntimeException("Refresh token not found"));

        storedToken.setRevoked(true);
        refreshTokenRepository.save(storedToken);
    }

    public record LoginTokens(String accessToken, String refreshToken) {}
}

