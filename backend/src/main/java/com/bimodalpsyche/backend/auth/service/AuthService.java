package com.bimodalpsyche.backend.auth.service;

import com.bimodalpsyche.backend.auth.dto.LoginRequest;
import com.bimodalpsyche.backend.auth.exception.InvalidTokenException;
import com.bimodalpsyche.backend.auth.model.RefreshToken;
import com.bimodalpsyche.backend.auth.repository.RefreshTokenRepository;
import com.bimodalpsyche.backend.auth.security.JWTService;
import com.bimodalpsyche.backend.auth.security.TokenHashUtil;
import com.bimodalpsyche.backend.user.model.User;
import com.bimodalpsyche.backend.user.repository.UserRepository;
import io.jsonwebtoken.JwtException;
import jakarta.transaction.Transactional;
import org.springframework.security.authentication.AuthenticationManager;
import org.springframework.security.authentication.BadCredentialsException;
import org.springframework.security.authentication.UsernamePasswordAuthenticationToken;
import org.springframework.security.core.Authentication;
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

    @Transactional
    public LoginTokens login(LoginRequest loginRequest){
        Authentication authentication=authenticationManager.authenticate(
                new UsernamePasswordAuthenticationToken(loginRequest.getEmail(),loginRequest.getPassword()));

        String username=authentication.getName();

        User user =userRepository.findByEmail(username).orElseThrow(()->new BadCredentialsException("Invalid Credentials"));

        String refreshToken=jwtService.generateRefreshToken(username);

        // One refresh token row per user; a new login replaces (and so invalidates) the previous one
        RefreshToken token = refreshTokenRepository.findByUser(user).orElse(new RefreshToken());
        token.setUser(user);
        token.setTokenHash(TokenHashUtil.hash(refreshToken));
        token.setExpiresAt(LocalDateTime.now().plusDays(7));
        token.setRevoked(false);
        token.setTokenVersion(token.getTokenVersion() == null ? 1 : token.getTokenVersion() + 1);

        RefreshToken savedToken = refreshTokenRepository.saveAndFlush(token);
        String accessToken = jwtService.generateAccessToken(
                username, savedToken.getId(), savedToken.getTokenVersion());

        return new LoginTokens(accessToken, refreshToken);
    }

    @Transactional(dontRollbackOn = InvalidTokenException.class)
    public LoginTokens refreshTokens(String refreshToken) {

        String username = usernameFromRefreshToken(refreshToken);

        User user = userRepository.findByEmail(username)
                .orElseThrow(() -> new InvalidTokenException("Invalid refresh token"));

        RefreshToken storedToken = refreshTokenRepository.findByUserForUpdate(user)
                .orElseThrow(() -> new InvalidTokenException("Invalid refresh token"));

        if (storedToken.isRevoked() || storedToken.getExpiresAt().isBefore(LocalDateTime.now())) {
            throw new InvalidTokenException("Invalid refresh token");
        }

        if (!TokenHashUtil.matches(refreshToken, storedToken.getTokenHash())) {
            // A validly signed but already-rotated token means it was replayed (likely stolen):
            // revoke the current token too so both parties have to log in again
            storedToken.setRevoked(true);
            refreshTokenRepository.save(storedToken);
            throw new InvalidTokenException("Invalid refresh token");
        }

        String newRefreshToken = jwtService.generateRefreshToken(username);

        if (storedToken.getTokenVersion() == null) {
            storedToken.setTokenVersion(1);
        }
        storedToken.setTokenHash(TokenHashUtil.hash(newRefreshToken));
        storedToken.setExpiresAt(LocalDateTime.now().plusDays(7));

        RefreshToken savedToken = refreshTokenRepository.saveAndFlush(storedToken);
        String newAccessToken = jwtService.generateAccessToken(
                username, savedToken.getId(), savedToken.getTokenVersion());

        return new LoginTokens(newAccessToken, newRefreshToken);
    }

    @Transactional
    public void logout(String refreshToken) {

        if (refreshToken == null || refreshToken.isBlank()) {
            return;
        }

        String username;
        try {
            username = usernameFromRefreshToken(refreshToken);
        } catch (InvalidTokenException e) {
            // Nothing to revoke; the cookie is cleared by the controller anyway
            return;
        }

        userRepository.findByEmail(username)
                .flatMap(refreshTokenRepository::findByUserForUpdate)
                // Only the holder of the current token may revoke it
                .filter(stored -> TokenHashUtil.matches(refreshToken, stored.getTokenHash()))
                .ifPresent(stored -> {
                    stored.setRevoked(true);
                    refreshTokenRepository.save(stored);
                });
    }

    private String usernameFromRefreshToken(String refreshToken) {
        try {
            if (!jwtService.isRefreshToken(refreshToken)) {
                throw new InvalidTokenException("Invalid refresh token");
            }
            return jwtService.extractUsername(refreshToken);
        } catch (JwtException | IllegalArgumentException e) {
            throw new InvalidTokenException("Invalid refresh token");
        }
    }

    public record LoginTokens(String accessToken, String refreshToken) {}
}
