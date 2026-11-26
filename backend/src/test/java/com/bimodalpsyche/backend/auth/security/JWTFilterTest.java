package com.bimodalpsyche.backend.auth.security;

import com.bimodalpsyche.backend.auth.model.RefreshToken;
import com.bimodalpsyche.backend.auth.repository.RefreshTokenRepository;
import com.bimodalpsyche.backend.user.model.Role;
import com.bimodalpsyche.backend.user.model.User;
import com.bimodalpsyche.backend.user.service.UserService;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.springframework.mock.web.MockFilterChain;
import org.springframework.mock.web.MockHttpServletRequest;
import org.springframework.mock.web.MockHttpServletResponse;
import org.springframework.security.core.context.SecurityContextHolder;

import java.util.Optional;

import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

@ExtendWith(MockitoExtension.class)
class JWTFilterTest {
    @Mock
    private JWTService jwtService;

    @Mock
    private UserService userService;

    @Mock
    private RefreshTokenRepository refreshTokenRepository;

    private JWTFilter jwtFilter;

    @BeforeEach
    void setUp() {
        SecurityContextHolder.clearContext();
        jwtFilter = new JWTFilter(jwtService, userService, refreshTokenRepository);
    }

    @AfterEach
    void clearSecurityContext() {
        SecurityContextHolder.clearContext();
    }

    @Test
    void authenticatesAccessTokenForCurrentActiveSession() throws Exception {
        User user = user();
        RefreshToken refreshToken = refreshToken(user, false, 3);
        when(jwtService.isAccessToken("access-token")).thenReturn(true);
        when(jwtService.extractUsername("access-token")).thenReturn(user.getEmail());
        when(jwtService.extractRefreshTokenId("access-token")).thenReturn(12L);
        when(jwtService.extractTokenVersion("access-token")).thenReturn(3);
        when(refreshTokenRepository.findByIdAndRevokedFalse(12L)).thenReturn(Optional.of(refreshToken));
        when(userService.loadUserByUsername(user.getEmail())).thenReturn(user);
        when(jwtService.validateToken("access-token", user)).thenReturn(true);

        filter("access-token");

        assertNotNull(SecurityContextHolder.getContext().getAuthentication());
    }

    @Test
    void rejectsAccessTokenAfterRefreshSessionIsRevoked() throws Exception {
        when(jwtService.isAccessToken("access-token")).thenReturn(true);
        when(jwtService.extractUsername("access-token")).thenReturn("person@example.com");
        when(jwtService.extractRefreshTokenId("access-token")).thenReturn(12L);
        when(jwtService.extractTokenVersion("access-token")).thenReturn(3);
        when(refreshTokenRepository.findByIdAndRevokedFalse(12L)).thenReturn(Optional.empty());

        filter("access-token");

        assertNull(SecurityContextHolder.getContext().getAuthentication());
        verify(userService, never()).loadUserByUsername("person@example.com");
    }

    @Test
    void rejectsAccessTokenFromPreviousLoginVersion() throws Exception {
        User user = user();
        when(jwtService.isAccessToken("access-token")).thenReturn(true);
        when(jwtService.extractUsername("access-token")).thenReturn(user.getEmail());
        when(jwtService.extractRefreshTokenId("access-token")).thenReturn(12L);
        when(jwtService.extractTokenVersion("access-token")).thenReturn(3);
        when(refreshTokenRepository.findByIdAndRevokedFalse(12L))
                .thenReturn(Optional.of(refreshToken(user, false, 4)));

        filter("access-token");

        assertNull(SecurityContextHolder.getContext().getAuthentication());
        verify(userService, never()).loadUserByUsername(user.getEmail());
    }

    private void filter(String token) throws Exception {
        MockHttpServletRequest request = new MockHttpServletRequest("GET", "/session/draft");
        request.addHeader("Authorization", "Bearer " + token);
        jwtFilter.doFilter(
                request,
                new MockHttpServletResponse(),
                new MockFilterChain());
    }

    private User user() {
        User user = new User();
        user.setEmail("person@example.com");
        user.setEmailVerified(true);
        user.setRole(Role.ROLE_USER);
        return user;
    }

    private RefreshToken refreshToken(User user, boolean revoked, int tokenVersion) {
        RefreshToken refreshToken = new RefreshToken();
        refreshToken.setUser(user);
        refreshToken.setRevoked(revoked);
        refreshToken.setTokenVersion(tokenVersion);
        return refreshToken;
    }
}
