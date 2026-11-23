package com.bimodalpsyche.backend.auth.security;

import com.bimodalpsyche.backend.user.service.UserService;
import com.bimodalpsyche.backend.auth.repository.RefreshTokenRepository;
import io.jsonwebtoken.JwtException;
import jakarta.servlet.FilterChain;
import jakarta.servlet.ServletException;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import org.springframework.security.authentication.UsernamePasswordAuthenticationToken;
import org.springframework.security.core.Authentication;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.security.core.userdetails.UserDetails;
import org.springframework.security.core.userdetails.UsernameNotFoundException;
import org.springframework.security.web.authentication.WebAuthenticationDetailsSource;
import org.springframework.stereotype.Component;
import org.springframework.web.filter.OncePerRequestFilter;

import java.io.IOException;

@Component
public class JWTFilter extends OncePerRequestFilter {

    private final JWTService jwtService;
    private final UserService userService;
    private final RefreshTokenRepository refreshTokenRepository;

    public JWTFilter(
            JWTService jwtService,
            UserService userService,
            RefreshTokenRepository refreshTokenRepository) {
        this.jwtService = jwtService;
        this.userService = userService;
        this.refreshTokenRepository = refreshTokenRepository;
    }

    @Override
    protected void doFilterInternal(HttpServletRequest request, HttpServletResponse response, FilterChain filterChain) throws ServletException, IOException {
        String authHeader=request.getHeader("Authorization");

        if(authHeader==null || !authHeader.startsWith("Bearer ")){
            filterChain.doFilter(request,response);
            return;
        }

        String jwt=authHeader.substring(7);

        // Any parsing failure leaves the request unauthenticated; Spring Security then returns 401/403
        try {
            if (jwtService.isAccessToken(jwt)) {
                String username = jwtService.extractUsername(jwt);
                Long refreshTokenId = jwtService.extractRefreshTokenId(jwt);
                Integer tokenVersion = jwtService.extractTokenVersion(jwt);
                Authentication authentication = SecurityContextHolder.getContext().getAuthentication();

                if (username != null && tokenVersion != null && authentication == null
                        && isCurrentSession(refreshTokenId, tokenVersion, username)) {
                    UserDetails userDetails = userService.loadUserByUsername(username);

                    if (userDetails.isEnabled() && jwtService.validateToken(jwt, userDetails)) {
                        UsernamePasswordAuthenticationToken authToken =
                                new UsernamePasswordAuthenticationToken(
                                        userDetails,
                                        null,
                                        userDetails.getAuthorities()
                                );
                        authToken.setDetails(new WebAuthenticationDetailsSource().buildDetails(request));

                        SecurityContextHolder.getContext().setAuthentication(authToken);
                    }
                }
            }
        } catch (JwtException | IllegalArgumentException | UsernameNotFoundException e) {
            SecurityContextHolder.clearContext();
        }

        filterChain.doFilter(request,response);
    }

    private boolean isCurrentSession(Long refreshTokenId, Integer tokenVersion, String username) {
        if (refreshTokenId == null) {
            return false;
        }

        return refreshTokenRepository.findByIdAndRevokedFalse(refreshTokenId)
                .filter(token -> token.getTokenVersion() != null
                        && token.getTokenVersion().equals(tokenVersion)
                        && username.equals(token.getUser().getEmail()))
                .isPresent();
    }
}
