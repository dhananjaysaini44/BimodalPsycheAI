package com.bimodalpsyche.backend.auth.security;

import io.jsonwebtoken.Claims;
import io.jsonwebtoken.Jwts;
import io.jsonwebtoken.security.Keys;
import jakarta.annotation.PostConstruct;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.security.core.userdetails.UserDetails;
import org.springframework.stereotype.Service;

import javax.crypto.SecretKey;
import java.util.Base64;
import java.util.Date;
import java.util.UUID;
import java.util.function.Function;

@Service
public class JWTService {

    @Value("${jwt.secret}")
    private String secretKey;

    private SecretKey key;

    private final long accessTokenExpiration = 15 * 60 * 1000L;      // 15 minutes
    private final long refreshTokenExpiration = 7 * 24 * 60 * 60 * 1000L; // 7 days

    @PostConstruct
    void init() {
        if (secretKey == null || secretKey.isBlank()) {
            throw new IllegalStateException("jwt.secret must be set (Base64, at least 256 bits)");
        }
        // hmacShaKeyFor rejects keys shorter than 256 bits, so a weak secret fails at startup
        key = Keys.hmacShaKeyFor(Base64.getDecoder().decode(secretKey));
    }

    public String generateAccessToken(String username, Long refreshTokenId, int tokenVersion) {
        return Jwts.builder()
                .subject(username)
                .claim("type", "access")
                .claim("refreshTokenId", refreshTokenId)
                .claim("tokenVersion", tokenVersion)
                .issuedAt(new Date())
                .expiration(new Date(System.currentTimeMillis() + accessTokenExpiration))
                .signWith(key)
                .compact();
    }

    public String generateRefreshToken(String username) {
        return Jwts.builder()
                .subject(username)
                .claim("type", "refresh")
                // Unique id so two tokens issued in the same second never collide
                .id(UUID.randomUUID().toString())
                .issuedAt(new Date())
                .expiration(new Date(System.currentTimeMillis() + refreshTokenExpiration))
                .signWith(key)
                .compact();
    }

    public String extractUsername(String token) {
        return extractClaim(token, Claims::getSubject);
    }

    public boolean validateToken(String token, UserDetails userDetails) {
        // Signature and expiry are already enforced by the parser
        String username = extractUsername(token);
        return username != null && username.equals(userDetails.getUsername());
    }

    private <T> T extractClaim(String token,
                               Function<Claims, T> claimsResolver) {

        Claims claims = extractAllClaims(token);

        return claimsResolver.apply(claims);
    }

    private Claims extractAllClaims(String token) {

        return Jwts.parser()
                .verifyWith(key)
                .build()
                .parseSignedClaims(token)
                .getPayload();
    }

    public boolean isRefreshToken(String token) {
        String type = extractClaim(token, claims -> claims.get("type", String.class));
        return "refresh".equals(type);
    }

    public boolean isAccessToken(String token) {
        String type = extractClaim(token, claims->claims.get("type",String.class));
        return "access".equals(type);
    }

    public Long extractRefreshTokenId(String token) {
        return extractClaim(token, claims -> claims.get("refreshTokenId", Long.class));
    }

    public Integer extractTokenVersion(String token) {
        return extractClaim(token, claims -> {
            Number version = claims.get("tokenVersion", Number.class);
            return version == null ? null : version.intValue();
        });
    }
}
