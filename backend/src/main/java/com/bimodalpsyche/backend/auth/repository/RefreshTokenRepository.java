package com.bimodalpsyche.backend.auth.repository;

import com.bimodalpsyche.backend.auth.model.RefreshToken;
import com.bimodalpsyche.backend.user.model.User;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.Optional;

public interface RefreshTokenRepository extends JpaRepository<RefreshToken, Long> {

    Optional<RefreshToken> findByUser(User user);

    void deleteByUser(User user);
}