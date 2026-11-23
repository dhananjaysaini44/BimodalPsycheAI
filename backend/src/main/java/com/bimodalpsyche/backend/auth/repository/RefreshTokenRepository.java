package com.bimodalpsyche.backend.auth.repository;

import com.bimodalpsyche.backend.auth.model.RefreshToken;
import com.bimodalpsyche.backend.user.model.User;
import jakarta.persistence.LockModeType;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Lock;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;

import java.util.Optional;

public interface RefreshTokenRepository extends JpaRepository<RefreshToken, Long> {

    Optional<RefreshToken> findByUser(User user);

    @Lock(LockModeType.PESSIMISTIC_WRITE)
    @Query("select token from RefreshToken token where token.user = :user")
    Optional<RefreshToken> findByUserForUpdate(@Param("user") User user);

    Optional<RefreshToken> findByIdAndRevokedFalse(Long id);

    void deleteByUser(User user);
}