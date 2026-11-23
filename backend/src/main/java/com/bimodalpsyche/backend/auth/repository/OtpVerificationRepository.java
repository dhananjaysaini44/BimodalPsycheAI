package com.bimodalpsyche.backend.auth.repository;

import com.bimodalpsyche.backend.auth.model.OtpVerification;
import jakarta.persistence.LockModeType;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Lock;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;

import java.util.Optional;

public interface OtpVerificationRepository extends JpaRepository<OtpVerification, Long> {
    Optional<OtpVerification> findByEmail(String email);

    void deleteByEmail(String email);


    @Lock(LockModeType.PESSIMISTIC_WRITE)
    @Query("select verification from OtpVerification verification where verification.email = :email")
    Optional<OtpVerification> findByEmailForUpdate(@Param("email") String email);
}
