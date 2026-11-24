package com.bimodalpsyche.backend.session.repository;

import com.bimodalpsyche.backend.session.model.AssessmentSession;
import com.bimodalpsyche.backend.session.model.SessionStatus;
import jakarta.persistence.LockModeType;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Lock;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;

import java.util.Collection;
import java.util.List;
import java.util.Optional;

public interface AssessmentSessionRepository extends JpaRepository<AssessmentSession, Long> {
    Optional<AssessmentSession> findFirstByUser_IdAndStatusInOrderByCreatedAtDesc(
            Long userId,
            Collection<SessionStatus> statuses);

    Optional<AssessmentSession> findByIdAndUser_Id(Long id, Long userId);

    List<AssessmentSession> findAllByUser_Id(Long userId);

    List<AssessmentSession> findAllByUser_IdOrderByCreatedAtDesc(Long userId);

    Optional<AssessmentSession> findByIdAndUser_Email(Long id, String email);

    Optional<AssessmentSession> findByIdAndUser_EmailAndStatus(
            Long id, String email, SessionStatus status);

    @Lock(LockModeType.PESSIMISTIC_WRITE)
    @Query("select s from AssessmentSession s where s.id = :sessionId and s.user.email = :email")
    Optional<AssessmentSession> findByIdAndUser_EmailForUpdate(
            @Param("sessionId") Long sessionId,
            @Param("email") String email);
}
