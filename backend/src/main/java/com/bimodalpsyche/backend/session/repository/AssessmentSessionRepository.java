package com.bimodalpsyche.backend.session.repository;

import com.bimodalpsyche.backend.session.model.AssessmentSession;
import com.bimodalpsyche.backend.session.model.SessionStatus;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.Collection;
import java.util.Optional;

public interface AssessmentSessionRepository extends JpaRepository<AssessmentSession, Long> {
    Optional<AssessmentSession> findFirstByUser_IdAndStatusInOrderByCreatedAtDesc(
            Long userId,
            Collection<SessionStatus> statuses);

    Optional<AssessmentSession> findByIdAndUser_Id(Long id, Long userId);
}
