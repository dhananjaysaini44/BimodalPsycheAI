package com.bimodalpsyche.backend.assessment.repository;

import com.bimodalpsyche.backend.assessment.model.Assessment;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.List;
import java.util.Optional;

public interface AssessmentRepository extends JpaRepository<Assessment, Long> {
    Optional<Assessment> findBySession_Id(Long sessionId);

    List<Assessment> findAllBySession_IdIn(List<Long> sessionIds);
}
