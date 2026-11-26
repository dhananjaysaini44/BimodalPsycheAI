package com.bimodalpsyche.backend.assessment.repository;

import com.bimodalpsyche.backend.assessment.model.AssessmentSignal;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.Collection;

public interface AssessmentSignalRepository extends JpaRepository<AssessmentSignal, Long> {
    void deleteAllByAssessment_IdIn(Collection<Long> assessmentIds);
}
