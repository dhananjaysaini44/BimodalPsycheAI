package com.bimodalpsyche.backend.assessment.repository;

import com.bimodalpsyche.backend.assessment.model.AssessmentAnalysisResult;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.Collection;
import java.util.List;

public interface AssessmentAnalysisResultRepository extends JpaRepository<AssessmentAnalysisResult, Long> {
    List<AssessmentAnalysisResult> findAllByAssessment_IdOrderByIdAsc(Long assessmentId);

    void deleteAllByAssessment_IdIn(Collection<Long> assessmentIds);
}
