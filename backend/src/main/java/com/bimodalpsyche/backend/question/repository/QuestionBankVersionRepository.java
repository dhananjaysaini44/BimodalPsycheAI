package com.bimodalpsyche.backend.question.repository;

import com.bimodalpsyche.backend.question.model.QuestionBankVersion;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.Optional;

public interface QuestionBankVersionRepository extends JpaRepository<QuestionBankVersion, Long> {
    Optional<QuestionBankVersion> findFirstByActiveTrueOrderByIdDesc();
    Optional<QuestionBankVersion> findByVersion(String version);
}
