package com.bimodalpsyche.backend.question.repository;

import com.bimodalpsyche.backend.question.model.QuestionBankVersion;
import com.bimodalpsyche.backend.question.model.QuestionTheme;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.Optional;

public interface QuestionThemeRepository extends JpaRepository<QuestionTheme, Long> {
    Optional<QuestionTheme> findByBankVersionAndThemeCode(QuestionBankVersion bankVersion, String themeCode);
}
