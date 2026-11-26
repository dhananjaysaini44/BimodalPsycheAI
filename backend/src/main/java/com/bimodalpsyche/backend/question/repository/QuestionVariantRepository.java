package com.bimodalpsyche.backend.question.repository;

import com.bimodalpsyche.backend.question.model.QuestionVariant;
import com.bimodalpsyche.backend.question.model.QuestionTheme;
import com.bimodalpsyche.backend.question.model.VariantType;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.Optional;

public interface QuestionVariantRepository extends JpaRepository<QuestionVariant, Long> {
    Optional<QuestionVariant> findByIdAndTheme_BankVersion_ActiveTrue(Long id);
    Optional<QuestionVariant> findByIdAndTheme_BankVersion_Id(Long id, Long bankVersionId);
    Optional<QuestionVariant> findByThemeAndVariantType(QuestionTheme theme, VariantType variantType);
    java.util.List<QuestionVariant> findAllByTheme_BankVersion_IdOrderByTheme_IdAscIdAsc(Long bankVersionId);
}
