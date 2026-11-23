package com.bimodalpsyche.backend.question.service;

import com.bimodalpsyche.backend.question.model.QuestionBankVersion;
import com.bimodalpsyche.backend.question.repository.QuestionBankVersionRepository;
import com.bimodalpsyche.backend.question.repository.QuestionVariantRepository;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.List;

@Service
public class QuestionBankService {
    private final QuestionBankVersionRepository versionRepository;
    private final QuestionVariantRepository variantRepository;

    public QuestionBankService(
            QuestionBankVersionRepository versionRepository,
            QuestionVariantRepository variantRepository) {
        this.versionRepository = versionRepository;
        this.variantRepository = variantRepository;
    }

    @Transactional(readOnly = true)
    public QuestionBankResponse getActiveBank() {
        QuestionBankVersion version = versionRepository.findFirstByActiveTrueOrderByIdDesc()
                .orElseThrow(() -> new IllegalStateException("No active question bank is configured"));
        List<QuestionVariantResponse> variants = variantRepository
                .findAllByTheme_BankVersion_IdOrderByTheme_IdAscIdAsc(version.getId())
                .stream()
                .map(variant -> new QuestionVariantResponse(
                        variant.getId(),
                        variant.getTheme().getThemeCode(),
                        variant.getTheme().getDisplayName(),
                        variant.getVariantType(),
                        variant.getQuestionText()))
                .toList();
        return new QuestionBankResponse(version.getVersion(), variants);
    }

    public record QuestionBankResponse(String version, List<QuestionVariantResponse> variants) {
    }

    public record QuestionVariantResponse(
            Long id,
            String themeCode,
            String themeName,
            com.bimodalpsyche.backend.question.model.VariantType variantType,
            String questionText) {
    }
}
