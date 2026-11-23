package com.bimodalpsyche.backend.question.service;

import com.bimodalpsyche.backend.question.model.QuestionBankVersion;
import com.bimodalpsyche.backend.question.model.QuestionTheme;
import com.bimodalpsyche.backend.question.model.QuestionVariant;
import com.bimodalpsyche.backend.question.model.VariantType;
import com.bimodalpsyche.backend.question.repository.QuestionBankVersionRepository;
import com.bimodalpsyche.backend.question.repository.QuestionThemeRepository;
import com.bimodalpsyche.backend.question.repository.QuestionVariantRepository;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.boot.ApplicationArguments;
import org.springframework.boot.ApplicationRunner;
import org.jspecify.annotations.NonNull;
import org.springframework.stereotype.Component;
import org.springframework.transaction.annotation.Transactional;

import java.util.List;

@Component
public class QuestionBankSeeder implements ApplicationRunner {
    private static final List<ThemeSeed> THEMES = List.of(
            new ThemeSeed("mood", "Mood", "How's your week been so far?",
                    "It sounds like things have been heavy lately — want to tell me more about that?"),
            new ThemeSeed("sleep", "Sleep", "How have you been sleeping?",
                    "Has your sleep been affected by how you've been feeling?"),
            new ThemeSeed("energy", "Energy", "How's your energy been lately?",
                    "Have you noticed it's been harder to get through the day?"),
            new ThemeSeed("social", "Social connection", "Have you been spending time with friends or family?",
                    "Have you found yourself pulling away from people lately?"),
            new ThemeSeed("concentration", "Concentration", "How's it been to focus on things — work, study, hobbies?",
                    "Has it felt harder than usual to concentrate on things?"),
            new ThemeSeed("self_worth", "Self-worth", "How have you been feeling about yourself lately?",
                    "Have you been feeling down on yourself more than usual?"),
            new ThemeSeed("outlook", "Outlook", "How are you feeling about things going forward?",
                    "Has it felt hard to see things getting better lately?"));

    private final QuestionBankVersionRepository bankVersionRepository;
    private final QuestionThemeRepository themeRepository;
    private final QuestionVariantRepository variantRepository;
    private final String bankVersion;

    public QuestionBankSeeder(
            QuestionBankVersionRepository bankVersionRepository,
            QuestionThemeRepository themeRepository,
            QuestionVariantRepository variantRepository,
            @Value("${app.question-bank.version:initial-v1}") String bankVersion) {
        this.bankVersionRepository = bankVersionRepository;
        this.themeRepository = themeRepository;
        this.variantRepository = variantRepository;
        this.bankVersion = bankVersion;
    }

    @Override
    @Transactional
    public void run(@NonNull ApplicationArguments args) {
        QuestionBankVersion version = bankVersionRepository.findFirstByActiveTrueOrderByIdDesc()
                .orElseGet(() -> {
                    QuestionBankVersion created = new QuestionBankVersion();
                    created.setVersion(bankVersion);
                    created.setActive(true);
                    return bankVersionRepository.save(created);
                });

        for (ThemeSeed seed : THEMES) {
            QuestionTheme theme = themeRepository.findByBankVersionAndThemeCode(version, seed.code())
                    .orElseGet(() -> {
                        QuestionTheme created = new QuestionTheme();
                        created.setBankVersion(version);
                        created.setThemeCode(seed.code());
                        created.setDisplayName(seed.displayName());
                        return themeRepository.save(created);
                    });
            createVariantIfMissing(theme, VariantType.LIGHT, seed.light());
            createVariantIfMissing(theme, VariantType.DEEP, seed.deep());
        }
    }

    private void createVariantIfMissing(QuestionTheme theme, VariantType type, String text) {
        if (variantRepository.findByThemeAndVariantType(theme, type).isEmpty()) {
            QuestionVariant variant = new QuestionVariant();
            variant.setTheme(theme);
            variant.setVariantType(type);
            variant.setQuestionText(text);
            variantRepository.save(variant);
        }
    }

    private record ThemeSeed(String code, String displayName, String light, String deep) {
    }
}
