package com.bimodalpsyche.backend.question.model;

import jakarta.persistence.*;
import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

@Getter
@Setter
@Entity
@Table(name = "question_themes",
        uniqueConstraints = @UniqueConstraint(columnNames = {"bank_version_id", "theme_code"}))
@AllArgsConstructor
@NoArgsConstructor
public class QuestionTheme {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "bank_version_id", nullable = false)
    private QuestionBankVersion bankVersion;

    @Column(nullable = false, length = 100)
    private String themeCode;

    @Column(nullable = false)
    private String displayName;
}
