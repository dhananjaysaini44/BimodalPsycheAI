package com.bimodalpsyche.backend.question.model;

import jakarta.persistence.*;
import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

@Getter
@Setter
@Entity
@Table(name = "question_variants")
@AllArgsConstructor
@NoArgsConstructor
public class QuestionVariant {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "theme_id", nullable = false)
    private QuestionTheme theme;

    @Enumerated(EnumType.STRING)
    @Column(nullable = false)
    private VariantType variantType;

    @Column(nullable = false, columnDefinition = "TEXT")
    private String questionText;
}
