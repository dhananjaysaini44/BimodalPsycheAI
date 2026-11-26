package com.bimodalpsyche.backend.assessment.model;

import jakarta.persistence.*;
import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

@Getter
@Setter
@Entity
@Table(name = "assessment_signals")
@AllArgsConstructor
@NoArgsConstructor
public class AssessmentSignal {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "assessment_id", nullable = false)
    private Assessment assessment;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "analysis_result_id")
    private AssessmentAnalysisResult analysisResult;

    @Column(nullable = false, length = 100)
    private String signalCode;

    @Enumerated(EnumType.STRING)
    @Column(nullable = false)
    private SignalType signalType;

    private Float importance;

    @Column(columnDefinition = "TEXT")
    private String descriptionEncrypted;
}
