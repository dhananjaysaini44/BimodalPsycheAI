package com.bimodalpsyche.backend.session.model;

import com.bimodalpsyche.backend.mlmodel.model.ModelVersion;
import jakarta.persistence.*;
import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

import java.time.LocalDateTime;

@Getter
@Setter
@Entity
@Table(name = "turn_analysis")
@AllArgsConstructor
@NoArgsConstructor
public class TurnAnalysis {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @OneToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "turn_id", nullable = false, unique = true)
    private SessionTurn turn;

    private Float sentimentScore;

    @Column(nullable = false)
    private boolean crisisDetected;

    @Column(length = 100)
    private String crisisRuleCode;

    @Column(nullable = false)
    private LocalDateTime analyzedAt;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "model_version_id")
    private ModelVersion modelVersion;
}
