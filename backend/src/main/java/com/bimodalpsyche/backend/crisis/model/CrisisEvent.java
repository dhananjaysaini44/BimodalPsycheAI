package com.bimodalpsyche.backend.crisis.model;

import com.bimodalpsyche.backend.session.model.AssessmentSession;
import com.bimodalpsyche.backend.session.model.SessionTurn;
import jakarta.persistence.*;
import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

import java.time.LocalDateTime;

@Getter
@Setter
@Entity
@Table(name = "crisis_events")
@AllArgsConstructor
@NoArgsConstructor
public class CrisisEvent {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "session_id", nullable = false)
    private AssessmentSession session;

    // At most one crisis event per turn; nullable for session-level events
    @OneToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "turn_id", unique = true)
    private SessionTurn turn;

    @Column(nullable = false)
    private LocalDateTime detectedAt;

    @Column(nullable = false, length = 50)
    private String detectionType;

    @Column(length = 100)
    private String patternCode;

    @Column(nullable = false)
    private boolean resourceShown;

    private LocalDateTime resolvedAt;
}
