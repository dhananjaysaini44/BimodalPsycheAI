package com.bimodalpsyche.backend.session.model;

import com.bimodalpsyche.backend.question.model.QuestionVariant;
import jakarta.persistence.*;
import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;
import org.hibernate.annotations.CreationTimestamp;

import java.time.LocalDateTime;

@Getter
@Setter
@Entity
@Table(name = "session_turns",
        uniqueConstraints = @UniqueConstraint(columnNames = {"session_id", "turn_number"}))
@AllArgsConstructor
@NoArgsConstructor
public class SessionTurn {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "session_id", nullable = false)
    private AssessmentSession session;

    @Column(nullable = false)
    private int turnNumber;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "question_variant_id", nullable = false)
    private QuestionVariant questionVariant;

    // Copy of the question as asked, so later edits to the bank don't rewrite history
    @Column(nullable = false, columnDefinition = "TEXT")
    private String questionTextSnapshot;

    @Enumerated(EnumType.STRING)
    @Column(nullable = false)
    private ResponseType responseType;

    @Column(columnDefinition = "TEXT")
    private String responseTextEncrypted;

    private Float sentimentScore;

    @Column(nullable = false)
    private boolean crisisDetected;

    @CreationTimestamp
    @Column(nullable = false, updatable = false)
    private LocalDateTime createdAt;
}
