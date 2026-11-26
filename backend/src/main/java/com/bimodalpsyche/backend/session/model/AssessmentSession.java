package com.bimodalpsyche.backend.session.model;

import com.bimodalpsyche.backend.question.model.QuestionVariant;
import com.bimodalpsyche.backend.question.model.QuestionBankVersion;
import com.bimodalpsyche.backend.user.model.User;
import jakarta.persistence.*;
import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;
import org.hibernate.annotations.CreationTimestamp;
import org.hibernate.annotations.UpdateTimestamp;

import java.time.LocalDateTime;

@Getter
@Setter
@Entity
@Table(name = "assessment_sessions")
@AllArgsConstructor
@NoArgsConstructor
public class AssessmentSession {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "user_id", nullable = false)
    private User user;

    @Enumerated(EnumType.STRING)
    @Column(nullable = false)
    private SessionStatus status;

    @Column(nullable = false)
    private int currentTurnNumber;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "current_question_variant_id")
    private QuestionVariant currentQuestionVariant;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "question_bank_version_id")
    private QuestionBankVersion questionBankVersion;

    private LocalDateTime startedAt;

    private LocalDateTime lastActivityAt;

    private LocalDateTime completedAt;

    @CreationTimestamp
    @Column(nullable = false, updatable = false)
    private LocalDateTime createdAt;

    @UpdateTimestamp
    @Column(nullable = false)
    private LocalDateTime updatedAt;
}
