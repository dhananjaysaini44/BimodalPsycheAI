package com.bimodalpsyche.backend.session.model;

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
@Table(name = "audio_files")
@AllArgsConstructor
@NoArgsConstructor
public class AudioFile {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "turn_id", nullable = false)
    private SessionTurn turn;

    @Column(nullable = false, length = 500)
    private String storagePath;

    private String originalFilename;

    @Column(nullable = false, length = 100)
    private String mimeType;

    private Long fileSize;

    private Integer durationMs;

    @Column(length = 50)
    private String encryptionVersion;

    private LocalDateTime encryptedAt;

    @CreationTimestamp
    @Column(nullable = false, updatable = false)
    private LocalDateTime createdAt;

    private LocalDateTime expiresAt;

    private LocalDateTime deletedAt;
}
