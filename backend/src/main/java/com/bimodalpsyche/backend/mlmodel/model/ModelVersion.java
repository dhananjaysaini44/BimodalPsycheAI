package com.bimodalpsyche.backend.mlmodel.model;

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
@Table(name = "model_versions")
@AllArgsConstructor
@NoArgsConstructor
public class ModelVersion {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Enumerated(EnumType.STRING)
    @Column(nullable = false)
    private ModelModality modality;

    @Column(nullable = false, length = 100)
    private String modelName;

    @Column(nullable = false, length = 50)
    private String version;

    @Column(length = 128)
    private String configurationHash;

    @Column(nullable = false)
    private boolean active;

    @CreationTimestamp
    @Column(nullable = false, updatable = false)
    private LocalDateTime createdAt;
}
