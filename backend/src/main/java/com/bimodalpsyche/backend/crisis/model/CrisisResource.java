package com.bimodalpsyche.backend.crisis.model;

import jakarta.persistence.*;
import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

@Getter
@Setter
@Entity
@Table(name = "crisis_resources")
@AllArgsConstructor
@NoArgsConstructor
public class CrisisResource {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(nullable = false)
    private String name;

    @Column(length = 1000)
    private String description;

    @Column(length = 30)
    private String phone;

    @Column(length = 500)
    private String url;

    @Column(length = 100)
    private String country;

    @Column(length = 50)
    private String resourceType;

    @Column(nullable = false)
    private boolean active;

    @Column(nullable = false)
    private int priority;
}
