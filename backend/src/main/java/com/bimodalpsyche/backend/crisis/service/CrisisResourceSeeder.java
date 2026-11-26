package com.bimodalpsyche.backend.crisis.service;

import com.bimodalpsyche.backend.crisis.model.CrisisResource;
import com.bimodalpsyche.backend.crisis.repository.CrisisResourceRepository;
import org.jspecify.annotations.NonNull;
import org.springframework.boot.ApplicationArguments;
import org.springframework.boot.ApplicationRunner;
import org.springframework.stereotype.Component;
import org.springframework.transaction.annotation.Transactional;

@Component
public class CrisisResourceSeeder implements ApplicationRunner {
    private static final String RESOURCE_NAME = "Tele-MANAS";
    private static final String COUNTRY = "India";

    private final CrisisResourceRepository repository;

    public CrisisResourceSeeder(CrisisResourceRepository repository) {
        this.repository = repository;
    }

    @Override
    @Transactional
    public void run(@NonNull ApplicationArguments args) {
        if (repository.existsByNameAndCountry(RESOURCE_NAME, COUNTRY)) {
            return;
        }

        CrisisResource resource = new CrisisResource();
        resource.setName(RESOURCE_NAME);
        resource.setDescription("Government of India's 24/7 tele-mental health support service.");
        resource.setPhone("14416");
        resource.setUrl("https://telemanas.mohfw.gov.in/");
        resource.setCountry(COUNTRY);
        resource.setResourceType("MENTAL_HEALTH_HELPLINE");
        resource.setActive(true);
        resource.setPriority(1);
        repository.save(resource);
    }
}
