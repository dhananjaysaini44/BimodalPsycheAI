package com.bimodalpsyche.backend.crisis.service;

import com.bimodalpsyche.backend.crisis.model.CrisisResource;
import com.bimodalpsyche.backend.crisis.repository.CrisisResourceRepository;
import org.junit.jupiter.api.Test;
import org.mockito.ArgumentCaptor;
import org.springframework.boot.DefaultApplicationArguments;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;
import static org.mockito.Mockito.mock;

class CrisisResourceSeederTest {
    private final CrisisResourceRepository repository = mock(CrisisResourceRepository.class);

    @Test
    void seedsIndiaTeleManasResourceWhenMissing() {
        when(repository.existsByNameAndCountry("Tele-MANAS", "India")).thenReturn(false);

        new CrisisResourceSeeder(repository).run(new DefaultApplicationArguments());

        ArgumentCaptor<CrisisResource> captor = ArgumentCaptor.forClass(CrisisResource.class);
        verify(repository).save(captor.capture());
        CrisisResource saved = captor.getValue();
        assertEquals("14416", saved.getPhone());
        assertEquals("India", saved.getCountry());
        assertEquals("https://telemanas.mohfw.gov.in/", saved.getUrl());
        assertTrue(saved.isActive());
    }

    @Test
    void doesNotDuplicateExistingIndiaTeleManasResource() {
        when(repository.existsByNameAndCountry("Tele-MANAS", "India")).thenReturn(true);

        new CrisisResourceSeeder(repository).run(new DefaultApplicationArguments());

        verify(repository, never()).save(org.mockito.ArgumentMatchers.any(CrisisResource.class));
    }
}
