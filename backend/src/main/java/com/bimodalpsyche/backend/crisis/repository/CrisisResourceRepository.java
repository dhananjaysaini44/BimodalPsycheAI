package com.bimodalpsyche.backend.crisis.repository;

import com.bimodalpsyche.backend.crisis.model.CrisisResource;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.List;

public interface CrisisResourceRepository extends JpaRepository<CrisisResource, Long> {
    List<CrisisResource> findAllByActiveTrueOrderByPriorityAsc();

    boolean existsByNameAndCountry(String name, String country);
}
