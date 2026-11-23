package com.bimodalpsyche.backend.session.repository;

import com.bimodalpsyche.backend.crisis.model.CrisisEvent;
import org.springframework.data.jpa.repository.JpaRepository;

public interface CrisisEventRepository extends JpaRepository<CrisisEvent, Long> {
    void deleteAllBySession_Id(Long sessionId);
}
