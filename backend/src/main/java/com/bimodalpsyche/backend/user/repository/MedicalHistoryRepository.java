package com.bimodalpsyche.backend.user.repository;

import com.bimodalpsyche.backend.user.model.MedicalHistory;
import com.bimodalpsyche.backend.user.model.User;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.Optional;

public interface MedicalHistoryRepository extends JpaRepository<MedicalHistory, Long> {
    Optional<MedicalHistory> findByUser(User user);
}
