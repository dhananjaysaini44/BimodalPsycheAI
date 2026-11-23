package com.bimodalpsyche.backend.user.repository;

import com.bimodalpsyche.backend.user.model.FamilyHistory;
import com.bimodalpsyche.backend.user.model.User;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.Optional;

public interface FamilyHistoryRepository extends JpaRepository<FamilyHistory, Long> {
    Optional<FamilyHistory> findByUser(User user);
}
