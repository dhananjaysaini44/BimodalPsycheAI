package com.bimodalpsyche.backend.session.repository;

import com.bimodalpsyche.backend.session.model.TurnAnalysis;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.Collection;

public interface TurnAnalysisRepository extends JpaRepository<TurnAnalysis, Long> {
    void deleteAllByTurn_IdIn(Collection<Long> turnIds);
}
