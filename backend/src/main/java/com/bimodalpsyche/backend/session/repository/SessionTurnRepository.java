package com.bimodalpsyche.backend.session.repository;

import com.bimodalpsyche.backend.session.model.SessionTurn;
import org.springframework.data.jpa.repository.JpaRepository;

public interface SessionTurnRepository extends JpaRepository<SessionTurn, Long> {
    java.util.List<SessionTurn> findAllBySession_IdOrderByTurnNumberAsc(Long sessionId);

    java.util.List<SessionTurn> findAllBySession_IdAndResponseTypeOrderByTurnNumberAsc(
            Long sessionId, com.bimodalpsyche.backend.session.model.ResponseType responseType);
}
