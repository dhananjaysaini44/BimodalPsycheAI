package com.bimodalpsyche.backend.session.repository;

import com.bimodalpsyche.backend.session.model.AudioFile;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.List;
import java.util.Optional;

public interface AudioFileRepository extends JpaRepository<AudioFile, Long> {
    List<AudioFile> findAllByTurn_Session_IdOrderByTurn_TurnNumberAsc(Long sessionId);

    Optional<AudioFile> findByTurn_Id(Long turnId);
}
