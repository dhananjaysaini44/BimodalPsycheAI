package com.bimodalpsyche.backend.user.service;

import com.bimodalpsyche.backend.user.dto.HistoryDetailsRequest;
import com.bimodalpsyche.backend.user.model.FamilyHistory;
import com.bimodalpsyche.backend.user.model.MedicalHistory;
import com.bimodalpsyche.backend.user.model.User;
import com.bimodalpsyche.backend.user.repository.FamilyHistoryRepository;
import com.bimodalpsyche.backend.user.repository.MedicalHistoryRepository;
import com.bimodalpsyche.backend.user.repository.UserRepository;
import com.bimodalpsyche.backend.session.service.TextEncryptionService;
import org.springframework.security.core.userdetails.UsernameNotFoundException;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

@Service
public class UserHistoryService {
    private final UserRepository userRepository;
    private final MedicalHistoryRepository medicalHistoryRepository;
    private final FamilyHistoryRepository familyHistoryRepository;
    private final TextEncryptionService textEncryptionService;

    public UserHistoryService(
            UserRepository userRepository,
            MedicalHistoryRepository medicalHistoryRepository,
            FamilyHistoryRepository familyHistoryRepository,
            TextEncryptionService textEncryptionService) {
        this.userRepository = userRepository;
        this.medicalHistoryRepository = medicalHistoryRepository;
        this.familyHistoryRepository = familyHistoryRepository;
        this.textEncryptionService = textEncryptionService;
    }

    @Transactional
    public HistoryResponse getMedicalHistory(String email) {
        User user = findUser(email);
        return medicalHistoryRepository.findByUser(user)
                .map(history -> {
                    String details = textEncryptionService.decrypt(history.getDetails());
                    if (!textEncryptionService.isEncrypted(history.getDetails())) {
                        history.setDetails(textEncryptionService.encrypt(details));
                        medicalHistoryRepository.save(history);
                    }
                    return new HistoryResponse(details, true);
                })
                .orElseGet(() -> new HistoryResponse(null, false));
    }

    @Transactional
    public HistoryResponse saveMedicalHistory(String email, HistoryDetailsRequest request) {
        User user = lockUser(email);
        MedicalHistory history = medicalHistoryRepository.findByUser(user).orElseGet(() -> {
            MedicalHistory newHistory = new MedicalHistory();
            newHistory.setUser(user);
            return newHistory;
        });
        history.setDetails(textEncryptionService.encrypt(request.getDetails().trim()));
        medicalHistoryRepository.save(history);
        user.setMedicalProfileCompleted(true);
        return new HistoryResponse(request.getDetails().trim(), true);
    }

    @Transactional
    public HistoryResponse getFamilyHistory(String email) {
        User user = findUser(email);
        return familyHistoryRepository.findByUser(user)
                .map(history -> {
                    String details = textEncryptionService.decrypt(history.getDetails());
                    if (!textEncryptionService.isEncrypted(history.getDetails())) {
                        history.setDetails(textEncryptionService.encrypt(details));
                        familyHistoryRepository.save(history);
                    }
                    return new HistoryResponse(details, true);
                })
                .orElseGet(() -> new HistoryResponse(null, false));
    }

    @Transactional
    public HistoryResponse saveFamilyHistory(String email, HistoryDetailsRequest request) {
        User user = lockUser(email);
        FamilyHistory history = familyHistoryRepository.findByUser(user).orElseGet(() -> {
            FamilyHistory newHistory = new FamilyHistory();
            newHistory.setUser(user);
            return newHistory;
        });
        history.setDetails(textEncryptionService.encrypt(request.getDetails().trim()));
        familyHistoryRepository.save(history);
        user.setFamilyHistoryCompleted(true);
        return new HistoryResponse(request.getDetails().trim(), true);
    }

    private User findUser(String email) {
        return userRepository.findByEmail(email)
                .orElseThrow(() -> new UsernameNotFoundException("User not found"));
    }

    private User lockUser(String email) {
        return userRepository.findByEmailForUpdate(email)
                .orElseThrow(() -> new UsernameNotFoundException("User not found"));
    }

    public record HistoryResponse(String details, boolean completed) {}
}
