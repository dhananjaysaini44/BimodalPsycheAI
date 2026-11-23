package com.bimodalpsyche.backend.user.service;

import com.bimodalpsyche.backend.session.service.TextEncryptionService;
import com.bimodalpsyche.backend.user.model.MedicalHistory;
import com.bimodalpsyche.backend.user.model.User;
import com.bimodalpsyche.backend.user.repository.FamilyHistoryRepository;
import com.bimodalpsyche.backend.user.repository.MedicalHistoryRepository;
import com.bimodalpsyche.backend.user.repository.UserRepository;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import java.util.Base64;
import java.util.Optional;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

@ExtendWith(MockitoExtension.class)
class UserHistoryServiceTest {
    private static final String EMAIL = "person@example.com";

    @Mock
    private UserRepository userRepository;
    @Mock
    private MedicalHistoryRepository medicalHistoryRepository;
    @Mock
    private FamilyHistoryRepository familyHistoryRepository;

    @Test
    void returnsLegacyHistoryAndEncryptsItForMigration() {
        User user = new User();
        MedicalHistory history = new MedicalHistory();
        history.setUser(user);
        history.setDetails("Legacy plaintext history");
        when(userRepository.findByEmail(EMAIL)).thenReturn(Optional.of(user));
        when(medicalHistoryRepository.findByUser(user)).thenReturn(Optional.of(history));
        UserHistoryService service = service();

        UserHistoryService.HistoryResponse response = service.getMedicalHistory(EMAIL);

        assertEquals("Legacy plaintext history", response.details());
        assertTrue(response.completed());
        assertEquals("v1:", history.getDetails().substring(0, 3));
        verify(medicalHistoryRepository).save(history);
    }

    @Test
    void returnsEncryptedHistoryAsPlaintext() {
        User user = new User();
        MedicalHistory history = new MedicalHistory();
        history.setUser(user);
        TextEncryptionService encryptionService = encryptionService();
        history.setDetails(encryptionService.encrypt("Encrypted history"));
        when(userRepository.findByEmail(EMAIL)).thenReturn(Optional.of(user));
        when(medicalHistoryRepository.findByUser(user)).thenReturn(Optional.of(history));

        UserHistoryService.HistoryResponse response = service(encryptionService).getMedicalHistory(EMAIL);

        assertEquals("Encrypted history", response.details());
        assertTrue(response.completed());
    }

    private UserHistoryService service() {
        return service(encryptionService());
    }

    private UserHistoryService service(TextEncryptionService encryptionService) {
        return new UserHistoryService(
                userRepository, medicalHistoryRepository, familyHistoryRepository, encryptionService);
    }

    private TextEncryptionService encryptionService() {
        return new TextEncryptionService(Base64.getEncoder().encodeToString(new byte[32]));
    }
}
