package com.bimodalpsyche.backend.session.service;

import org.junit.jupiter.api.Test;

import java.util.Base64;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class TextEncryptionServiceTest {
    private final TextEncryptionService service = new TextEncryptionService(
            Base64.getEncoder().encodeToString(new byte[32]));

    @Test
    void encryptsWithRandomNonceAndCanDecrypt() {
        String plaintext = "Sensitive assessment response";

        String first = service.encrypt(plaintext);
        String second = service.encrypt(plaintext);

        assertTrue(first.startsWith("v1:"));
        assertNotEquals(plaintext, first);
        assertNotEquals(first, second);
        assertEquals(plaintext, service.decrypt(first));
    }

    @Test
    void rejectsTamperedCiphertext() {
        String encrypted = service.encrypt("private response");
        String tampered = encrypted.substring(0, encrypted.length() - 2) + "AA";

        assertThrows(IllegalArgumentException.class, () -> service.decrypt(tampered));
    }

    @Test
    void requiresA256BitBase64Key() {
        assertThrows(IllegalArgumentException.class, () -> new TextEncryptionService("not-a-key"));
        assertThrows(IllegalArgumentException.class, () -> new TextEncryptionService(
                Base64.getEncoder().encodeToString(new byte[16])));
    }
}
