package com.bimodalpsyche.backend.session.service;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.springframework.mock.web.MockMultipartFile;

import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Base64;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class AudioStorageServiceTest {
    @TempDir
    Path tempDir;

    private final TextEncryptionService encryptionService = new TextEncryptionService(
            Base64.getEncoder().encodeToString(new byte[32]));

    @Test
    void encryptsAudioAtRestAndDecryptsItForAnalysis() throws Exception {
        byte[] wav = "RIFF0000WAVE".getBytes(StandardCharsets.US_ASCII);
        AudioStorageService service = new AudioStorageService(tempDir.toString(), 1024, encryptionService);

        AudioStorageService.StoredAudio stored = service.store(
                new MockMultipartFile("file", "sample.wav", "audio/wav", wav), 7L, 9L);
        Path persisted = tempDir.resolve(stored.storagePath());

        assertEquals(Path.of("7", "9", "audio.enc"), Path.of(stored.storagePath()));
        assertNotEquals(Arrays.toString(wav), Arrays.toString(Files.readAllBytes(persisted)));
        assertArrayEquals(wav, service.read(stored.storagePath()));
        assertEquals("sample.wav", stored.originalFilename());
    }

    @Test
    void rejectsAudioWhoseDeclaredTypeDoesNotMatchItsSignature() {
        AudioStorageService service = new AudioStorageService(tempDir.toString(), 1024, encryptionService);
        MockMultipartFile upload = new MockMultipartFile(
                "file", "fake.wav", "audio/wav", "not audio".getBytes(StandardCharsets.US_ASCII));

        assertThrows(AudioStorageService.InvalidAudioUploadException.class,
                () -> service.store(upload, 7L, 9L));
    }

    @Test
    void rejectsPathsOutsideTheStorageRoot() {
        AudioStorageService service = new AudioStorageService(tempDir.toString(), 1024, encryptionService);

        assertThrows(IllegalArgumentException.class, () -> service.read("../outside.enc"));
    }

    @Test
    void removesAudioFilesForOneSessionOnly() throws Exception {
        AudioStorageService service = new AudioStorageService(tempDir.toString(), 1024, encryptionService);
        Path sessionAudio = tempDir.resolve("7").resolve("9");
        Path anotherSessionAudio = tempDir.resolve("8").resolve("9").resolve("audio.enc");
        Files.createDirectories(sessionAudio);
        Files.createDirectories(anotherSessionAudio.getParent());
        Files.write(sessionAudio.resolve("audio.enc"), new byte[]{1});
        Files.write(anotherSessionAudio, new byte[]{2});

        service.deleteSessionAudio(7L);

        assertFalse(Files.exists(tempDir.resolve("7")));
        assertTrue(Files.exists(anotherSessionAudio));
    }
}
