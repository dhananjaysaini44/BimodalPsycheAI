package com.bimodalpsyche.backend.session.service;

import com.bimodalpsyche.backend.session.model.AudioFile;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;
import java.nio.file.FileVisitResult;
import java.nio.file.Files;
import java.nio.file.NoSuchFileException;
import java.nio.file.Path;
import java.nio.file.SimpleFileVisitor;
import java.nio.file.StandardOpenOption;
import java.nio.file.attribute.BasicFileAttributes;
import java.time.LocalDateTime;
import java.util.Locale;
import java.util.Set;
import java.util.UUID;

@Service
public class AudioStorageService {
    private static final long DEFAULT_MAX_BYTES = 25L * 1024 * 1024;
    private static final Set<String> ALLOWED_TYPES = Set.of(
            "audio/webm", "audio/wav", "audio/x-wav", "audio/mpeg",
            "audio/mp4", "audio/ogg", "audio/aac");

    private final Path storageRoot;
    private final long maxBytes;
    private final TextEncryptionService encryptionService;

    public AudioStorageService(
            @Value("${app.audio.storage-directory:../audio_files}") String storageDirectory,
            @Value("${app.audio.max-size-bytes:26214400}") long maxBytes,
            TextEncryptionService encryptionService) {
        this.storageRoot = Path.of(storageDirectory).toAbsolutePath().normalize();
        this.maxBytes = maxBytes > 0 ? maxBytes : DEFAULT_MAX_BYTES;
        this.encryptionService = encryptionService;
    }

    public StoredAudio store(MultipartFile upload, Long sessionId, Long turnId) {
        if (upload == null || upload.isEmpty()) {
            throw new InvalidAudioUploadException("Audio file is required and must not be empty.");
        }
        if (upload.getSize() > maxBytes) {
            throw new InvalidAudioUploadException("Audio file exceeds the configured upload limit.");
        }
        String mimeType = normalizedMimeType(upload.getContentType());
        if (!ALLOWED_TYPES.contains(mimeType)) {
            throw new InvalidAudioUploadException("Unsupported audio type. Use WebM, WAV, MP3, MP4, OGG, or AAC.");
        }

        byte[] content;
        try {
            content = upload.getBytes();
        } catch (IOException ex) {
            throw new InvalidAudioUploadException("Audio file could not be read.", ex);
        }
        if (content.length == 0 || content.length > maxBytes || !matchesSignature(mimeType, content)) {
            throw new InvalidAudioUploadException("Audio file content does not match its declared audio type.");
        }

        String relativePath = sessionId + "/" + turnId + "/audio.enc";
        Path destination = resolve(relativePath);
        try {
            Files.createDirectories(destination.getParent());
            Files.write(destination, encryptionService.encryptBytes(content), StandardOpenOption.CREATE_NEW);
        } catch (IOException ex) {
            throw new IllegalStateException("Audio could not be stored.", ex);
        }
        return new StoredAudio(relativePath, safeFilename(upload.getOriginalFilename()), mimeType, content.length);
    }

    public byte[] read(String relativePath) {
        try {
            return encryptionService.decryptBytes(Files.readAllBytes(resolve(relativePath)));
        } catch (IOException ex) {
            throw new IllegalStateException("Stored audio could not be read.", ex);
        }
    }

    public void delete(String relativePath) {
        try {
            Files.deleteIfExists(resolve(relativePath));
        } catch (IOException ex) {
            throw new IllegalStateException("Stored audio could not be removed.", ex);
        }
    }

    public void deleteSessionAudio(Long sessionId) {
        Path sessionDirectory = storageRoot.resolve(sessionId.toString()).normalize();
        if (!sessionDirectory.startsWith(storageRoot) || sessionDirectory.equals(storageRoot)) {
            throw new IllegalArgumentException("Invalid session audio directory");
        }
        try {
            Files.walkFileTree(sessionDirectory, new SimpleFileVisitor<>() {
                @Override
                public FileVisitResult visitFile(Path file, BasicFileAttributes attributes) throws IOException {
                    Files.delete(file);
                    return FileVisitResult.CONTINUE;
                }

                @Override
                public FileVisitResult postVisitDirectory(Path directory, IOException exception) throws IOException {
                    if (exception != null) {
                        throw exception;
                    }
                    Files.delete(directory);
                    return FileVisitResult.CONTINUE;
                }
            });
        } catch (NoSuchFileException ignored) {
            if (ignored.getFile() != null
                    && Path.of(ignored.getFile()).toAbsolutePath().normalize().equals(sessionDirectory)) {
                return;
            }
            throw new IllegalStateException("Session audio files could not be removed.", ignored);
        } catch (IOException ex) {
            throw new IllegalStateException("Session audio files could not be removed.", ex);
        }
    }

    private Path resolve(String relativePath) {
        Path resolved = storageRoot.resolve(relativePath).normalize();
        if (!resolved.startsWith(storageRoot)) {
            throw new IllegalArgumentException("Invalid audio storage path");
        }
        return resolved;
    }

    private static String normalizedMimeType(String contentType) {
        if (contentType == null) {
            return "";
        }
        return contentType.split(";", 2)[0].trim().toLowerCase(Locale.ROOT);
    }

    private static String safeFilename(String filename) {
        if (filename == null || filename.isBlank()) {
            return "recording";
        }
        String normalized = filename.replace('\\', '/');
        String basename = normalized.substring(normalized.lastIndexOf('/') + 1)
                .replaceAll("[^A-Za-z0-9._-]", "_");
        return basename.isBlank() ? "recording" : basename.substring(0, Math.min(basename.length(), 255));
    }

    private static boolean matchesSignature(String mimeType, byte[] bytes) {
        return switch (mimeType) {
            case "audio/wav", "audio/x-wav" -> bytes.length >= 12
                    && ascii(bytes, 0, "RIFF") && ascii(bytes, 8, "WAVE");
            case "audio/webm" -> bytes.length >= 4 && (bytes[0] & 0xFF) == 0x1A
                    && (bytes[1] & 0xFF) == 0x45 && (bytes[2] & 0xFF) == 0xDF && (bytes[3] & 0xFF) == 0xA3;
            case "audio/ogg" -> bytes.length >= 4 && ascii(bytes, 0, "OggS");
            case "audio/mp4" -> bytes.length >= 8 && ascii(bytes, 4, "ftyp");
            case "audio/mpeg" -> bytes.length >= 3 && (ascii(bytes, 0, "ID3")
                    || ((bytes[0] & 0xFF) == 0xFF && (bytes[1] & 0xE0) == 0xE0));
            case "audio/aac" -> bytes.length >= 2 && (bytes[0] & 0xFF) == 0xFF
                    && (bytes[1] & 0xF6) == 0xF0;
            default -> false;
        };
    }

    private static boolean ascii(byte[] bytes, int offset, String value) {
        if (bytes.length < offset + value.length()) {
            return false;
        }
        for (int i = 0; i < value.length(); i++) {
            if (bytes[offset + i] != (byte) value.charAt(i)) {
                return false;
            }
        }
        return true;
    }

    public record StoredAudio(String storagePath, String originalFilename, String mimeType, long sizeBytes) {
    }

    public static class InvalidAudioUploadException extends RuntimeException {
        public InvalidAudioUploadException(String message) {
            super(message);
        }

        public InvalidAudioUploadException(String message, Throwable cause) {
            super(message, cause);
        }
    }
}
