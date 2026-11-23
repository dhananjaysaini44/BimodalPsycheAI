package com.bimodalpsyche.backend.session.service;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import javax.crypto.Cipher;
import javax.crypto.spec.GCMParameterSpec;
import javax.crypto.spec.SecretKeySpec;
import java.nio.charset.StandardCharsets;
import java.security.GeneralSecurityException;
import java.security.SecureRandom;
import java.util.Arrays;
import java.util.Base64;

@Service
public class TextEncryptionService {
    private static final String CIPHER = "AES/GCM/NoPadding";
    private static final String PREFIX = "v1:";
    private static final int NONCE_LENGTH_BYTES = 12;
    private static final int TAG_LENGTH_BITS = 128;

    private final SecretKeySpec key;
    private final SecureRandom secureRandom = new SecureRandom();

    public TextEncryptionService(@Value("${app.encryption.key}") String base64Key) {
        byte[] keyBytes;
        try {
            keyBytes = Base64.getDecoder().decode(base64Key);
        } catch (IllegalArgumentException ex) {
            throw new IllegalArgumentException("ENCRYPTION_KEY must be Base64-encoded AES-256 key material", ex);
        }
        if (keyBytes.length != 32) {
            Arrays.fill(keyBytes, (byte) 0);
            throw new IllegalArgumentException("ENCRYPTION_KEY must decode to exactly 32 bytes");
        }
        this.key = new SecretKeySpec(keyBytes, "AES");
        Arrays.fill(keyBytes, (byte) 0);
    }

    public String encrypt(String plaintext) {
        return PREFIX + Base64.getEncoder().encodeToString(encryptBytes(plaintext.getBytes(StandardCharsets.UTF_8)));
    }

    public byte[] encryptBytes(byte[] plaintext) {
        byte[] nonce = new byte[NONCE_LENGTH_BYTES];
        secureRandom.nextBytes(nonce);
        try {
            Cipher cipher = Cipher.getInstance(CIPHER);
            cipher.init(Cipher.ENCRYPT_MODE, key, new GCMParameterSpec(TAG_LENGTH_BITS, nonce));
            byte[] encrypted = cipher.doFinal(plaintext);
            byte[] payload = new byte[nonce.length + encrypted.length];
            System.arraycopy(nonce, 0, payload, 0, nonce.length);
            System.arraycopy(encrypted, 0, payload, nonce.length, encrypted.length);
            return payload;
        } catch (GeneralSecurityException ex) {
            throw new IllegalStateException("Unable to encrypt session response", ex);
        }
    }

    public byte[] decryptBytes(byte[] payload) {
        if (payload == null || payload.length <= NONCE_LENGTH_BYTES) {
            throw new IllegalArgumentException("Invalid encrypted response format");
        }
        byte[] nonce = Arrays.copyOf(payload, NONCE_LENGTH_BYTES);
        byte[] ciphertext = Arrays.copyOfRange(payload, NONCE_LENGTH_BYTES, payload.length);
        try {
            Cipher cipher = Cipher.getInstance(CIPHER);
            cipher.init(Cipher.DECRYPT_MODE, key, new GCMParameterSpec(TAG_LENGTH_BITS, nonce));
            return cipher.doFinal(ciphertext);
        } catch (GeneralSecurityException ex) {
            throw new IllegalArgumentException("Encrypted response could not be authenticated", ex);
        }
    }

    public String decrypt(String encryptedText) {
        if (encryptedText == null) {
            throw new IllegalArgumentException("Encrypted value must not be null");
        }
        if (!encryptedText.startsWith(PREFIX)) {
            return encryptedText;
        }
        byte[] payload;
        try {
            payload = Base64.getDecoder().decode(encryptedText.substring(PREFIX.length()));
        } catch (IllegalArgumentException ex) {
            throw new IllegalArgumentException("Invalid encrypted response format", ex);
        }
        if (payload.length <= NONCE_LENGTH_BYTES) {
            throw new IllegalArgumentException("Invalid encrypted response format");
        }
        byte[] nonce = Arrays.copyOf(payload, NONCE_LENGTH_BYTES);
        byte[] ciphertext = Arrays.copyOfRange(payload, NONCE_LENGTH_BYTES, payload.length);
        try {
            Cipher cipher = Cipher.getInstance(CIPHER);
            cipher.init(Cipher.DECRYPT_MODE, key, new GCMParameterSpec(TAG_LENGTH_BITS, nonce));
            return new String(cipher.doFinal(ciphertext), StandardCharsets.UTF_8);
        } catch (GeneralSecurityException ex) {
            throw new IllegalArgumentException("Encrypted response could not be authenticated", ex);
        }
    }

    public boolean isEncrypted(String value) {
        return value != null && value.startsWith(PREFIX);
    }
}
