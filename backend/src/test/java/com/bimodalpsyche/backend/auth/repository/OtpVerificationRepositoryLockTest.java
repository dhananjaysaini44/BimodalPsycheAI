package com.bimodalpsyche.backend.auth.repository;

import jakarta.persistence.LockModeType;
import org.junit.jupiter.api.Test;
import org.springframework.data.jpa.repository.Lock;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;

class OtpVerificationRepositoryLockTest {
    @Test
    void emailUpdateLookupUsesPessimisticWriteLock() throws NoSuchMethodException {
        Lock lock = OtpVerificationRepository.class
                .getMethod("findByEmailForUpdate", String.class)
                .getAnnotation(Lock.class);

        assertNotNull(lock);
        assertEquals(LockModeType.PESSIMISTIC_WRITE, lock.value());
    }
}
