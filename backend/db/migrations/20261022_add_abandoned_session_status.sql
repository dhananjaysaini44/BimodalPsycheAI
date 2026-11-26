ALTER TABLE assessment_sessions
    MODIFY COLUMN status ENUM(
        'COMPLETED',
        'DRAFT',
        'IN_PROGRESS',
        'PAUSED',
        'ABANDONED'
    ) NOT NULL;
