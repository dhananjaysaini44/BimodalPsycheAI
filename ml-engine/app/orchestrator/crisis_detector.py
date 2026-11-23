# NOTE: keep this list in a config file, not hardcoded, so it can be
# reviewed/updated without a code change. This is a known-limitation
# component — pattern matching alone will miss indirect risk signals.
RISK_PATTERNS = [
    "want to die",
    "kill myself",
    "suicide",
    "end my life",
    "better off dead"
]


def check_crisis_signal(text: str) -> bool:
    lowered = text.lower()
    return any(pattern in lowered for pattern in RISK_PATTERNS)
