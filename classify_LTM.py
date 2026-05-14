import re

DROP_PATTERNS = [
    r"\bHUMAN_MESSAGE\b",
    r"\bASSISTANT\b",
    r"\bUSER\b",
    r"\bI remember\b",
    r"\byou told me\b",
    r"\bas we discussed\b",
    r"\bmy persona\b",
    r"\byour persona\b",
    r"\bchat\b",
    r"\bsession\b",
    r"\btimestamp\b",
]

SECRET_PATTERNS = [
    r"sk-[A-Za-z0-9_\-]{20,}",
    r"(?i)\b(api[_-]?key|secret|token|password)\b",
    r"(?i)\btelegram bot token\b",
    r"(?i)\bauthorization:\s*bearer\b",
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
]

def classify_text(text: str) -> str:
    t = text or ""

    if any(re.search(p, t) for p in SECRET_PATTERNS):
        return "DROP"

    if any(re.search(p, t, flags=re.IGNORECASE) for p in DROP_PATTERNS):
        return "QUARANTINE"

    if len(t.strip()) < 40:
        return "DROP"

    return "DISTILL"