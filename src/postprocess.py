import re


_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_MONTH_WORDS = {"january","february","march","april","may","june","july","august","september","october","november","december"}
_NAME_HINT = {"john","michael","sarah","david","james","mary","robert","patricia","linda","jennifer"}


def _digits(text: str) -> str:
    return "".join(ch for ch in text if ch.isdigit())


def validate_span(label: str, text: str) -> bool:
    """Lightweight precision-oriented validators.
    Return True if span should be kept.
    Only rejects obvious false positives; recall impact kept small.
    """
    lower = text.lower()
    if label == "EMAIL":
        return bool(_EMAIL_RE.fullmatch(text.strip()))
    if label == "CREDIT_CARD":
        d = _digits(text)
        # Typical card lengths: 13-16 digits. Reject if outside.
        return 13 <= len(d) <= 16
    if label == "PHONE":
        d = _digits(text)
        # Require at least 10 digits; reject very long unlikely sequences (>15)
        return 10 <= len(d) <= 15
    if label == "DATE":
        d = _digits(text)
        # Accept if contains month word or matches simple numeric date forms
        if any(m in lower for m in _MONTH_WORDS):
            return True
        # Numeric patterns like 2023 05 12 or 05 12 2023 etc.
        parts = [p for p in re.split(r"[^0-9]", text) if p]
        if len(parts) >= 2 and all(1 <= len(p) <= 4 for p in parts):
            return True
        # Fallback: if 6-8 digits (e.g., yyyymmdd) or 8 digits (ddmmyyyy)
        return len(d) in (6, 8)
    if label == "PERSON_NAME":
        # Keep most; light precision bump: must have >=2 chars and not purely digits
        if text.isdigit():
            return False
        tokens = lower.split()
        if any(t in _NAME_HINT for t in tokens):
            return True
        return len(text.strip()) >= 2
    # Non-PII categories: keep as-is
    return True
