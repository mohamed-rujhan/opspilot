"""Guardrails applied around the agent.

* Input:  length limit and prompt-injection detection before the request reaches the LLM.
* Context: retrieved document chunks that contain injection-style instructions are dropped.
* Output: system-prompt leakage is blocked and sensitive identifiers are redacted.
* Logs:   PII is redacted before anything is logged.
"""

import re
from dataclasses import dataclass

INJECTION_PATTERNS = [
    r"ignore\s+(all\s+|any\s+)?(the\s+)?(previous|prior|above|earlier)\s+(instructions|prompts|rules)",
    r"disregard\s+(all\s+|the\s+|your\s+)?(previous|prior|system|above)",
    r"(reveal|show|print|repeat|leak)\s+(me\s+)?(your|the)\s+(system\s+|hidden\s+)?(prompt|instructions)",
    r"\byou\s+are\s+now\b",
    r"\b(jailbreak|jailbroken|developer\s+mode|dan\s+mode)\b",
    r"act\s+as\s+(an?\s+)?(unrestricted|unfiltered|evil)",
    r"pretend\s+(that\s+)?you\s+(have\s+no|don't\s+have)\s+(rules|restrictions)",
]
_INJECTION_RE = re.compile("|".join(INJECTION_PATTERNS), re.IGNORECASE)

# Order matters: card and NIC numbers before generic phone numbers.
PII_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("EMAIL", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")),
    ("CARD", re.compile(r"\b(?:\d[ -]?){13,16}\b")),
    ("NIC", re.compile(r"\b(?:\d{9}[VvXx]|\d{12})\b")),
    ("PHONE", re.compile(r"(?<!\w)\+?(?:\d[\s-]?){9,14}\d\b")),
]

# A unique marker placed in the system prompt; if it ever appears in an answer, the prompt has leaked.
SYSTEM_PROMPT_CANARY = "opspilot-canary-5c1e"

BLOCKED_INPUT_MESSAGE = (
    "I can't help with that request. I can answer questions about company HR and IT policies, "
    "raise or check IT tickets, look up public holidays and submit leave requests."
)


@dataclass(frozen=True)
class GuardrailResult:
    allowed: bool
    reason: str | None = None


def check_input(text: str, max_chars: int = 2000) -> GuardrailResult:
    if not text or not text.strip():
        return GuardrailResult(False, "empty_input")
    if len(text) > max_chars:
        return GuardrailResult(False, "input_too_long")
    if _INJECTION_RE.search(text):
        return GuardrailResult(False, "prompt_injection")
    return GuardrailResult(True)


def contains_injection(text: str) -> bool:
    return bool(_INJECTION_RE.search(text))


def redact_pii(text: str) -> str:
    for label, pattern in PII_PATTERNS:
        text = pattern.sub(f"[{label}]", text)
    return text


def check_output(text: str) -> tuple[str, GuardrailResult]:
    """Return the text that is safe to show the user, and whether it was changed."""
    if SYSTEM_PROMPT_CANARY in text:
        return BLOCKED_INPUT_MESSAGE, GuardrailResult(False, "system_prompt_leak")
    cleaned = PII_PATTERNS[1][1].sub("[CARD]", text)
    cleaned = PII_PATTERNS[2][1].sub("[NIC]", cleaned)
    if cleaned != text:
        return cleaned, GuardrailResult(False, "sensitive_identifier_redacted")
    return text, GuardrailResult(True)
