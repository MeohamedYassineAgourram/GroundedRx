"""Context compression / distillation (Layer 2: the token-efficiency lever).

Reduce each retrieved chunk to its high-signal clinical core. This is a cheap,
deterministic extractive compressor (no model call): keep the sentence(s) carrying the
key fact. Danger-sign passages are kept essentially intact (safety > brevity). This is
what drives the efficiency frontier -- accuracy held while token budget shrinks.
"""
from __future__ import annotations

import re

_STOP_TAIL = re.compile(r"\s*(,|;|:)\s.*$")  # trim trailing clauses


def _first_sentence(text):
    m = re.split(r"(?<=[.!?])\s+", text.strip())
    return m[0] if m else text


def compress_passage(p, enabled=True):
    """Return a (possibly) shortened copy of a passage's text."""
    if not enabled:
        return p["text"]
    if p["type"] == "danger_sign":
        # Keep danger signs nearly whole -- never compress away a safety fact.
        return _first_sentence(p["text"])
    if p["type"] == "medication":
        # Keep the drug's purpose + primary caution: first sentence is the core.
        return _first_sentence(p["text"])
    # lifestyle / other: first sentence, trailing clause trimmed.
    return _STOP_TAIL.sub(".", _first_sentence(p["text"]))


def compress_context(passages, enabled=True):
    """Return list of (passage, compressed_text). Non-destructive to originals."""
    return [(p, compress_passage(p, enabled)) for p in passages]
