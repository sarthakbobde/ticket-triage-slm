"""Text cleaning shared by the split script and the rare-class mining script.

The teacher and the student must see exactly the same cleaned text (ADR-011), so these
functions live in one place.
"""

import re


def clean_text(text: str) -> str:
    """What the teacher AND the student will see, so it must be identical everywhere (ADR-011)."""
    text = text.replace("\\n", "\n").replace("\\t", " ")  # literal escape sequences in the data
    text = re.sub(r"<br\s*/?>", "\n", text)  # HTML line breaks
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def normalise(text: str) -> str:
    """Key for near-duplicate detection: case, punctuation and spacing don't make a new ticket."""
    return " ".join(re.sub(r"[^\w\s]", " ", text.lower()).split())
