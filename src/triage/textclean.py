"""Text cleaning shared by the split code and the rare-class mining code.

The teacher and the small model must see exactly the same cleaned text (ADR-011), so these
functions live in one place.
"""

import re


def clean_text(text):
    """Clean a ticket so the teacher AND the small model see identical text everywhere (ADR-011)."""
    # The data contains the two characters backslash + n instead of a real new line. Fix those.
    text = text.replace("\\n", "\n")
    text = text.replace("\\t", " ")

    # HTML line breaks become real new lines.
    text = re.sub(r"<br\s*/?>", "\n", text)

    # Squeeze runs of spaces into one space, and runs of 3+ new lines into 2.
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def normalise(text):
    """A key for spotting near-duplicates: capital letters, punctuation and spacing do not make a new ticket."""
    lower = text.lower()
    without_punctuation = re.sub(r"[^\w\s]", " ", lower)
    words = without_punctuation.split()
    return " ".join(words)
