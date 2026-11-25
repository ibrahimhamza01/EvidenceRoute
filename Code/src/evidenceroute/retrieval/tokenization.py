"""Deterministic lexical tokenization for BM25.

Lowercased alphanumeric runs. Identifiers such as ``H100``, ``MI300X``, ``3A090``
and ``1A`` stay intact. Typographic apostrophes are normalized and possessive
``'s`` is dropped. No stemming, so exact-term queries match exact terms.
"""
from __future__ import annotations

import re
import unicodedata

TOKEN_RE = re.compile(r"[a-z0-9]+")

STOPWORDS = frozenset(
    """
    a an and are as at be been being but by can could did do does for from had has
    have how if in into is it its may might more most of on or our such than that
    the their them then there these they this those to was we were what when where
    which while who whom why will with would you your
    """.split()
)


def tokenize(text: str, *, remove_stopwords: bool = True) -> list[str]:
    text = unicodedata.normalize("NFKC", text).lower()
    text = text.replace("’", "'").replace("‘", "'")
    text = re.sub(r"'s\b", "", text)
    tokens = TOKEN_RE.findall(text)
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS]
    return tokens
