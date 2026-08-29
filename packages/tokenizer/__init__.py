"""packages.tokenizer — real trainable byte-level BPE tokenizer.

Byte-level so any text (incl. Amharic/Ethiopic, emoji, binary) is representable
without an unknown token. Vocabulary is versioned and persisted as JSON.
"""
from .tokenizer import BPETokenizer

__all__ = ["BPETokenizer"]
