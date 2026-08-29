"""Real byte-level BPE tokenizer (GPT-2 style), trainable and versioned.

- Operates on UTF-8 bytes (0..255) so every Unicode codepoint is coverable,
  including Amharic/Ethiopic script. No <unk> needed for encoding.
- Special tokens: <pad>=0? No — pad id configurable. We reserve:
    0: <pad>
    1: <bos>
    2: <eos>
    3: <unk>  (kept for API compat; encoding never produces it for text)
  Byte/merge vocab follows after special tokens.
- Versioned: each trained tokenizer stores version, vocab, merges, config.
- Persisted as JSON; load/save are exact round-trips.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterable, Optional

# Special tokens (fixed ids)
PAD_TOKEN = "<pad>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"
SPECIAL_TOKENS = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN]
PAD_ID, BOS_ID, EOS_ID, UNK_ID = 0, 1, 2, 3

# Byte->visible-char mapping (GPT-2 trick) so merges work on a printable alphabet.
def _build_bytes_to_unicode() -> dict:
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


class ByteEncoder:
    def __init__(self):
        self.b2u = _build_bytes_to_unicode()
        self.u2b = {v: k for k, v in self.b2u.items()}

    def encode(self, text: str) -> str:
        return "".join(self.b2u[b] for b in text.encode("utf-8"))

    def decode(self, text: str) -> str:
        b = bytes(self.u2b[c] for c in text)
        return b.decode("utf-8", errors="replace")


# Pre-tokenization regex (GPT-2 style). Python's `re` lacks \p{} support, so we
# use \w/\d character classes which are Unicode-aware under re.UNICODE (default in
# Py3). Letters -> [^\W\d_], numbers -> \d, else -> non-space non-word.
GPT2_SPLIT = re.compile(
    r"""'s|'t|'re|'ve|'m|'ll|'d| ?[^\W\d_]+| ?\d+| ?[^\s\w]+|\s+(?!\S)|\s+""",
    re.UNICODE,
)


@dataclass
class TokenizerInfo:
    version: str
    vocab_size: int
    num_merges: int
    created_from_tokens: int = 0


class BPETokenizer:
    """Trainable byte-level BPE tokenizer."""

    def __init__(self, version: str = "0.1.0"):
        self.version = version
        self.byte_encoder = ByteEncoder()
        # vocab maps token-string -> id; ids 0..3 special, 4..259 byte tokens, then merges
        self.vocab: dict[str, int] = {}
        self.merges: list[list[str]] = []  # ordered list of [a,b] pairs
        self._merges_rank: dict[tuple[str, str], int] = {}
        self._byte_tokens: list[str] = []
        self._build_base_vocab()
        self._rebuild_id_map()
        self.info = TokenizerInfo(version=version, vocab_size=len(self.vocab), num_merges=0)

    def _build_base_vocab(self) -> None:
        self.vocab = {}
        for i, tok in enumerate(SPECIAL_TOKENS):
            self.vocab[tok] = i
        self._byte_tokens = [self.byte_encoder.b2u[i] for i in range(256)]
        for i, tok in enumerate(self._byte_tokens):
            self.vocab[tok] = len(SPECIAL_TOKENS) + i

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @property
    def pad_id(self) -> int:
        return PAD_ID

    @property
    def bos_id(self) -> int:
        return BOS_ID

    @property
    def eos_id(self) -> int:
        return EOS_ID

    @property
    def unk_id(self) -> int:
        return UNK_ID

    def train(self, texts: Iterable[str], target_vocab_size: int,
              verbose: bool = False) -> TokenizerInfo:
        """Train BPE merges on texts until vocab reaches target_vocab_size.

        Operates on byte-level pre-tokenized pieces. Each piece's symbols start
        as individual byte-chars; merges combine the most frequent adjacent pair.
        """
        if target_vocab_size < len(self.vocab):
            raise ValueError(f"target_vocab_size {target_vocab_size} < base vocab {len(self.vocab)}")

        # Gather word frequency over pre-tokenized byte-encoded pieces.
        word_freqs: Counter = Counter()
        total_tokens_seen = 0
        for text in texts:
            for piece in GPT2_SPLIT.findall(text):
                encoded = self.byte_encoder.encode(piece)
                # each symbol is a single char in the byte-unicode space
                symbols = tuple(encoded)
                if symbols:
                    word_freqs[symbols] += 1
                    total_tokens_seen += len(symbols)

        # Maintain splits as lists of symbols per unique word.
        words = list(word_freqs.keys())
        splits = {w: list(w) for w in words}

        max_merges = target_vocab_size - len(self.vocab)
        for merge_idx in range(max_merges):
            pair_counts: Counter = Counter()
            for w, freq in word_freqs.items():
                syms = splits[w]
                for i in range(len(syms) - 1):
                    pair_counts[(syms[i], syms[i + 1])] += freq
            if not pair_counts:
                break
            best, best_count = pair_counts.most_common(1)[0]
            if best_count < 2:
                # Only merge pairs occurring at least twice (avoid noise)
                break
            # Apply merge
            new_token = best[0] + best[1]
            self.merges.append([best[0], best[1]])
            self.vocab[new_token] = len(self.vocab)
            self._merges_rank[best] = len(self._merges_rank)
            for w in words:
                syms = splits[w]
                if len(syms) < 2:
                    continue
                i = 0
                merged: list[str] = []
                while i < len(syms):
                    if i < len(syms) - 1 and (syms[i], syms[i + 1]) == best:
                        merged.append(new_token)
                        i += 2
                    else:
                        merged.append(syms[i])
                        i += 1
                splits[w] = merged
            if verbose and (merge_idx + 1) % 50 == 0:
                print(f"  merge {merge_idx + 1}: vocab={len(self.vocab)}")

        self.info = TokenizerInfo(
            version=self.version,
            vocab_size=self.vocab_size,
            num_merges=len(self.merges),
            created_from_tokens=total_tokens_seen,
        )
        self._rebuild_id_map()
        return self.info

    def _bpe(self, word: str) -> list[str]:
        """Apply learned merges to a single pre-tokenized byte-encoded word."""
        symbols = list(word)
        if len(symbols) < 2:
            return symbols
        while True:
            # find best pair by rank
            best_rank = None
            best_idx = -1
            for i in range(len(symbols) - 1):
                rank = self._merges_rank.get((symbols[i], symbols[i + 1]))
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank = rank
                    best_idx = i
            if best_idx < 0:
                break
            new_sym = symbols[best_idx] + symbols[best_idx + 1]
            symbols = symbols[:best_idx] + [new_sym] + symbols[best_idx + 2:]
        return symbols

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        ids: list[int] = []
        if add_bos:
            ids.append(BOS_ID)
        for piece in GPT2_SPLIT.findall(text):
            encoded = self.byte_encoder.encode(piece)
            for sym in self._bpe(encoded):
                tid = self.vocab.get(sym)
                if tid is None:
                    # Fallback: encode each char to its byte token id.
                    for ch in sym:
                        tid = self.vocab.get(ch, UNK_ID)
                        ids.append(tid)
                else:
                    ids.append(tid)
        if add_eos:
            ids.append(EOS_ID)
        return ids

    def decode(self, ids: Iterable[int], skip_special: bool = True) -> str:
        chars: list[str] = []
        for i in ids:
            if skip_special and i in (PAD_ID, BOS_ID, EOS_ID, UNK_ID):
                continue
            # reverse lookup
            tok = self._id_to_token.get(i)
            if tok is None:
                continue
            chars.append(tok)
        text = "".join(chars)
        return self.byte_encoder.decode(text)

    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        return [self.encode(t) for t in texts]

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "vocab": self.vocab,
            "merges": self.merges,
            "info": asdict(self.info),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "BPETokenizer":
        tok = cls(version=data.get("version", "0.1.0"))
        tok.vocab = {k: int(v) for k, v in data["vocab"].items()}
        tok.merges = [list(m) for m in data.get("merges", [])]
        tok._merges_rank = {(a, b): i for i, (a, b) in enumerate(tok.merges)}
        tok._rebuild_id_map()
        info = data.get("info", {})
        tok.info = TokenizerInfo(
            version=tok.version,
            vocab_size=tok.vocab_size,
            num_merges=len(tok.merges),
            created_from_tokens=info.get("created_from_tokens", 0),
        )
        return tok

    def _rebuild_id_map(self) -> None:
        self._id_to_token = {v: k for k, v in self.vocab.items()}

    def save(self, path: str | Path) -> None:
        self._rebuild_id_map()
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        tok = cls.from_dict(data)
        return tok

    def unicode_coverage(self, texts: list[str]) -> dict:
        """Report Unicode block coverage for analysis (esp. Ethiopic)."""
        import unicodedata
        blocks: Counter = Counter()
        for text in texts:
            for ch in text:
                if ch.isspace():
                    continue
                cat = unicodedata.category(ch)
                # crude block detection via name
                try:
                    name = unicodedata.name(ch, "")
                except ValueError:
                    name = ""
                block = "UNKNOWN"
                if "ETHIOPIC" in name:
                    block = "ETHIOPIC"
                elif "LATIN" in name:
                    block = "LATIN"
                elif "ARABIC" in name:
                    block = "ARABIC"
                elif "CJK" in name or "HIRAGANA" in name or "KATAKANA" in name or "HANGUL" in name:
                    block = "CJK_EA"
                elif cat.startswith("N"):
                    block = "NUMBER"
                elif cat.startswith("P"):
                    block = "PUNCT"
                else:
                    block = "OTHER"
                blocks[block] += 1
        return dict(blocks)
