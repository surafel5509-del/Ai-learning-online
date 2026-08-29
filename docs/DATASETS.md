# Datasets

The Training Library is permanent. Originals are never silently destroyed.

## Supported formats

| Format | Parsing |
|--------|---------|
| `.txt` | Split into documents by blank lines / paragraphs |
| `.md` | Treated as text (markdown stripped of noise where possible) |
| `.json` | Each top-level object/array item → one document (`text` field preferred) |
| `.jsonl` | One JSON object per line → one document each |
| `.csv` | A `text` column → one document per row |

## Pipeline

```
Upload / paste / drag-drop
   → Validate (file type allowlist, size limits)
   → Clean (strip control chars, normalize whitespace)
   → Deduplicate (exact-match, with count of removed)
   → Analyze (documents, chars, estimated words, Unicode coverage)
   → Tokenize (active versioned BPE tokenizer)
   → Split train/validation (default 90/10)
   → Count tokens
   → Store binary token file + dataset version metadata
   → Create training plan
   → Train
```

## Dataset versions

Each dataset can have many versions. A version is immutable once created; new uploads
create new versions. This preserves provenance for continual learning and reproducibility.

## Operations

- Upload one file, upload multiple files, drag-and-drop, or paste text.
- Create / rename / inspect / queue / train datasets.
- Pause / resume / cancel training jobs.
- Preview a version's documents.

## Knowledge vs tokens

Tokens ≠ words. The UI displays them separately and labels estimates:
- **tokens processed** — exact count from the tokenizer.
- **estimated words** — heuristic (`tokens × ~0.75`), clearly labelled "estimate".
- **unique vocabulary tokens** — distinct token ids seen.
- **documents** — count of documents.
- **dataset size** — bytes/chars on disk.
- **knowledge categories** — user-assigned category per dataset (Vocabulary, Amharic,
  English, Grammar, General, Technical, Instructions, Conversation, Corrections,
  User-provided).

## Security

Uploaded files are validated by extension allowlist, size-limited, stored outside the web
root, and **never executed**. See [SECURITY.md](SECURITY.md).
