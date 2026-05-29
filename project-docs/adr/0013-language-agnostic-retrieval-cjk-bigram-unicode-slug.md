# Language-agnostic retrieval — CJK bigram + Unicode slug

Phase 16 makes `markdown_kb`'s BM25 pipeline work for non-Latin corpora without partitioning the Section Index or adding a language-specific dependency. Two pure-module changes — a redesigned `tokenize()` and a redesigned `slugify()` — are sufficient.

## Decision

### Tokeniser: character bigram over CJK runs

`tokenize()` (in `markdown_kb/app/indexer.py`) is extended to detect CJK Unified Ideograph runs (U+4E00–U+9FFF, Extension A, Extension B, Compatibility Ideographs). CJK runs are tokenised as **sliding character bigrams** with a **unigram fallback** for length-1 runs (so single-character queries like `錢` are never silently dropped). All other text (Latin, digits, punctuation) continues to use the existing `TOKEN_RE`/`STOP_WORDS` path unchanged.

The new CJK branch only triggers on codepoints > 127; characters ≤ 127 are entirely handled by the legacy path, so pure-ASCII input tokenises byte-identically to the pre-Phase-16 implementation (tagged as an **Invariant** under [Consequences](#consequences)). This guarantees English BM25 scores, `KB_SCORE_THRESHOLD`, and the Phase 8 paraphrase-comparison baseline are unaffected by the change.

### Slug generator: Unicode-preserving

`slugify()` (same module) is extended to preserve Unicode letters — including CJK — verbatim. ASCII letters are lowercased; non-ASCII letters are kept as-is; everything else (punctuation, spaces) collapses to a single hyphen; leading/trailing hyphens are stripped. The `section` fallback fires only when no slug-able character remains after the pass.

This follows the GitHub/Obsidian Unicode anchor convention: `## 退款政策` → `退款政策`, making Citations human-readable and clickable in both platforms. The `-2`/`-3` collision suffix for duplicate slugs is retained unchanged.

The Unicode letter path only activates for codepoints > 127, so pure-ASCII input produces byte-identical output to `re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")` (tagged as an **Invariant** under [Consequences](#consequences)).

### Single mixed Section Index — no per-language partition

The Section Index remains a single BM25 structure over all sources regardless of language.

**Justification:** Chinese-bigram tokens and Latin-word tokens occupy **disjoint** vocabularies. A Chinese query (`退款`) contains no ASCII tokens; an English query (`refund`) contains no CJK tokens. Cross-language retrieval cannot occur accidentally — it would require a token to appear in both vocabularies simultaneously, which the disjoint ranges prevent. The only token overlap arises from ASCII numerals and Latin-script proper nouns embedded in Chinese text (e.g. `iPhone`), which is a **desirable** match and not a false positive.

## Considered Options

### jieba (dictionary-based Chinese segmentation)

Rejected for two reasons:

1. **Simplified-leaning dictionary.** jieba's bundled corpus skews strongly toward Simplified Chinese vocabulary; Traditional Chinese terms are under-represented, producing erratic segmentation for a Traditional corpus (e.g. `退款時限` may be cut in unexpected positions).
2. **Language-specificity conflicts with the language-agnostic goal.** The decision was to support *any* non-Latin script, not only Chinese. jieba is Chinese-only. Selecting it would require a different segmenter for Japanese (MeCab/fugashi), Arabic (Farasa/Stanza), Korean (konlpy), etc. — a combinatorial dependency problem with no clear finish line.

### CKIP (Academia Sinica NLP toolkit)

Rejected on the same language-specificity grounds, plus **model weight size**: CKIP requires downloading transformer weights (~300 MB), making the dev/test environment significantly heavier and non-free for commercial use.

### Character unigrams (1-gram)

Retained only as a **fallback** for length-1 CJK runs (single-character queries). Using unigrams as the primary tokenisation strategy would cause high-IDF inflation for every character and poor precision, because many CJK characters are meaningful only in combination (e.g. `政` alone matches both `政策` and `行政` and `政府`). Bigrams strike the balance: affordable vocabulary size, acceptable precision for short corpus-sized queries, no external dependency.

## Consequences

- **Invariant**: `tokenize()` (`markdown_kb/app/indexer.py`) — pure-ASCII input produces a byte-identical token list to the pre-Phase-16 implementation; the CJK branch fires only on codepoints > 127. English BM25 scores, `KB_SCORE_THRESHOLD`, and the Phase 8 paraphrase-comparison baseline are therefore unaffected. A change that breaks this requires a superseding ADR (§2.5).
- **Invariant**: `slugify()` (same module) — pure-ASCII input produces byte-identical output to `re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")`; the Unicode-letter path activates only on codepoints > 127.
- **Re-index required** after the tokeniser change: the persisted `.kb/index.json` snapshot encodes the token list per Section; existing indexes built with the old ASCII-only `TOKEN_RE` are not compatible with the new bigram tokens for any non-ASCII section. Operators must re-run `POST /index` after upgrading.
- **BM25 segregates languages.** A Chinese query structurally cannot match English-only Sections (disjoint tokens), and vice versa. This is the expected and documented behaviour for the Wiki/BM25 stack. Vector RAG blends languages naturally via its multilingual embedding space — that contrast is a deliberate architectural talking point (see Phase 8 paraphrase comparison).
- **Chinese stop-word handling is deferred.** BM25 IDF already down-weights high-frequency tokens; bigram tokenisation produces different high-frequency tokens than unigram would. Function-word noise should be assessed from real Chinese usage before committing to a stop set. Tracking issue deferred.
- **Encoding safety is unchanged.** `Import` decodes raw bytes as strict UTF-8; a non-UTF-8 file (e.g. Big5/cp950) raises `UnicodeDecodeError` per-file and leaves the batch unaffected. Auto-detection/transcoding is out of scope.
